from typing import Callable
from argparse import ArgumentParser
import os
import json
import logging
from pathlib import Path
from ray import tune, init as ray_init
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler

from config.config import BaseConfig
from training import Trainer


class TrainableWrapper(tune.Trainable):
    def setup(self,
        config,
        update_config: Callable[[BaseConfig, dict], BaseConfig],
        datasets=None, dataconfigs=None):
        # modify config based on hyperparameters
        config_instance = update_config(BaseConfig("convnext"), config)
        self.trainer = Trainer(
            config=config_instance,
            log_directory=Path(self.logdir),
            datasets=datasets,
            dataconfigs=dataconfigs,
            # when doing hyper parameter search, the console is cluttered enough
            no_tqdm=True,
        )
        self.epoch_num = 0
        self.best_metrics = None

    def step(self):
        metrics = self.trainer.epoch(self.epoch_num)
        self.epoch_num += 1

        # keep track of best metrics as well
        # note that if training is resumed, the best metrics will be lost
        if self.best_metrics is None or metrics["val_loss"] < self.best_metrics["val_loss"]:
            self.best_metrics = metrics

        bestm = {f"best_{name}": value for name, value in self.best_metrics.items()}
        return { **metrics, **bestm }

    def save_checkpoint(self, tmp_checkpoint_dir):
        self.trainer.save_checkpoint(tmp_checkpoint_dir)

    def load_checkpoint(self, tmp_checkpoint_dir):
        assert isinstance(tmp_checkpoint_dir, str), "Checkpoint dir is not a string"
        return self.trainer.load_checkpoint(tmp_checkpoint_dir)


def main(num_samples: int, max_pending_trials: int = 30, name=None, fail_fast=False):
    logging.getLogger().setLevel(logging.INFO)

    # make sure that CUDA_VISIBLE_DEVICES is set correctly
    logging.info(f"Visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")

    params = {
        "lr": tune.loguniform(1e-6, 5 * 1e-4),
        "weight_decay": tune.loguniform(1e-10, 1e-2),
        "use_transformer": False,
        "use_metadata": True,
        "pretrained_mode": "segmentation",
        "optim": "adamw",
    }

    def update_config(config: BaseConfig, update: dict):
        config.modelconfig.learning_rate = update["lr"]
        config.modelconfig.weight_decay = update["weight_decay"]
        config.modelconfig.optim_name = update["optim"]
        config.modelconfig.use_transformer = update["use_transformer"]
        config.modelconfig.use_metadata = update["use_metadata"]
        config.modelconfig.pretrained_mode = update["pretrained_mode"]
        config.use_ema = False

        return config

    scheduler = AsyncHyperBandScheduler(
        metric="val_loss",
        mode="min",
        # TODO: get max epochs from the config
        max_t=30,
        grace_period=8,
        reduction_factor=12,
    )
    reporter = CLIReporter(
        metric_columns=["val_loss", "auc_inf", "auc_sev", "auc_sev2", "training_iteration"],
    )

    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(max_pending_trials)

    ray_init(dashboard_port=36007)

    result = tune.run(
        tune.with_parameters(
            TrainableWrapper,
            update_config=update_config,
            datasets=None,
            dataconfigs=None,
        ),
        name=name,
        config=params,
        num_samples=num_samples,
        resources_per_trial=dict(cpu=8, gpu=1),
        scheduler=scheduler,
        progress_reporter=reporter,
        fail_fast=fail_fast,
    )

    best_trial = result.get_best_trial("val_loss", "min", "all")
    print("===== Best Trial =====")
    print(best_trial)
    print("Config:")
    print(json.dumps(best_trial.config, indent=2))
    print("Final metrics:")
    print(json.dumps(best_trial.last_result, indent=2))

    logging.info("done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--pending", type=int, default=30)
    parser.add_argument("--fail-fast", action="store_true", dest="fail_fast", help="If an error is raised, stop everything")
    parser.set_defaults(fail_fast=False)
    args = parser.parse_args()

    main(num_samples=args.samples, max_pending_trials=args.pending, name=args.name, fail_fast=args.fail_fast)
