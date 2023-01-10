# run using:
# python -m experiments.train_two_stages

import logging
import os
from pathlib import Path
import pandas as pd

proj_root = Path(__file__).parent.parent

from config.config import BaseConfig
from data.split import get_split_dataconfigs
from training import parse_args, Trainer


def stage_inf(config: BaseConfig, split_df: pd.DataFrame):
    config.dataconfigs = get_split_dataconfigs(config, split_df)
    config.modelconfig.loss_fn = "inf_only"

    trainer_inf = Trainer(config)
    split_df.to_csv(trainer_inf.output_dir/"split.csv", index=False)
    trainer_inf.run()
    return trainer_inf


def stage_sev(config: BaseConfig, trainer_inf: Trainer, split_df: pd.DataFrame):
    reference = pd.read_csv(Path(config.DATA_PATH) / "metadata" / "reference.csv").set_index("PatientID")
    # discard all non-covid cases
    split_df = split_df.set_index("PatientID")
    split_df["inf"] = reference["probCOVID"]
    split_df["sev"] = reference["probSevere"]
    split_df = split_df[split_df["inf"] == 1]

    sevs = split_df[split_df["sev"] == 1]
    split_df = pd.concat([split_df, sevs, sevs])["set"].reset_index()

    config.modelconfig.loss_fn = "sev_only"
    config.dataconfigs = get_split_dataconfigs(config, split_df)

    sev_dir = trainer_inf.output_dir / "stage_sev"
    sev_dir.mkdir(exist_ok=True)
    trainer_sev = Trainer(config, log_directory=sev_dir)
    split_df.to_csv(trainer_sev.output_dir/"split.csv", index=False)
    # model is initialized with new weights
    fresh_head = trainer_sev.model.main_model.head
    # load model from previous run and give it a new head
    trainer_sev.load_checkpoint(trainer_inf.output_dir, load_best=True)
    trainer_sev.model.main_model.head = fresh_head
    trainer_sev.run()
    return trainer_sev


def main():
    try:
        import yatlogger
        yatlogger.register(session_name="Two Stage")
    except ModuleNotFoundError:
        pass

    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print("Use GPU: {} for training".format(args.gpu))

    logging.getLogger().setLevel(logging.INFO)

    config = BaseConfig(args.model)
    config.LOGS_PATH = str(Path(config.LOGS_PATH) / "two_stage")

    split_df = pd.read_csv(proj_root/"data"/"split.csv")

    trainer_inf = stage_inf(config, split_df)
    logging.info("trainer_inf completed")

    stage_sev(config, trainer_inf, split_df)
    logging.info("trainer_sev completed")


if __name__ == "__main__":
    main()
