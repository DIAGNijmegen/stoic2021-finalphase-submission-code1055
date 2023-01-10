import os
from argparse import ArgumentParser
import logging
from config.config import BaseConfig
from training import run_config
from misc_utilities.determinism import set_deterministic


def get_config():
    config = BaseConfig(split="split_sev_cv5.csv", num_steps=2, cv_enabled=True)
    config.modelconfig.siamese = False
    config.MAX_EPOCHS = 30
    return config


def main():
    print('PID:', os.getpid())
    parser = ArgumentParser()
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    logging.getLogger().setLevel(logging.INFO)

    print("### Tiny with drop path ###")
    set_deterministic()
    config = get_config()
    config.modelconfig.loss_fn = "bce_sev"
    config.modelconfig.use_transformer = False
    config.modelconfig.pos_weight = None
    config.modelconfig.drop_path = 0.1
    config.modelconfig.size = 'tiny'
    run_config(config, nickname="dropPath01_tiny")

    print("### Tiny without drop path ###")
    set_deterministic()
    config = get_config()
    config.modelconfig.loss_fn = "bce_sev"
    config.modelconfig.use_transformer = False
    config.modelconfig.pos_weight = None
    config.modelconfig.drop_path = 0.0
    config.modelconfig.size = 'tiny'
    run_config(config, nickname="dropPath00_tiny")

    print("### Small with drop path ###")
    set_deterministic()
    config = get_config()
    config.modelconfig.loss_fn = "bce_sev"
    config.modelconfig.use_transformer = False
    config.modelconfig.pos_weight = None
    config.modelconfig.drop_path = 0.4
    config.modelconfig.size = 'small'
    run_config(config, nickname="dropPath04_small")

    print("### Small without drop path ###")
    set_deterministic()
    config = get_config()
    config.modelconfig.loss_fn = "bce_sev"
    config.modelconfig.use_transformer = False
    config.modelconfig.pos_weight = None
    config.modelconfig.drop_path = 0.0
    config.modelconfig.size = 'small'
    run_config(config, nickname="dropPath00_small")




if __name__ == '__main__':
    main()