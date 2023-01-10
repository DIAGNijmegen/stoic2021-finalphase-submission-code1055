import os
from argparse import ArgumentParser
import logging
from config.config import BaseConfig
from training import run_config
from misc_utilities.determinism import set_deterministic


def get_config():
    config = BaseConfig(split="split_sev_cv5.csv", num_steps=2, cv_enabled=True)
    config.modelconfig.siamese = False
    return config


def main():
    print('PID:', os.getpid())
    parser = ArgumentParser()
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    logging.getLogger().setLevel(logging.INFO)

    print("### Transformer as last stage ###")
    set_deterministic()
    config = get_config()
    config.modelconfig.loss_fn = "bce_sev"
    config.modelconfig.use_transformer = True
    config.modelconfig.pos_weight = None
    run_config(config, nickname="transVsConv_transformer")

    print("### Convolution as last stage ###")
    set_deterministic()
    config = get_config()
    config.modelconfig.loss_fn = "bce_sev"
    config.modelconfig.use_transformer = False
    config.modelconfig.pos_weight = None
    run_config(config, nickname="transVsConv_convolution")


if __name__ == '__main__':
    main()