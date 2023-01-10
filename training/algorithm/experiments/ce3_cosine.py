# experiments to test CrossEntropy 3
from argparse import ArgumentParser
import os
from config.config import BaseConfig
from training import multirun


def get_config(nickname):
    config = BaseConfig(split="cv5_all_bal.csv", nickname=nickname, cv_enabled=True)
    config.modelconfig.loss_fn = "3ce"
    config.modelconfig.num_classes = 3
    config.MAX_EPOCHS = 40
    return config


parser = ArgumentParser()
parser.add_argument("gpu", type=int)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

cosine_config = get_config("ce3-cosine")
cosine_config.modelconfig.cosine_decay = dict(lr_min=1e-8)
baseline_config = get_config("ce3-baseline")
moving_config = get_config("ce3-moving")
moving_config.modelconfig.ce3_moving_epoch = 24

multirun([cosine_config, baseline_config], no_tqdm=True, reset_deterministic=True)
