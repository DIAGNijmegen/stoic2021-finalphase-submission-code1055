# experiments to test CrossEntropy 3
from argparse import ArgumentParser
import os
import logging
from config.config import BaseConfig
from training import run
from misc_utilities import determinism


def get_config(split, nickname):
    config = BaseConfig(split=split, nickname=nickname, cv_enabled=True)
    config.modelconfig.loss_fn = "3ce"
    config.modelconfig.num_classes = 3
    config.MAX_EPOCHS = 25
    config.dataconfigs["train"].cache_img_size = 128
    config.dataconfigs["train"].img_size = 112
    config.dataconfigs["val"].cache_img_size = 128
    config.dataconfigs["val"].img_size = 112
    return config


parser = ArgumentParser()
parser.add_argument("gpu", type=int)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
logging.getLogger().setLevel(logging.INFO)

for balance, weighted in [[True, False], [False, True], [True, True], [False, False]]:
    split = "cv5_all_bal.csv" if balance else "cv5.csv"
    nickname = "-".join(
        ["ce3", "weighted" if weighted else "normal", "bal" if balance else "unbal", "smalldata"]
    )
    print("###", nickname, "###")
    determinism.set_deterministic()
    config = get_config(split, nickname)
    config.modelconfig.ce3_weighted = weighted
    run(config)
