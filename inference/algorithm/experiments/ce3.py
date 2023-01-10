# experiments to test CrossEntropy 3
from argparse import ArgumentParser
import os
from config.config import BaseConfig
from training import run
from misc_utilities import determinism


def get_config(split, nickname):
    config = BaseConfig(split=split, nickname=nickname, cv_enabled=True)
    config.modelconfig.loss_fn = "3ce"
    config.modelconfig.num_classes = 3
    config.MAX_EPOCHS = 25
    return config


parser = ArgumentParser()
parser.add_argument("gpu", type=int)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

for balance, weighted in [[True, False], [False, True], [True, True], [False, False]]:
    split = "cv5_all_bal.csv" if balance else "cv5.csv"
    nickname = "-".join(
        ["ce3", "weighted" if weighted else "normal", "bal" if balance else "unbal"]
    )
    print("###", nickname, "###")
    determinism.set_deterministic()
    config = get_config(split, nickname)
    config.modelconfig.ce3_weighted = weighted
    run(config)
