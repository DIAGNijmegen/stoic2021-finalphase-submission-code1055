import os
from argparse import ArgumentParser
import logging
from config.config import BaseConfig
from training import run
from misc_utilities.determinism import set_deterministic

parser = ArgumentParser()
parser.add_argument("gpu", type=int)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


def get_config(split, nickname):
    config = BaseConfig(split=split, num_steps=2, cv_enabled=True, nickname=nickname)
    config.modelconfig.siamese = False
    config.MAX_EPOCHS = 30
    return config


logging.getLogger().setLevel(logging.INFO)

print("### infsev moving ###")
set_deterministic()
config = get_config("cv5_all_bal.csv", "infsev-moving")
config.modelconfig.loss_fn = "bce_inf_sev"
config.modelconfig.loss_max_sev_ratio_epoch = 10
run(config)

print("### baseline all ###")
set_deterministic()
config = get_config("cv5_all_bal.csv", "bce_sev-all")
config.modelconfig.loss_fn = "bce_sev"
run(config)

print("### baseline ###")
set_deterministic()
config = get_config("cv5_infonly_bal.csv", "bce_sev-baeline")
config.modelconfig.loss_fn = "bce_sev"
run(config)

print("### infsev const ###")
set_deterministic()
config = get_config("cv5_all_bal.csv", "infsev-const")
config.modelconfig.loss_fn = "bce_inf_sev"
run(config)
