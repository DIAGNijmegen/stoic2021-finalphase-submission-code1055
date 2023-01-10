import os
from argparse import ArgumentParser
import logging
from config.config import BaseConfig
from training import run_config
from misc_utilities.determinism import set_deterministic

parser = ArgumentParser()
parser.add_argument("gpu", type=int)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

def get_config():
    config = BaseConfig(split="split_sev_cv5.csv", num_steps=2, cv_enabled=True)
    config.modelconfig.siamese = True
    config.MAX_EPOCHS = 30
    return config

logging.getLogger().setLevel(logging.INFO)

print("### AUC ###")
set_deterministic()
config = get_config()
config.modelconfig.loss_fn = "auc"
run_config(config, nickname="auc")

print("### Batch AUC ###")
set_deterministic()
config = get_config()
config.modelconfig.loss_fn = "batch_auc"
run_config(config, nickname="batch_auc")

print("### AUC l2 ###")
set_deterministic()
config = get_config()
config.modelconfig.loss_fn = "auc_l2"
run_config(config, nickname="auc-l2")
