from config.config import BaseConfig
from training import run_config
from misc_utilities.determinism import set_deterministic
import os
import logging



os.environ["CUDA_VISIBLE_DEVICES"] = "3"
logging.getLogger().setLevel(logging.INFO)


def get_config():
    config = BaseConfig(split="sevbal_cv5.csv", cv_enabled=True, num_steps=2)
    return config

set_deterministic()
config = get_config()
config.modelconfig.pos_weight = 1.0
config.Batch_SIZE = 4
run_config(config, nickname="sevbal_cv5")

