from config.config import BaseConfig
from training import run_config
from misc_utilities.determinism import set_deterministic
import os
import logging



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logging.getLogger().setLevel(logging.INFO)


def get_config():
    config = BaseConfig(split="split_sev_cv5.csv", cv_enabled=True, num_steps=2)
    return config

set_deterministic()
config = get_config()
config.modelconfig.pos_weight = 1.0
config.Batch_SIZE = 4
run_config(config, nickname="pos_weight_1")

set_deterministic()
config = get_config()
config.modelconfig.pos_weight = 2.0
config.Batch_SIZE = 4
run_config(config, nickname="pos_weight_2")

set_deterministic()
config = get_config()
config.modelconfig.pos_weight = 3.0
config.Batch_SIZE = 4
run_config(config, nickname="pos_weight_3")
