from config.config import BaseConfig
from training import run_config
from misc_utilities.determinism import set_deterministic
import os
import logging



os.environ["CUDA_VISIBLE_DEVICES"] = "3"
logging.getLogger().setLevel(logging.INFO)


def get_config():
    config = BaseConfig(split="split_sev_cv5.csv", cv_enabled=True, num_steps=2)
    return config

set_deterministic()
config = get_config()
config.modelconfig.pos_weight = 1.0
config.modelconfig.lr_decay = {
    "gamma" : 0.5,
    "every_num_epochs" : 6
}
config.modelconfig.cosine_decay = None
config.Batch_SIZE = 4
run_config(config, nickname="halving_every_6_epochs")


set_deterministic()
config = get_config()
config.modelconfig.pos_weight = 1.0
config.modelconfig.lr_decay = None
config.modelconfig.cosine_decay = {
    "lr_min" : 1e-8
}
config.Batch_SIZE = 4
run_config(config, nickname="cosine_decay_1e-5_1e-8")

