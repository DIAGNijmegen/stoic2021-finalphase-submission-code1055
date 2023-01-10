import os
import logging
from training import Trainer
from config.config import BaseConfig
from misc_utilities import determinism

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
logging.getLogger().setLevel(logging.INFO)


def get_config():
    config = BaseConfig(num_steps=2)
    config.Batch_SIZE = 16
    config.MAX_EPOCHS = 30
    return config

# sev only
print("### sevonly ###")
determinism.set_deterministic()
config = get_config()
config.modelconfig.loss_fn = "bce_sev"
config.modelconfig.loss_max_sev_ratio_epoch = None
Trainer(config, nickname="sevonly").run()

# inf sev exclude with moving ratio
print("### infsev-exclude with moving ratio ###")
determinism.set_deterministic()
config = get_config()
config.modelconfig.loss_fn = "bce_inf_sev_exclude"
config.modelconfig.loss_max_sev_ratio_epoch = 10
Trainer(config, nickname="infsev-exclude-moving").run()

# inf sev with moving ratio
print("### infsev-orig with moving ratio ###")
determinism.set_deterministic()
config = get_config()
config.modelconfig.loss_fn = "bce_inf_sev"
config.modelconfig.loss_max_sev_ratio_epoch = 10
Trainer(config, nickname="infsev-orig-moving").run()

# inf sev exclude with constant ratio
print("### infsev-exclude with constant ratio ###")
determinism.set_deterministic()
config = get_config()
config.modelconfig.loss_fn = "bce_inf_sev_exclude"
config.modelconfig.loss_max_sev_ratio_epoch = None
Trainer(config, nickname="infsev-exclude-constant").run()

# inf sev with constant ratio
print("### infsev-orig with constant ratio ###")
determinism.set_deterministic()
config = get_config()
config.modelconfig.loss_fn = "bce_inf_sev"
config.modelconfig.loss_max_sev_ratio_epoch = None
Trainer(config, nickname="infsev-orig-constant")
