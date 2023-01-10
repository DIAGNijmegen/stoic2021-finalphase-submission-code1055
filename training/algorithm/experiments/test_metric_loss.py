from config.config import BaseConfig
from training import run
from misc_utilities.determinism import set_deterministic
import os
import logging



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logging.getLogger().setLevel(logging.INFO)


def get_config():
    config = BaseConfig(split="cv5_infonly_bal.csv", cv_enabled=True, num_steps=2, nickname="run_metric_loss")
    return config

set_deterministic()
config = get_config()

# Parameters from the ConvNeXt paper
config.modelconfig.learning_rate = 5e-5

config.modelconfig.cosine_decay = {
    "lr_min" : 1e-6
}

config.modelconfig.pos_weight = 1.0
config.Batch_SIZE = 4
config.modelconfig.loss_fn = "auc_bce_sev"
config.modelconfig.siamese = True

run(config)

