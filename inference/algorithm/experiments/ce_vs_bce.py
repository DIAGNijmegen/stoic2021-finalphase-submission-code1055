from config.config import BaseConfig
from training import run
from misc_utilities.determinism import set_deterministic
import os
import logging


def ce():
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    logging.getLogger().setLevel(logging.INFO)

    config = BaseConfig(split="cv5_infonly_bal.csv", cv_enabled=True, num_steps=1, data_name="gpudeformed", nickname="use_ce")

    config.modelconfig.loss_fn = 'ce_sev'

    config.modelconfig.pretrained_mode = 'segmentation'
    config.modelconfig.pos_weight = 1.0
    # Parameters from the ConvNeXt paper
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = {
        "lr_min" : 1e-6
    }
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 25
    config.modelconfig.decay_lr_until = 20
    config.dataconfigs['train'].do_normalization = True
    config.dataconfigs['val'].do_normalization = True
    config.modelconfig.size = 'tiny'

    set_deterministic()

    run(config, reset_deterministic=True)


def bce():
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    logging.getLogger().setLevel(logging.INFO)

    config = BaseConfig(split="cv5_infonly_bal.csv", cv_enabled=True, num_steps=1, data_name="gpudeformed", nickname="use_bce")

    config.modelconfig.loss_fn = 'bce_sev'

    config.modelconfig.pretrained_mode = 'segmentation'
    config.modelconfig.pos_weight = 1.0
    # Parameters from the ConvNeXt paper
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = {
        "lr_min" : 1e-6
    }
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 25
    config.modelconfig.decay_lr_until = 20
    config.dataconfigs['train'].do_normalization = True
    config.dataconfigs['val'].do_normalization = True
    config.modelconfig.size = 'tiny'

    set_deterministic()

    run(config, reset_deterministic=True)


if __name__ == "__main__":
    #ce()
    bce()