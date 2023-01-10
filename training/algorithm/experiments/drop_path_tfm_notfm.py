import logging
from config.config import BaseConfig
from training import multirun
from experiments.gpucli import parse_gpu

def get_config(use_transformer):
    config = BaseConfig(nickname=f"dropath-{'tfm' if use_transformer else 'notfm'}", cv_enabled=True)
    config.modelconfig.loss_fn = "bce_sev"
    config.modelconfig.drop_path = 0.1
    config.Batch_SIZE = 4
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = dict(lr_min=1e-6)
    config.modelconfig.use_transformer = use_transformer
    return config

parse_gpu()
logging.getLogger().setLevel(logging.INFO)
multirun([get_config(False), get_config(True)], no_tqdm=True, reset_deterministic=True, reverse_cv=False)
