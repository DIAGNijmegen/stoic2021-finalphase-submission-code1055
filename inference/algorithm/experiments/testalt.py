import logging
from experiments.gpucli import parse_gpu
from config.config import BaseConfig
from training import run_cv_alternating

parse_gpu()

config = BaseConfig(nickname="test_alternating", cv_enabled=True)
config.MAX_EPOCHS = 40
config.modelconfig.size = "micro"
run_cv_alternating(config, no_tqdm=True)
logging.info("done")