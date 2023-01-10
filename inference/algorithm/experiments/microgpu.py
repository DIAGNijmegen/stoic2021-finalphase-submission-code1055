import logging
from experiments.gpucli import parse_gpu
from config.config import BaseConfig
from training import run

logging.getLogger().setLevel(logging.INFO)

parse_gpu()
config = BaseConfig(nickname="microgpu", cv_enabled=True, data_name="gpudeformed")
config.MAX_EPOCHS = 40
config.modelconfig.size = "micro"
run(config, reset_deterministic=True, no_tqdm=True)
