# test with and without babysitting
from argparse import ArgumentParser
import logging
import os
from training import run
from config.config import BaseConfig
from misc_utilities import determinism

logging.getLogger().setLevel(logging.INFO)

parser = ArgumentParser()
parser.add_argument("gpu", type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

def get_config(nickname):
    config = BaseConfig(nickname=nickname)
    config.MAX_EPOCHS = 60
    return config

print("without babysitting")
determinism.set_deterministic()
config = get_config("without_babysitting")
config.babysitter_decay = None
config.babysitter_grace_steps = None
run(config)

print("with babysitting")
determinism.set_deterministic()
config = get_config("with_babysitting")
config.babysitter_decay = 10
config.babysitter_grace_steps = 8
run(config)
