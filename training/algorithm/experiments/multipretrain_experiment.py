from segmentation.segconfig.segconfig import MultiConfig
from segmentation.multitrain import start_experiment
from misc_utilities.determinism import set_deterministic
import os
import logging
from argparse import ArgumentParser
from segmentation.segconfig.segmodelconfig import get_modelconfig


def tcia_mosmed_hust():
    gpu_num = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    args = parser.parse_args()
    args.gpu = str(gpu_num)
    args.cv = False
    args.imsize = str(256)
    args.model = 'multinext'

    args.nick = 'multipretrain_experiment_TciaMosmedHust'

    config = MultiConfig(args=args, split="cv5_infonly_bal.csv", cv_enabled=args.cv)
    config.modelconfig = get_modelconfig(config)
    config.change_loss_weight = False
    config.LOSS_CLS = 'bce'
    config.modelconfig.size = 'tiny'
    config.DO_NORMALIZATION = True
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 150

    config.datasets = ['tcia', 'mosmed', 'hust']
    config.dataset_weights = [1 for i in range(len(config.datasets))]

    start_experiment(config, args)

def tcia_hust():
    gpu_num = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    args = parser.parse_args()
    args.gpu = str(gpu_num)
    args.cv = False
    args.imsize = str(256)
    args.model = 'multinext'

    args.nick = 'multipretrain_experiment_TciaHust'

    config = MultiConfig(args=args, split="cv5_infonly_bal.csv", cv_enabled=args.cv)
    config.modelconfig = get_modelconfig(config)
    config.change_loss_weight = False
    config.LOSS_CLS = 'bce'
    config.modelconfig.size = 'tiny'
    config.DO_NORMALIZATION = True
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 150

    config.datasets = ['tcia', 'hust']
    config.dataset_weights = [1 for i in range(len(config.datasets))]

    start_experiment(config, args)

def tcia_mosmed():
    gpu_num = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    args = parser.parse_args()
    args.gpu = str(gpu_num)
    args.cv = False
    args.imsize = str(256)
    args.model = 'multinext'

    args.nick = 'multipretrain_experiment_TciaMosmed'

    config = MultiConfig(args=args, split="cv5_infonly_bal.csv", cv_enabled=args.cv)
    config.modelconfig = get_modelconfig(config)
    config.change_loss_weight = False
    config.LOSS_CLS = 'bce'
    config.modelconfig.size = 'tiny'
    config.DO_NORMALIZATION = True
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 150

    config.datasets = ['tcia', 'mosmed']
    config.dataset_weights = [1 for i in range(len(config.datasets))]

    start_experiment(config, args)


if __name__ == "__main__":
    tcia_mosmed_hust()
    tcia_hust()
    tcia_mosmed()

