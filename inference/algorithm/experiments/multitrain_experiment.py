from segmentation.segconfig.segconfig import MultiConfig
from segmentation.multitrain import start_experiment
from misc_utilities.determinism import set_deterministic
import os
import logging
from argparse import ArgumentParser
from segmentation.segconfig.segmodelconfig import get_modelconfig


def stoic_tcia_mosmed_hust():
    set_deterministic(1055)
    gpu_num = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    args = parser.parse_args()
    args.gpu = str(gpu_num)
    args.cv = True
    args.imsize = str(256)
    args.model = 'multinext'

    args.nick = 'multitrain_experiment_StoicTciaMosmedHust'

    config = MultiConfig(args=args, split="cv5_infonly_bal.csv", cv_enabled=args.cv)
    config.modelconfig = get_modelconfig(config)
    config.change_loss_weight = True
    config.LOSS_CLS = 'ce'
    config.modelconfig.size = 'tiny'
    config.DO_NORMALIZATION = True
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 25

    config.datasets = ['stoic', 'tcia', 'mosmed', 'hust']
    config.dataset_weights = [1 for i in range(len(config.datasets))]

    start_experiment(config, args)


def stoic_tcia_hust():
    set_deterministic(1055)
    gpu_num = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    args = parser.parse_args()
    args.gpu = str(gpu_num)
    args.cv = True
    args.imsize = str(256)
    args.model = 'multinext'

    args.nick = 'multitrain_experiment_StoicTciaHust'

    config = MultiConfig(args=args, split="cv5_infonly_bal.csv", cv_enabled=args.cv)
    config.modelconfig = get_modelconfig(config)
    config.change_loss_weight = True
    config.LOSS_CLS = 'ce'
    config.modelconfig.size = 'tiny'
    config.DO_NORMALIZATION = True
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 25

    config.datasets = ['stoic', 'tcia', 'hust']
    config.dataset_weights = [1 for i in range(len(config.datasets))]

    start_experiment(config, args)


def stoic_tcia():
    set_deterministic(1055)
    gpu_num = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    args = parser.parse_args()
    args.gpu = str(gpu_num)
    args.cv = True
    args.imsize = str(256)
    args.model = 'multinext'

    args.nick = 'multitrain_experiment_StoicTcia'

    config = MultiConfig(args=args, split="cv5_infonly_bal.csv", cv_enabled=args.cv)
    config.modelconfig = get_modelconfig(config)
    config.change_loss_weight = True
    config.LOSS_CLS = 'ce'
    config.modelconfig.size = 'tiny'
    config.DO_NORMALIZATION = True
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 25

    config.datasets = ['stoic', 'tcia']
    config.dataset_weights = [1 for i in range(len(config.datasets))]

    start_experiment(config, args)



def stoic_mosmed():
    set_deterministic(1055)
    gpu_num = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    args = parser.parse_args()
    args.gpu = str(gpu_num)
    args.cv = True
    args.imsize = str(256)
    args.model = 'multinext'

    args.nick = 'multitrain_experiment_StoicMosmed_seginit'

    config = MultiConfig(args=args, split="cv5_infonly_bal.csv", cv_enabled=args.cv)
    config.modelconfig = get_modelconfig(config)
    config.modelconfig.pretrained_mode = 'segmentation'
    config.change_loss_weight = True
    config.LOSS_CLS = 'ce'
    config.modelconfig.size = 'tiny'
    config.DO_NORMALIZATION = True
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 25

    config.datasets = ['stoic', 'mosmed']
    config.dataset_weights = [1 for i in range(len(config.datasets))]

    start_experiment(config, args)









if __name__ == "__main__":
    stoic_tcia_mosmed_hust()
    #stoic_tcia_hust()
    #stoic_tcia()
    #stoic_mosmed()