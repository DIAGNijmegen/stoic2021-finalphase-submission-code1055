import os
import logging
import torch
from segmentation.segtrain import run, set_deterministic
from segmentation.segconfig.segconfig import BaseConfig
import argparse
from segmentation.segconfig.segmodelconfig import get_model, get_modelconfig
from copy import deepcopy


logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Train STOIC network')
    # general
    parser.add_argument('--gpu',
                        default="6",
                        help='gpu id',
                        type=str)
    parser.add_argument('--nick',
                        default='',
                        help='Prepend a nickname to the output directory',
                        type=str)
    args = parser.parse_args()
    args.mode = 1 #onlySupervised
    args.model = 'upernext'
    args.cv = False
    args.imsize = '256'
    return args


def main():
    print('PID:', os.getpid())
    torch.set_num_threads(1)
    args_orig = parse_args()

    print('### pretraining of convnextMicro ###')
    set_deterministic()
    args = deepcopy(args_orig)
    args.mode = 1
    args.nick = ''.join(args.nick)
    args.imsize = '256'
    config = BaseConfig(args)
    config.UNSUPERVISED_DATA = 'tcia'
    config.Batch_SIZE = 8
    config.MAX_EPOCHS = 50
    config.STARTEPOCH_UNSUPERVISED = 20
    config.NUM_UNSUP_STEPS = 1
    config.DO_NORMALIZATION = False
    config.SUPERVISED_AUGMENTATIONS = False
    config.modelconfig = get_modelconfig(config)
    config.modelconfig.use_transformer = False
    config.modelconfig.size = 'micro'
    run(config, args)

    print('### pretraining of convnextransformerMicro ###')
    set_deterministic()
    args = deepcopy(args_orig)
    args.mode = 1
    args.nick = ''.join(args.nick)
    args.imsize = '256'
    config = BaseConfig(args)
    config.UNSUPERVISED_DATA = 'tcia'
    config.Batch_SIZE = 8
    config.MAX_EPOCHS = 50
    config.STARTEPOCH_UNSUPERVISED = 20
    config.NUM_UNSUP_STEPS = 1
    config.DO_NORMALIZATION = False
    config.SUPERVISED_AUGMENTATIONS = False
    config.modelconfig = get_modelconfig(config)
    config.modelconfig.use_transformer = True
    config.modelconfig.size = 'micro'
    run(config, args)

if __name__ == '__main__':
    main()