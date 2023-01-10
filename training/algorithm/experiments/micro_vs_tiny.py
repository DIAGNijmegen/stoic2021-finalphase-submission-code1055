from argparse import ArgumentParser
import os
from config.config import BaseConfig
from training import run
from misc_utilities.determinism import set_deterministic


def get_config(nickname):
    config = BaseConfig(split="cv5_infonly_bal.csv", num_steps=1, nickname=nickname, cv_enabled=True)
    config.modelconfig.siamese = False
    config.MAX_EPOCHS = 30
    return config

def main():
    print('PID:', os.getpid())
    parser = ArgumentParser()
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # print('### convnextMicro drop_path=0.1')
    # set_deterministic()
    # config = get_config("microVsTiny_convnextMicro_01")
    # config.modelconfig.cosine_decay = dict(lr_min=1e-6)
    # config.modelconfig.loss_fn = "bce_sev"
    # config.modelconfig.use_transformer = False
    # config.modelconfig.pos_weight = None
    # config.modelconfig.drop_path = 0.1
    # config.modelconfig.size = 'micro'
    # config.modelconfig.learning_rate = 5e-5
    # config.modelconfig.cosine_decay = {"lr_min": 1e-6}
    # config.modelconfig.pos_weight = 1.0
    # config.Batch_SIZE = 4
    # run(config, reset_deterministic=True)
    #
    # print('### convnextransformerMicro drop_path=0.1')
    # set_deterministic()
    # config = get_config("microVsTiny_convnextransformerMicro_01")
    # config.modelconfig.cosine_decay = dict(lr_min=1e-6)
    # config.modelconfig.loss_fn = "bce_sev"
    # config.modelconfig.use_transformer = True
    # config.modelconfig.pos_weight = None
    # config.modelconfig.drop_path = 0.1
    # config.modelconfig.size = 'micro'
    # config.modelconfig.learning_rate = 5e-5
    # config.modelconfig.cosine_decay = {"lr_min": 1e-6}
    # config.modelconfig.pos_weight = 1.0
    # config.Batch_SIZE = 4
    # run(config, reset_deterministic=True)
    #
    # print('### convnextTiny droppath=0.1')
    # set_deterministic()
    # config = get_config("microVsTiny_convnextTiny_01")
    # config.modelconfig.cosine_decay = dict(lr_min=1e-6)
    # config.modelconfig.loss_fn = "bce_sev"
    # config.modelconfig.use_transformer = False
    # config.modelconfig.pos_weight = None
    # config.modelconfig.drop_path = 0.1
    # config.modelconfig.size = 'tiny'
    # config.modelconfig.learning_rate = 5e-5
    # config.modelconfig.cosine_decay = {"lr_min": 1e-6}
    # config.modelconfig.pos_weight = 1.0
    # config.Batch_SIZE = 4
    # run(config, reset_deterministic=True)
    #
    # print('### convnextransformerTiny droppath=0.1')
    # set_deterministic()
    # config = get_config("microVsTiny_convnextransformerTiny_01")
    # config.modelconfig.cosine_decay = dict(lr_min=1e-6)
    # config.modelconfig.loss_fn = "bce_sev"
    # config.modelconfig.use_transformer = True
    # config.modelconfig.pos_weight = None
    # config.modelconfig.drop_path = 0.1
    # config.modelconfig.size = 'tiny'
    # config.modelconfig.learning_rate = 5e-5
    # config.modelconfig.cosine_decay = {"lr_min": 1e-6}
    # config.modelconfig.pos_weight = 1.0
    # config.Batch_SIZE = 4
    # run(config, reset_deterministic=True)

    print('### convnextMicro drop_path=0.0')
    set_deterministic()
    config = get_config("microVsTiny_convnextMicro_00")
    config.modelconfig.cosine_decay = dict(lr_min=1e-6)
    config.modelconfig.loss_fn = "bce_sev"
    config.modelconfig.use_transformer = False
    config.modelconfig.pos_weight = None
    config.modelconfig.drop_path = 0.0
    config.modelconfig.size = 'micro'
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = {"lr_min": 1e-6}
    config.modelconfig.pos_weight = 1.0
    config.Batch_SIZE = 4
    run(config, reset_deterministic=True)

    print('### convnextTiny droppath=0.0')
    set_deterministic()
    config = get_config("microVsTiny_convnextTiny_00")
    config.modelconfig.cosine_decay = dict(lr_min=1e-6)
    config.modelconfig.loss_fn = "bce_sev"
    config.modelconfig.use_transformer = False
    config.modelconfig.pos_weight = None
    config.modelconfig.drop_path = 0.0
    config.modelconfig.size = 'tiny'
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = {"lr_min": 1e-6}
    config.modelconfig.pos_weight = 1.0
    config.Batch_SIZE = 4
    run(config, reset_deterministic=True)



if __name__ == '__main__':
    main()