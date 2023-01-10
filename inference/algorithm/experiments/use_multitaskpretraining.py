from config.config import BaseConfig
from training import run
from misc_utilities.determinism import set_deterministic
import os
import logging

def multitask():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logging.getLogger().setLevel(logging.INFO)

    config = BaseConfig(split="cv5_infonly_bal.csv", cv_enabled=True, num_steps=1, data_name="gpudeformed", nickname="use_multitaskpretraining")

    # config.pretrained_mode = 'multitask'
    config.modelconfig.pretrained_mode = 'multitask'
    # config.pos_weight = 1.0
    config.modelconfig.pos_weight = 1.0
    # Parameters from the ConvNeXt paper
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = {
        "lr_min" : 1e-6
    }
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 30
    config.modelconfig.decay_lr_until = 20
    config.dataconfigs['train'].do_normalization = True
    config.dataconfigs['val'].do_normalization = True
    config.modelconfig.size = 'tiny'

    set_deterministic()

    run(config, reset_deterministic=True)

def segmentation():
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    logging.getLogger().setLevel(logging.INFO)

    config = BaseConfig(split="cv5_infonly_bal.csv", cv_enabled=True, num_steps=1, data_name="gpudeformed",
                        nickname="use_segmentationpretraining")

    # config.pretrained_mode = 'segmentation'
    config.modelconfig.pretrained_mode = 'segmentation'
    # config.pos_weight = 1.0
    config.modelconfig.pos_weight = 1.0
    # Parameters from the ConvNeXt paper
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = {
        "lr_min": 1e-6
    }
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 30
    config.modelconfig.decay_lr_until = 20
    config.dataconfigs['train'].do_normalization = False
    config.dataconfigs['val'].do_normalization = False

    set_deterministic()

    run(config, reset_deterministic=True)


def tcia_mosmed():
    set_deterministic()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'#"0"
    logging.getLogger().setLevel(logging.INFO)

    config = BaseConfig(split="cv5_infonly_bal.csv", cv_enabled=True, num_steps=1, data_name="gpudeformed",
                        nickname="use_multitaskpretraining_TciaMosmed")

    # config.pretrained_mode = 'multitask'
    config.modelconfig.pretrained_mode = 'TciaMosmed'
    # config.pos_weight = 1.0
    config.modelconfig.pos_weight = 1.0
    # Parameters from the ConvNeXt paper
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = {
        "lr_min": 1e-6
    }
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 25
    config.modelconfig.decay_lr_until = 20
    config.dataconfigs['train'].do_normalization = True
    config.dataconfigs['val'].do_normalization = True
    config.modelconfig.size = 'tiny'

    run(config, reset_deterministic=True, reverse_cv=True)#False)

def tcia_hust():
    set_deterministic()
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'#"2"
    logging.getLogger().setLevel(logging.INFO)

    config = BaseConfig(split="cv5_infonly_bal.csv", cv_enabled=True, num_steps=1, data_name="gpudeformed",
                        nickname="use_multitaskpretraining_TciaHust")

    # config.pretrained_mode = 'multitask'
    config.modelconfig.pretrained_mode = 'TciaHust'
    # config.pos_weight = 1.0
    config.modelconfig.pos_weight = 1.0
    # Parameters from the ConvNeXt paper
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = {
        "lr_min": 1e-6
    }
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 25
    config.modelconfig.decay_lr_until = 20
    config.dataconfigs['train'].do_normalization = True
    config.dataconfigs['val'].do_normalization = True
    config.modelconfig.size = 'tiny'

    run(config, reset_deterministic=True, reverse_cv=True)#False)


def tcia_mosmed_hust():
    set_deterministic()
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'#"4"
    logging.getLogger().setLevel(logging.INFO)

    config = BaseConfig(split="cv5_infonly_bal.csv", cv_enabled=True, num_steps=1, data_name="gpudeformed",
                        nickname="use_multitaskpretraining_TciaMosmedHust")

    # config.pretrained_mode = 'multitask'
    config.modelconfig.pretrained_mode = 'TciaMosmedHust'
    # config.pos_weight = 1.0
    config.modelconfig.pos_weight = 1.0
    # Parameters from the ConvNeXt paper
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = {
        "lr_min": 1e-6
    }
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 25
    config.modelconfig.decay_lr_until = 20
    config.dataconfigs['train'].do_normalization = True
    config.dataconfigs['val'].do_normalization = True
    config.modelconfig.size = 'tiny'

    run(config, reset_deterministic=True, reverse_cv=True)#False)

def tcia():
    set_deterministic()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    logging.getLogger().setLevel(logging.INFO)

    config = BaseConfig(split="cv5_infonly_bal.csv", cv_enabled=True, num_steps=1, data_name="gpudeformed",
                        nickname="use_multitaskpretraining_Tcia")

    config.modelconfig.pretrained_mode = 'segmentation'
    # config.pos_weight = 1.0
    config.modelconfig.pos_weight = 1.0
    # Parameters from the ConvNeXt paper
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = {
        "lr_min": 1e-6
    }
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 25
    config.modelconfig.decay_lr_until = 20
    config.dataconfigs['train'].do_normalization = True
    config.dataconfigs['val'].do_normalization = True
    config.modelconfig.size = 'tiny'

    run(config, reset_deterministic=True, reverse_cv=False)


def imagenet():
    set_deterministic()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'#"0"
    logging.getLogger().setLevel(logging.INFO)

    config = BaseConfig(split="cv5_infonly_bal.csv", cv_enabled=True, num_steps=1, data_name="gpudeformed",
                        nickname="use_multitaskpretraining_Imagenet")

    config.modelconfig.pretrained_mode = 'imagenet'
    # config.pos_weight = 1.0
    config.modelconfig.pos_weight = 1.0
    # Parameters from the ConvNeXt paper
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = {
        "lr_min": 1e-6
    }
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 25
    config.modelconfig.decay_lr_until = 20
    config.dataconfigs['train'].do_normalization = True
    config.dataconfigs['val'].do_normalization = True
    config.modelconfig.size = 'tiny'

    run(config, reset_deterministic=True, reverse_cv=True)#False)


def tcia_mosmed_weakAlpha():
    set_deterministic()
    os.environ["CUDA_VISIBLE_DEVICES"] = '6'#"5"
    logging.getLogger().setLevel(logging.INFO)

    config = BaseConfig(split="cv5_infonly_bal.csv", cv_enabled=True, num_steps=1, data_name="gpudeformed",
                        nickname="use_multitaskpretraining_TciaMosmed_weakAlpha")

    # config.pretrained_mode = 'multitask'
    config.modelconfig.pretrained_mode = 'TciaMosmed'
    # config.pos_weight = 1.0
    config.modelconfig.pos_weight = 1.0
    # Parameters from the ConvNeXt paper
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = {
        "lr_min": 1e-6
    }
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = 25
    config.modelconfig.decay_lr_until = 20
    config.dataconfigs['train'].do_normalization = True
    config.dataconfigs['val'].do_normalization = True
    config.dataconfigs['train'].deform_alpha = (2, 4)
    config.modelconfig.size = 'tiny'

    run(config, reset_deterministic=True, reverse_cv=True)#False)






if __name__ == "__main__":
    #multitask()
    #segmentation()
    #tcia_mosmed()
    #tcia_hust()
    #tcia_mosmed_hust()
    #tcia()
    #imagenet()
    tcia_mosmed_weakAlpha()