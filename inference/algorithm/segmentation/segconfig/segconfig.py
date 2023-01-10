import sys
import torch
import os
import socket
from socket import gethostname
from segmentation.segconfig.segmodelconfig import get_modelconfig

from config.dataconfig import get_cvsplit_patients, get_dataconfig
from misc_utilities.git import get_git_commit
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime as datetimefunction

#from config.dataconfig import STOICDataConfig, STOICCachedDataConfig


WORKSTATION = 85
GROSSMUTTER = 65
SEPPEL = 243
WASTI_DANI = 180
WASTI_KATJA = 168
DIMPFELMOSER = 184
DANISSURFACE = 77
MMC094 = 240
DIMPFELMOSER_DANI = 73


def get_ip_end():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    # print(int(IP[IP.rfind(".") + 1:]))
    return int(IP[IP.rfind(".") + 1:])

class BaseConfig:
    """
    Base class for configuration files of all kinds
    """

    def __init__(self, args):
        #self.GPUS = [6]
        self.GPUS = args.gpu

        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in self.GPUS])

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        #self.DEVICE = 'cpu'
        self.DEVICE = torch.device(self.DEVICE)

        if sys.gettrace() is None:
            self.WORKERS = 4
        else:
            self.WORKERS = 0

        self.IMAGE_SIZE = args.imsize

        if get_ip_end() == WASTI_DANI:
            self.DATA_PATH = '/data/ssd1/kienzlda/data_stoic/segmentation/resized' + str(self.IMAGE_SIZE)
            self.LOGS_PATH = '/data/ssd1/kienzlda/stoic_logs/segmentation'
            #TODO maybe make it independent of stoic project
            self.PRETRAINED_PATH = '/data/ssd1/kienzlda/stoic_logs/saved_models/'
        elif get_ip_end() == DIMPFELMOSER_DANI:
            self.DATA_PATH = '/data/ssd1/kienzlda/data_stoic/segmentation/resized'
            self.LOGS_PATH = '/data/ssd1/kienzlda/stoic_logs/segmentation'
            self.PRETRAINED_PATH = '/data/ssd1/kienzlda/stoic_logs/saved_models/'
        # elif get_ip_end() == DANISSURFACE:
        #     self.DATA_PATH = os.path.join('//mmc01', 'MMCRW', 'Datensaetze', 'stoic_data')
        #     self.LOGS_PATH = os.path.join('..', 'logs', 'stoic_logs')
        # elif get_ip_end() == MMC094:
        #     self.DATA_PATH = '/data/raid_ssd/datasets/stoic_data/'
        #     self.LOGS_PATH = '/data/raid_ssd/datasets/stoic_logs/'
        else:
            raise Exception("No adequate system found.")

        #TODO create PATH for file to save the configuration of the run

        self.Batch_SIZE = 8
        self.MAX_EPOCHS = 100
        self.STARTEPOCH_UNSUPERVISED = 14

        self.MODEL_NAME = args.model
        self.SUPERVISED = True if int(args.mode) == 1 else False
        self.UNSUPERVISED_AUGMENTATION = None if int(args.mode) == 2 else 'elastic_deformation'
        self.UNSUPERVISED_DATA = 'tcia' #'stoic
        self.LOSS = 'dicece' #'balancedce'
        self.DO_NORMALIZATION = True
        self.SUPERVISED_AUGMENTATIONS = False

        self.modelconfig = get_modelconfig(self)


class MultiConfig:
    def __init__(self, args, split, cv_enabled):
        #self.GPUS = [6]
        self.GPUS = args.gpu

        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in self.GPUS])

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        #self.DEVICE = 'cpu'
        #self.DEVICE = 'cpu'
        self.DEVICE = torch.device(self.DEVICE)

        if sys.gettrace() is None:
            self.WORKERS = 4
        else:
            self.WORKERS = 0

        self.IMAGE_SIZE = args.imsize

        if get_ip_end() == WASTI_DANI:
            #TODO add classification data
            self.DATA_PATH = '/data/ssd1/kienzlda/data_stoic/'
            self.DATA_PATH_SEG = '/data/ssd1/kienzlda/data_stoic/segmentation/resized' + str(self.IMAGE_SIZE)
            self.LOGS_PATH = '/data/ssd1/kienzlda/stoic_logs/multitask'
            #TODO maybe make it independent of stoic project
            self.PRETRAINED_PATH = '/data/ssd1/kienzlda/stoic_logs/saved_models/'
        elif get_ip_end() == DIMPFELMOSER_DANI:
            self.DATA_PATH = '/data/howto100m_features/lorenjul/stoic/data'
            self.DATA_PATH_SEG = '/data/ssd1/kienzlda/data_stoic/segmentation/resized'
            self.LOGS_PATH = '/data/ssd1/kienzlda/stoic_logs/multitask'
            self.PRETRAINED_PATH = '/data/ssd1/kienzlda/stoic_logs/saved_models/'
        # elif get_ip_end() == DANISSURFACE:
        #     self.DATA_PATH = os.path.join('//mmc01', 'MMCRW', 'Datensaetze', 'stoic_data')
        #     self.LOGS_PATH = os.path.join('..', 'logs', 'stoic_logs')
        # elif get_ip_end() == MMC094:
        #     self.DATA_PATH = '/data/raid_ssd/datasets/stoic_data/'
        #     self.LOGS_PATH = '/data/raid_ssd/datasets/stoic_logs/'
        else:
            raise Exception("No adequate system found.")

        #TODO create PATH for file to save the configuration of the run

        self.Batch_SIZE = 4
        self.MAX_EPOCHS = 25

        self.MODEL_NAME = args.model
        self.LOSS_SEG = 'ce'#'dicece' #'balancedce'
        self.LOSS_CLS = 'bce' # 'ce'
        self.DO_NORMALIZATION = True

        self.DO_ELASTIC_DEFORM = True
        self.deform_prob = 0.5
        self.deform_sigma = (35, 35)
        self.deform_alpha = (1, 7)

        self.datasets = ['stoic', 'tcia', 'mosmed', 'hust']
        #self.datasets = ['stoic', 'tcia', 'mosmed']
        #self.datasets = ['mosmed', 'tcia']
        self.dataset_weights = [1, 1, 1, 1]
        self.change_loss_weight = True

        self.split = split
        self.cv_enabled = cv_enabled
        # check how many cv folds are available
        self.num_folds = 0
        self._current_fold = 0
        if self.split_path.exists():
            if cv_enabled:
                split_df = pd.read_csv(self.split_path)
                self.num_folds = split_df["cv"].nunique()
                del split_df
            else:
                self.num_folds = 1

        self.DATA_NAME = "deformed"
        self.dataconfigs = dict(
            train=get_dataconfig(self, is_validation=False),
            val=get_dataconfig(self, is_validation=True),
        )
        self.set_fold(0)

        self.NUM_STEPS = 1

        self.modelconfig = get_modelconfig(self)

        self.datetime = datetimefunction.now().strftime("%Y-%m-%d_%H-%M-%S")

    @property
    def split_path(self):
        return Path(__file__).parent.parent / '..' / "splits2" / self.split

    def set_fold(self, fold=0):
        self._current_fold = fold
        if self.split_path.exists():
            patients = get_cvsplit_patients(self.split_path, fold)
            self.dataconfigs["train"].patients = patients["train"]
            self.dataconfigs["val"].patients = patients["val"]
        else:
            logging.warn(f"""{self.split_path} does not exist. config.dataconfigs is set to None.
                You can ignore this message when building the submission container""")
            self.dataconfigs = None

    def to_dict(self):
        base_delkeys = ["DEVICE", "WORKERS", "LOGS_PATH", "DATA_PATH", "PRETRAINED_PATH"]
        data_delkeys = ["patients", "data_path"]

        based = dict(self.__dict__)
        # ignore some keys of the base config
        for delkey in base_delkeys:
            del based[delkey]

        if self.dataconfigs is not None:
            # dataconfigs require special handling (since they are objects)
            data_dict = {}
            for phase, d in self.dataconfigs.items():
                if d is None:
                    data_dict[phase] = None
                else:
                    d = dict(d.__dict__)
                    for delkey in data_delkeys:
                        if delkey in d:
                            del d[delkey]
                    data_dict[phase] = d
            based["dataconfigs"] = data_dict

        # modelconfig requires special treatment (is a python object)
        based["modelconfig"] = self.modelconfig.__dict__