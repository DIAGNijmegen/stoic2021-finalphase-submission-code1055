from datetime import datetime
from pathlib import Path
import logging
import sys
import torch
import os
import socket
from socket import gethostname
import pandas as pd

from config.modelconfig import get_modelconfig
from config.dataconfig import MosmedConfig, get_cvsplit_patients, get_dataconfig
from misc_utilities.git import get_git_commit


WORKSTATION = 85
GROSSMUTTER = 65
SEPPEL = 243
WASTI_DANI = 180
WASTI_KATJA = 168
DIMPFELMOSER = 184
DIMPFELMOSER_DANI = 73
DANISSURFACE = 77
MMC094 = 240


SIEBEN = 7
ZEHN = 10


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


    def __init__(
        self,
        model_name="convnext",
        num_steps: int = 1,
        cv_enabled=False,
        num_trained_stages: int = 4,
        split="cv5_infonly_bal.csv",
        nickname=None,
        data_name: str = "deformed",
        existing_output=None,
    ):
        if existing_output is not None and nickname is not None:
            logging.warn("Config uses existing output but nickname is set. Nickname will be ignored.")

        self.existing_output = existing_output
        self.nickname = nickname
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        #self.DEVICE = 'cpu'
        self.DEVICE = torch.device(self.DEVICE)

        if sys.gettrace() is None:
            self.WORKERS = 8
        else:
            self.WORKERS = 0
        self.WORKERS = 8

        # LOGS_PATH and data_path can either be set using an environment variable
        # or hardcoded based on IP address
        self.LOGS_PATH = os.environ.get("LOGS_PATH", None)
        self.DATA_PATH = os.environ.get("DATA_PATH", None)
        self.PRETRAINED_PATH = os.environ.get("PRETRAINED_PATH", None)  # TODO: Fill in your pretrained models path!
        self.cache_path = os.environ.get("CACHE_PATH", None)
        if self.cache_path is not None:
            logging.info(f"Use cache path setting from environment variable: {self.cache_path}")
            self.cache_path = Path(self.cache_path)
            self.cache_path.mkdir(exist_ok=True, parents=True)

        if self.LOGS_PATH is None or self.DATA_PATH is None:
            if get_ip_end() == WASTI_DANI:
                self.DATA_PATH = '/data/ssd1/kienzlda/data_stoic/'
                self.LOGS_PATH = '/data/ssd1/kienzlda/stoic_logs/logs/'
                self.PRETRAINED_PATH = '/data/ssd1/kienzlda/stoic_logs/saved_models/'
            elif get_ip_end() == DIMPFELMOSER_DANI:
                self.DATA_PATH = '/data/howto100m_features/lorenjul/stoic/data'
                self.LOGS_PATH = '/data/ssd1/kienzlda/stoic_logs/logs/'
                self.PRETRAINED_PATH = '/data/ssd1/kienzlda/stoic_logs/saved_models/'
            elif get_ip_end() == WASTI_KATJA:
                self.DATA_PATH = '/data/ssd1/kienzlda/data_stoic/'
                self.LOGS_PATH = '/data/ssd2/ludwikat/stoic/logs'
                self.PRETRAINED_PATH = None #TODO: Fill in your pretrained models path!
            elif get_ip_end() == DANISSURFACE:
                self.DATA_PATH = os.path.join('//mmc01', 'MMCRW', 'Datensaetze', 'stoic_data')
                self.LOGS_PATH = os.path.join('..', 'logs', 'stoic_logs')
                self.PRETRAINED_PATH = None  # TODO: Fill in your pretrained models path!
            elif get_ip_end() == MMC094:
                self.DATA_PATH = '/data/raid_ssd/datasets/stoic_data/'
                self.LOGS_PATH = '/data/raid_ssd/datasets/stoic_logs/'
                self.PRETRAINED_PATH = "/data/raid_ssd/schoerob/saved_stoic_models/saved_models/"  # TODO: Fill in your pretrained models path!
            else:
                raise Exception("No adequate system found.")

        self.DATA_NAME = data_name
        # self.mosmed_config = MosmedConfig(self)
        self.mosmed_config = None

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
        self.num_folds = 5

        self.dataconfigs = dict(
            train=get_dataconfig(self, is_validation=False, cache_path=self.cache_path),
            val=get_dataconfig(self, is_validation=True, cache_path=self.cache_path),
        )
        self.set_fold(0)

        self.MODEL_NAME = model_name
        self.modelconfig = get_modelconfig(model_name, num_trained_stages)

        #self.Batch_SIZE = 8
        self.Batch_SIZE = 4
        #self.MAX_EPOCHS = 20
        self.MAX_EPOCHS = 25
        self.NUM_STEPS = num_steps
        self.use_ema = True
        # set self.early_stopping to 0 to disable early stopping
        # self.early_stopping = 0
        self.early_stopping = SIEBEN
        self.early_stopping_grace_epochs = ZEHN

        self.babysitter_decay = None
        self.babysitter_grace_steps = None
        # self.babysitter_decay = 10
        # self.babysitter_grace_steps = 8
        assert (self.babysitter_decay is None) == (self.babysitter_grace_steps is None)

        # the last checkpoint is always saved, but how many additional checkpoints should be kept?
        # setting this number will set how many of the top performing checkpoints should be kept
        # set to -1 to keep all checkpoints
        self.keep_additional_checkpoints = 10
        assert self.keep_additional_checkpoints != 0, "Keeping zero checkpoints is currently not supported"

        # time used for the run identifier
        # when using cross validation, the identifier must stay the same between folds
        # therefore, use the same start time for the identifiers
        self.created = datetime.now()
        #self.created = datetime.fromisoformat("2022-03-28 16:08:38.531437")
        # during training, the git commit can change => store it now
        self.git_commit = get_git_commit()


    @property
    def split_path(self):
        return Path(__file__).parent.parent/"splits2"/self.split

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

    def make_identifier(self):
        "Returns an identifier string containing model name, date, and (optionally) git commit"

        identifier_parts = []
        # nickname
        if self.nickname is not None:
            identifier_parts.append(self.nickname)
        # cross validation
        if self.cv_enabled:
            identifier_parts.append("cv")
        # model
        identifier_parts.append(self.MODEL_NAME)
        # time
        identifier_parts.append(self.created.strftime("%Y%m%d-%H%M%S"))
        # git commit
        if self.git_commit is not None:
            identifier_parts.append(self.git_commit)

        return "_".join(identifier_parts)

    def get_output_dir(self):
        if self.existing_output is None:
            base_output: Path = Path(self.LOGS_PATH) / self.make_identifier()
        else:
            base_output = Path(self.existing_output)
        if self.cv_enabled:
            return base_output / f"cv{self._current_fold}"
        return base_output

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

        return based
