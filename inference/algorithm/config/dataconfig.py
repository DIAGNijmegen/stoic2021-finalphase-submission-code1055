import logging
from pathlib import Path
from typing import Sequence, Union
import pandas as pd

from data.data2d import get_slice2d_dataset
from data.deformed import get_deformed_dataset
from data.gpu_deformed import get_gpudeformed_dataset
from data.mosmed import get_mosmed_dataset
from data.noaug import get_noaug_dataset
from data.hust import get_hust_dataset


class Slice2DDataConfig:
    def __init__(self, config, patients: Sequence[int] = None, is_validation=False):
        self.data_path = config.DATA_PATH
        self.patients = patients
        self.is_validation = is_validation


class DeformedDataConfig:
    def __init__(
        self,
        config,
        patients: Sequence[int] = None,
        is_validation=False,
        cache_path=None,
    ):
        self.data_path = config.DATA_PATH
        self.patients = patients
        self.is_validation = is_validation
        if cache_path is None:
            self.precompute_path = cache_path
        else:
            self.precompute_path = Path(self.data_path) / "data" / "auto_cache"
        logging.info(f"Cache path: {self.precompute_path}")

        self.cache_img_size = 256
        self.img_size = 224

        self.num_repeats = 1  # 4

        # configs for augmentation
        # AxialFlip
        self.flip = True

        # Rotate
        self.rotate_prob = 0.2
        self.rotate = 30

        # GaussianFilter
        self.blur_prob = 0.2
        self.blur_sigma = (0.6, 0.8)

        # AddNoise
        self.noise_prob = 0.2
        self.noise_mean = (0.5, 0.5)
        self.noise_std = (0.3, 1)
        self.noise_weight = 0.03

        self.deform = True


class GPUDeformedDataConfig:
    def __init__(
        self,
        config,
        patients: Sequence[int] = None,
        is_validation=False,
        cache_path=None,
    ):
        self.data_path = config.DATA_PATH
        self.patients = patients
        self.is_validation = is_validation
        if cache_path is not None:
            self.precompute_path = cache_path
        else:
            self.precompute_path = Path(self.data_path) / "data" / "auto_cache"
        logging.info(f"Cache path: {self.precompute_path}")

        self.cache_img_size = 256
        self.img_size = 224

        # configs for augmentation
        # AxialFlip
        self.flip = True

        # Rotate
        self.rotate_prob = 0.2
        self.rotate = 30

        # GaussianFilter
        self.blur_prob = 0.2
        self.blur_sigma = (0.6, 0.8)

        # AddNoise
        self.noise_prob = 0.2
        self.noise_mean = (0.5, 0.5)
        self.noise_std = (0.3, 1)
        self.noise_weight = 0.03

        self.deform_prob = 0.5
        self.deform_sigma = (35, 35)
        self.deform_alpha = (1, 7)

        # self.do_normalization = False
        self.do_normalization = True
        self.mean = 0.3110
        self.std = 0.3154


class MosmedConfig:
    def __init__(self, config, patients=None, is_validation=False):
        self.data_path = config.DATA_PATH
        self.patients = patients
        self.is_validation = is_validation
        self.img_size = 224

        # configs for augmentation
        # AxialFlip
        self.flip = True

        # Rotate
        self.rotate_prob = 0.2
        self.rotate = 30

        # GaussianFilter
        self.blur_prob = 0.2
        self.blur_sigma = (0.6, 0.8)

        # AddNoise
        self.noise_prob = 0.2
        self.noise_mean = (0.5, 0.5)
        self.noise_std = (0.3, 1)
        self.noise_weight = 0.03

        self.deform_prob = 0.5
        self.deform_sigma = (35, 35)
        self.deform_alpha = (1, 7)

        self.mosmed_path = Path(config.DATA_PATH) / "data" / "mosmed"
        self.mosmed_inf_level = 1
        self.mosmed_sev_level = 3

        self.output_study = False

        self.do_normalization = False
        self.mean = 0.2926
        self.std = 0.3119


class HustConfig:
    def __init__(self, config, patients=None, is_validation=False):
        self.data_path = config.DATA_PATH
        self.patients = patients
        self.is_validation = is_validation
        self.img_size = 224

        # configs for augmentation
        # AxialFlip
        self.flip = True

        # Rotate
        self.rotate_prob = 0.2
        self.rotate = 30

        # GaussianFilter
        self.blur_prob = 0.2
        self.blur_sigma = (0.6, 0.8)

        # AddNoise
        self.noise_prob = 0.2
        self.noise_mean = (0.5, 0.5)
        self.noise_std = (0.3, 1)
        self.noise_weight = 0.03

        self.deform_prob = 0.5
        self.deform_sigma = (35, 35)
        self.deform_alpha = (1, 7)

        # self.hust_path = Path(config.DATA_PATH) / "data" / "hust"
        # TODO move hust in the correct directory...
        self.hust_path = Path(config.DATA_PATH) / "hust"

        self.output_study = False

        self.do_normalization = False
        self.mean = 0.7454
        self.std = 0.2002


class NoAugConfig:
    def __init__(self, config, patients=None, is_validation=False, cache_path=None):
        self.data_path = config.DATA_PATH
        self.patients = patients
        self.is_validation = is_validation
        if cache_path is not None:
            self.precompute_path = cache_path
        else:
            self.precompute_path = Path(self.data_path) / "data" / "auto_cache"
        logging.info(f"Cache path: {self.precompute_path}")

        self.img_size = 224


def get_dataconfig(
    config, patients: Sequence[int] = None, is_validation=False, cache_path=None
):
    "Provide `patients` to only select a subset of all cases. Useful for train/val/test split."

    if patients is not None and len(patients) == 0:
        # patients is empty which means we should skip the dataset
        return None

    dataset_name = config.DATA_NAME
    if dataset_name == "deformed":
        return DeformedDataConfig(
            config,
            patients=patients,
            is_validation=is_validation,
            cache_path=cache_path,
        )
    if dataset_name == "gpudeformed":
        return GPUDeformedDataConfig(
            config,
            patients=patients,
            is_validation=is_validation,
            cache_path=cache_path,
        )
    if dataset_name == "slice2d":
        return Slice2DDataConfig(config, patients=patients, is_validation=is_validation)
    if dataset_name == "noaug":
        return NoAugConfig(
            config,
            patients=patients,
            is_validation=is_validation,
            cache_path=cache_path,
        )
    if dataset_name == "mosmed_stoic" or dataset_name == "mosmed":
        return MosmedConfig(config, patients=patients, is_validation=is_validation)
    if dataset_name == "hust_stoic" or dataset_name == "hust":
        return HustConfig(config, patients=patients, is_validation=is_validation)
    raise KeyError(f"No dataset found for {dataset_name}")


def get_cvsplit_patients(index_path: Union[pd.DataFrame, str, Path], fold_id: int):
    if isinstance(index_path, pd.DataFrame):
        index = index_path
    else:
        index = pd.read_csv(index_path)

    val = index[index["cv"] == fold_id]
    train = index.drop(index=val.index)

    return dict(
        train=train["PatientID"].tolist(),
        val=val["PatientID"].unique().tolist(),
    )


def get_dataset(dataconfig):
    if dataconfig is None:
        # patients is empty which means we should skip the dataset
        return []

    if isinstance(dataconfig, DeformedDataConfig):
        return get_deformed_dataset(dataconfig)
    if isinstance(dataconfig, GPUDeformedDataConfig):
        return get_gpudeformed_dataset(dataconfig)
    if isinstance(dataconfig, Slice2DDataConfig):
        return get_slice2d_dataset(dataconfig)
    if isinstance(dataconfig, NoAugConfig):
        return get_noaug_dataset(dataconfig)
    if isinstance(dataconfig, MosmedConfig):
        return get_mosmed_dataset(dataconfig)
    if isinstance(dataconfig, HustConfig):
        return get_hust_dataset(dataconfig)

    raise KeyError(f"No dataset found for {dataconfig}")
