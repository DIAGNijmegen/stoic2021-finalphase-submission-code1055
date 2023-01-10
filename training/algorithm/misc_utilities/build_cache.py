# script to build cache for a given dataconfig
# this should run faster than simply iterating over the dataset because unnecessary transforms are skipped
from multiprocessing.pool import ThreadPool
from pathlib import Path
import logging
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
import pandas as pd

# you should execute this script as a module to make the following imports work
# (python -m misc_utilities.build_cache)
from data.data_api import *


def file_cache_only(dataset: Dataset):
    tfms = []
    for tfm in dataset.transforms:
        tfms.append(tfm)
        if isinstance(tfm, FileCache):
            break

    assert isinstance(tfms[-1], FileCache), "The pipeline does not contain a FileCache"
    return tfms


def get_inputs(data_root: Path):
    labels = pd.read_csv(data_root/"metadata/reference.csv").rename(columns={
        "probCOVID": "inf",
        "probSevere": "sev",
        "PatientID": "patient",
    })
    mha_dir = data_root/"data/mha"
    labels["path"] = [str(mha_dir/f"{patient}.mha") for patient in labels["patient"]]
    return labels[["patient", "path", "inf", "sev"]].to_dict(orient="records")


def build_cache(file_cache: FileCache, inputs, num_workers: int):
    def worker(item):
        file_cache(item)
        return None

    with ThreadPool(num_workers) as pool:
        list(tqdm(pool.imap(worker, inputs), total=len(inputs), desc=file_cache.cache_root.stem[:6]))


def main(data_root, cache_root, num_workers: int = 20, outer_size: int = 128, inner_size: int = 112):
    logging.basicConfig(level=logging.INFO)

    normal_caches = [
        FileCache(
            [
                LoadMha(),
                Zoom((outer_size, outer_size, outer_size)),
                Scale01(-1000, 500),
            ],
            cache_root=cache_root,
            keys=("patient", ),
            auto_dir=True,
        ),
        FileCache(
            [
                LoadMha(),
                Zoom((inner_size, inner_size, inner_size)),
                Scale01(-1000, 500),
            ],
            cache_root=cache_root,
            keys=("patient", ),
            auto_dir=True,
        ),
    ]

    with logging_redirect_tqdm():
        inputs = get_inputs(Path(data_root))

        for file_cache in tqdm(normal_caches, unit="filecache"):
            build_cache(file_cache, inputs, num_workers)


if __name__ == "__main__":
    try:
        import fire
        fire.Fire(main)
    except ModuleNotFoundError:
        main()
