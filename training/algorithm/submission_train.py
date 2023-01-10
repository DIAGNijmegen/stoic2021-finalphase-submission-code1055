import sys
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from config.config import BaseConfig
from training import run
import torch
from misc_utilities.determinism import set_deterministic
from misc_utilities.build_cache import main as build_cache


def assign_folds(df, num_folds, random_state=None):
    fold_size = len(df) // num_folds
    orig_len = len(df)
    folds = []
    for fold_id in range(num_folds - 1):
        fold_frac = fold_size / (orig_len - fold_id * fold_size)
        fold = df.groupby(["probCOVID", "probSevere"]).sample(
            frac=fold_frac, random_state=random_state
        )
        df = df.drop(index=fold.index)
        fold["cv"] = fold_id
        folds.append(fold)
    df["cv"] = num_folds - 1
    folds.append(df)
    return pd.concat(folds).sort_index()


def balance(df):
    sev_only = df[df.probSevere == 1]
    r = int(len(df) / len(sev_only) - 1)
    return pd.concat([df] + [sev_only] * r).sort_index()


def create_split():
    ref = pd.read_csv("/input/metadata/reference.csv", index_col="PatientID")
    cv5 = assign_folds(ref, num_folds=5, random_state=1055)
    cv5_infonly = cv5[cv5["probCOVID"] == 1]
    cv5_infonly_balanced = balance(cv5_infonly)
    split_dir = Path("/opt/train/splits2")
    split_dir.mkdir(exist_ok=True, parents=True)
    cv5_infonly_balanced.to_csv(split_dir / "cv5_infonly_bal.csv", index=True)

def main():
    parser = ArgumentParser()
    parser.add_argument("gpu", type=int)
    parser.add_argument("direction")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # create a new split for the new data
    create_split()

    # build_cache("/input", "/opt/cache/auto_cache", num_workers=20, outer_size=256, inner_size=224)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_deterministic()

    config = BaseConfig(data_name="gpudeformed", cv_enabled=True, nickname=args.direction)
    run(config, reverse_cv=args.direction == "backwards")


if __name__ == "__main__":
    # input data in config.DATA_PATH
    # write model weights to confug.LOGS_PATH
    main()
    print("Training completed.")
