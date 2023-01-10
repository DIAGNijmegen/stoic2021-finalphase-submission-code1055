#!/usr/bin/env python3
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from copy import deepcopy

from config.config import BaseConfig
from config.modelconfig import get_model
from config.dataconfig import get_dataset, get_dataconfig
from loss import get_loss
from misc_utilities.checkpoint import load_best_checkpoint
from misc_utilities.determinism import set_deterministic
from submission import assert_check_subset


def inference(checkpoint: dict, outfile: str, cv_enabled=False):
    torch.set_grad_enabled(False)

    config = BaseConfig(cv_enabled=cv_enabled)
    # check if the loaded config is the same as the codebase
    dataconfig = get_dataconfig(config, patients=None, is_validation=True)
    checkpoint["config"]["dataconfigs"] = None
    assert_check_subset(
        checkpoint["config"],
        config.to_dict(),
        ignore_paths=["config.git_commit", "config.created", "config._current_fold"],
    )

    model = get_model(config)
    model.load_state_dict(checkpoint["model_ema"])
    model.to(config.DEVICE)
    model.eval()

    loss = get_loss(config.modelconfig.loss_fn)

    dataset = get_dataset(dataconfig)
    loader = DataLoader(
        dataset, batch_size=config.Batch_SIZE, num_workers=config.WORKERS, shuffle=False
    )

    outputs = []
    infs = []
    sevs = []
    sexs = []
    ages = []
    patients = []

    for sample in tqdm(loader, unit="batch"):
        v_tensor, age, sex, inf_gt, sev_gt, patient = sample
        v_tensor = v_tensor
        age = age
        sex = sex
        set_deterministic()
        output = model(
            v_tensor.to(config.DEVICE), age.to(config.DEVICE), sex.to(config.DEVICE)
        ).cpu()
        outputs.append(deepcopy(output))
        infs.append(deepcopy(inf_gt))
        sevs.append(deepcopy(sev_gt))
        ages.append(deepcopy(age))
        sexs.append(deepcopy(sex))
        patients.append(deepcopy(patient))
        del output, inf_gt, sev_gt, age, sex, v_tensor, patient, sample

    outputs = torch.cat(outputs)
    df = pd.DataFrame(
        outputs.numpy(), columns=[f"out_{i}" for i in range(outputs.size(1))]
    )
    inf_pred, sev_pred = loss.finalize(outputs)
    df["pred_inf"] = inf_pred.numpy()
    df["pred_sev"] = sev_pred.numpy()
    df["inf"] = torch.cat(infs).numpy()
    df["sev"] = torch.cat(sevs).numpy()
    df["age"] = torch.cat(ages).numpy()
    sexs = torch.cat(sexs).numpy()
    df[[f"sex_{i}" for i in range(sexs.shape[1])]] = sexs
    df["patient"] = torch.cat(patients).numpy()

    df.to_csv(outfile, index=False)


def evaluate_cv(cv_root: Path):
    cv_root = Path(cv_root)
    for fold_dir in tqdm(list(cv_root.iterdir()), unit="fold"):
        checkpoint, checkpoint_path = load_best_checkpoint(
            fold_dir, by="auc_sev2_ema", mode="max", return_path=True
        )
        if checkpoint is None:
            continue
        out_path = fold_dir / f"preds_{checkpoint_path.stem}.csv"
        if out_path.exists():
            continue
        logging.info(
            f"Fold {fold_dir.name}: Using checkpoint {checkpoint_path.relative_to(fold_dir)}"
        )
        inference(checkpoint, out_path, cv_enabled=True)


def cli():
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    subparsers = parser.add_subparsers(dest="mode")

    parser_single = subparsers.add_parser("single")
    parser_single.add_argument("checkpoint")
    parser_single.add_argument("output")

    parser_cv = subparsers.add_parser("cv")
    parser_cv.add_argument("root")

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.mode == "cv":
        evaluate_cv(Path(args.root))

    elif args.mode == "single":
        checkpoint = torch.load(args.checkpoint)
        inference(checkpoint, args.output)
    else:
        raise ValueError("Bad command")


if __name__ == "__main__":
    cli()
