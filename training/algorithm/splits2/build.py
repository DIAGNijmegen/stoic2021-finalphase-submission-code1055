from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


def assign_folds(df, num_folds, random_state=None):
    fold_size = len(df) // num_folds
    orig_len = len(df)
    folds = []
    for fold_id in range(num_folds - 1):
        fold_frac = fold_size / (orig_len - fold_id * fold_size)
        fold = df.groupby(["probCOVID", "probSevere", "sex"]).sample(
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


def collect_metadata(dataroot):
    metadata_path = Path(__file__).parent / "ref.csv"
    if metadata_path.exists():
        return pd.read_csv(metadata_path, index_col="PatientID")

    dataroot = Path(dataroot)
    ref = pd.read_csv(dataroot / "metadata" / "reference.csv", index_col="PatientID")

    ages = []
    sexs = []
    # takes about an hour
    for patient in tqdm(ref.index, unit="patient", desc="Load metadata"):
        img = sitk.ReadImage(str(dataroot / "data" / "mha" / f"{patient}.mha"))
        ages.append(img.GetMetaData("PatientAge"))
        sexs.append(img.GetMetaData("PatientSex"))

    ref["age"] = ages
    ref["sex"] = sexs

    ref.to_csv(metadata_path, index=True)
    return ref


def main(dataroot):
    ref = collect_metadata(dataroot)
    split_dir = Path(__file__).parent

    # base split: 5-fold cross validation
    cv5 = assign_folds(ref, num_folds=5, random_state=1055)
    cv5.to_csv(split_dir / "cv5.csv", index=True)

    # all data and balanced
    cv5_balanced = balance(cv5)
    cv5_balanced.to_csv(split_dir / "cv5_all_bal.csv", index=True)

    # inf only and unbalanced
    cv5_infonly = cv5[cv5["probCOVID"] == 1]
    cv5_infonly.to_csv(split_dir / "cv5_infonly.csv", index=True)

    # inf only and balanced
    cv5_infonly_balanced = balance(cv5_infonly)
    cv5_infonly_balanced.to_csv(split_dir / "cv5_infonly_bal.csv", index=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "data", help="Contains at least a data folder and a metadata folder."
    )
    args = parser.parse_args()
    main(args.data)
