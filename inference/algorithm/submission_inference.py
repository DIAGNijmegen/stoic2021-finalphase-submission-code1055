from pathlib import Path
import shutil
import subprocess as sp
import json
import time
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm


def build_container():
    proc = sp.Popen([
        "docker",
        "build",
        "-t", "stoicalgorithm",
        str(Path(__file__).parent),
    ])
    assert proc.wait() == 0


def process_patient(patient_path, log_csv):
    parent = Path(__file__).parent.absolute()
    memory = "16g"

    # copy patient to input location
    container_io_path = parent/"container-io"
    container_input_path = container_io_path/"images"/"ct"
    container_input_path.mkdir(exist_ok=True, parents=True)
    container_output_path = container_io_path/"output"
    container_output_path.mkdir(exist_ok=True, parents=True)
    container_output_path.chmod(0o0777)

    # remove all previous files
    prev_mhas = list(container_input_path.glob("*.mha"))
    for path in prev_mhas:
        path.unlink()
    shutil.copyfile(patient_path, container_input_path/patient_path.name)

    t0 = time.time()
    proc = sp.Popen([
        "docker",
        "run",
        "--rm",
        "--gpus", "all",
        f"--memory={memory}",
        f"--memory-swap={memory}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--network", "none",
        "--shm-size=128m",
        "--pids-limit", "256",
        "-v", f"{container_io_path}/:/input/",
        "-v", f"{container_output_path}/:/output/",
        "--user", "algorithm",
        "stoicalgorithm",
    ])
    assert proc.wait() == 0
    t1 = time.time()

    # read output
    with open(container_output_path/"probability-severe-covid-19.json") as f_sev:
        sev_prob = json.load(f_sev)
    with open(container_output_path/"probability-covid-19.json") as f_inf:
        inf_prob = json.load(f_inf)

    row = {
        "patient": patient_path.stem,
        "sev": sev_prob,
        "inf": inf_prob,
        "time": t1 - t0,
    }
    if Path(log_csv).exists():
        logs = pd.read_csv(log_csv)
        logs = logs.append([row])
    else:
        logs = pd.DataFrame([row])

    logs.to_csv(log_csv, index=False)


def main(mha_repo, out_csv):
    mha_repo = Path(mha_repo)
    build_container()
    for patient_path in tqdm(list(mha_repo.glob("*.mha")), unit="patient"):
        process_patient(patient_path, out_csv)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mha")
    parser.add_argument("output")
    args = parser.parse_args()
    main(args.mha, args.output)
