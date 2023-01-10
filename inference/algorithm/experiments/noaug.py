import logging
from pathlib import Path
from experiments.gpucli import parse_gpu
from config.config import BaseConfig
from training import multirun


root = Path("~/data/stoic/logs").expanduser()
aug_run = (
    root / "aug_cv_convnext_20220324-163057_ebeb66dcbf94a2e6b2f9d2d93778d9adfe6aaf2a"
)
noaug_run = (
    root / "noaug_cv_convnext_20220324-163057_ebeb66dcbf94a2e6b2f9d2d93778d9adfe6aaf2a"
)
baseline_run = (
    root
    / "baseline_cv_convnext_20220324-163057_ebeb66dcbf94a2e6b2f9d2d93778d9adfe6aaf2a"
)


def get_config(aug: bool, split: str, existing=None):
    if split.startswith(".."):
        nickname = "baseline"
    elif aug:
        nickname = "aug"
    else:
        nickname = "noaug"
    config = BaseConfig(
        nickname=nickname,
        cv_enabled=True,
        num_steps=2,
        split=split,
        data_name="deformed" if aug else "noaug",
        existing_output=existing,
    )
    config.modelconfig.loss_fn = "bce_sev"
    config.Batch_SIZE = 4
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = dict(lr_min=1e-6)
    config.MAX_EPOCHS = 20
    config.modelconfig.pos_weight = 1.0
    return config


parse_gpu()
logging.getLogger().setLevel(logging.INFO)
multirun(
    [
        get_config(
            True,
            "cv5_infonly_bal.csv",
            aug_run,
        ),
        get_config(
            False,
            "cv5_infonly_bal.csv",
            noaug_run,
        ),
        get_config(
            True,
            "../splits/split_sev_cv5.csv",
            baseline_run,
        ),
    ],
    no_tqdm=True,
    reset_deterministic=True,
)
