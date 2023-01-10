from pathlib import Path
import logging
import sys

sys.path.append(str(Path(__file__).parent.parent))
from training import Trainer, BaseConfig


def main():
    try:
        import yatlogger
        yatlogger.register(session_name="test aug settings")
    except ModuleNotFoundError:
        pass

    logging.getLogger().setLevel(logging.INFO)

    for enabled_aug in ["none", "rotate", "blur", "noise", "flip"]:
        logging.info(f"Run with {enabled_aug}")

        config = BaseConfig("convnext")
        config.LOGS_PATH = str(Path(config.LOGS_PATH) / "test_aug" / enabled_aug)
        train_data = config.dataconfigs["train"]

        if enabled_aug == "flip":
            train_data.flip == True
        elif enabled_aug == "rotate":
            train_data.rotate = 10
        elif enabled_aug == "blur":
            train_data.blur_prob = 0.2
        elif enabled_aug == "noise":
            train_data.noise_prob = 0.2

        trainer = Trainer(config)
        trainer.run()
    logging.info("Done")


if __name__ == "__main__":
    main()
