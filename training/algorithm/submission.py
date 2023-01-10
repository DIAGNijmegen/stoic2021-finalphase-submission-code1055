# this script is run on the challenge servers to evaluate our performance
from typing import Sequence
from pathlib import Path
import json
import SimpleITK
import torch
from evalutils.validators import UniquePathIndicesValidator, UniqueImagesValidator
from evalutils.evalutils import Algorithm

from misc_utilities.determinism import set_deterministic
from data.data_api import apply_transforms
from config.config import BaseConfig
from config.modelconfig import get_model
from data.data_api import tfms_from_config
from loss import get_loss


# you must add a checkpoint.pt to the top directory for this to work


COVID_OUTPUT_NAME = Path("probability-covid-19")
SEVERE_OUTPUT_NAME = Path("probability-severe-covid-19")


def assert_check_subset(item, other, ignore_paths: Sequence[str] = None):
    """Check if `item` is equivalent to `other`

    @ignore_paths: Dot (.) separated paths that should not be checked (e.g.: config.modelconfig.size)
    """

    if ignore_paths is None:
        ignore_paths = []

    def check_subset(item, other, path: Sequence[str]):
        # ignore this path?
        for ign in ignore_paths:
            if ign == ".".join(path):
                return []

        mismatches = []
        if isinstance(item, dict):
            if not isinstance(other, dict):
                return [(path, item, other)]

            for key, value in item.items():
                mismatches.extend(check_subset(value, other[key], path=path + [key]))
        elif isinstance(item, list):
            for i, (child, child_other) in enumerate(zip(item, other)):
                mismatches.extend(
                    check_subset(child, child_other, path=path + [str(i)])
                )
        elif item != other:
            return [(path, item, other)]
        return mismatches

    mismatches = check_subset(item, other, path=["config"])
    if len(mismatches) > 0:
        mm_str = "\n".join(
            [
                f"({'.'.join(path)})\ncheckpoint: {item}\ncode: {other}"
                for (path, item, other) in mismatches
            ]
        )
        raise AssertionError(f"Not a subset:\n{mm_str}")


class StoicAlgorithm(Algorithm):
    def __init__(self, config):
        input_path = Path(config.DATA_PATH)
        output_path = Path(config.LOGS_PATH)
        output_path.mkdir(exist_ok=True)

        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=input_path,
            output_path=output_path,
        )

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        for ckp_path in sorted(
            Path("/opt/algorithm/artifact").glob("*.pt"),
            key=lambda p: "-".join(p.stem.split("-")[1:]),
            reverse=True,
        ):
            try:
                latest_ckp_path = ckp_path
                checkpoint = torch.load(latest_ckp_path, map_location=torch.device(self.device))
                break
            except:
                print("skip", ckp_path)
                pass

        print("Used checkpoint:", latest_ckp_path)

        # check if the loaded config is the same as the codebase
        assert_check_subset(
            checkpoint["config"],
            config.to_dict(),
            ignore_paths=[
                "config.git_commit",
                "config.created",
                "config._current_fold",
                "config.nickname",
                "config.dataconfigs",
                "config.cache_path",
            ],
        )

        # load models
        config.modelconfig.pretrained = False
        self.models = []
        for model_weights in checkpoint["models"]:
            model = get_model(config)
            model.load_state_dict(model_weights)
            model.to(self.device)
            model.eval()
            self.models.append(model)

        self.loss = get_loss(checkpoint["config"]["modelconfig"]["loss_fn"])

        self.pipeline = tfms_from_config(
            checkpoint["data_transforms"], submission_mode=True
        )

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Classify input_image image
        return self.predict(input_image=input_image)

    def predict(self, *, input_image: SimpleITK.Image):
        # pre-processing
        sample = apply_transforms(self.pipeline, dict(img=input_image))
        input_image = sample["img"]
        # unsqueeze for batch size = 1
        input_image = torch.tensor(input_image).unsqueeze(0)

        # run models
        outputs = []
        with torch.no_grad():
            for model in self.models:
                set_deterministic()
                outputs.append(model(input_image.to(self.device)).cpu())

        prob_covid, prob_severe = self.loss.finalize(torch.cat(outputs, dim=0))
        prob_covid = prob_covid.mean()
        prob_severe = prob_severe.mean()

        return {
            COVID_OUTPUT_NAME: prob_covid.item(),
            SEVERE_OUTPUT_NAME: prob_severe.item(),
        }

    def save(self):
        if len(self._case_results) > 1:
            raise RuntimeError(
                "Multiple case prediction not supported with single-value output interfaces."
            )
        case_result = self._case_results[0]

        for output_file, result in case_result.items():
            with open(str(self._output_path / output_file) + ".json", "w") as f:
                json.dump(result, f)


if __name__ == "__main__":
    # model_name must match the checkpoint's model
    # if not, an AssertionError will be raised
    config = BaseConfig(model_name="convnext", cv_enabled=True, data_name="gpudeformed")
    StoicAlgorithm(config).process()
