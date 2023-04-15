import argparse
import os

import hydra
import pandas as pd
import pyrootutils

from predict import DecisionWithoutMajority, EnsembleError, ensemble_predict
from utils.config import register_resolvers

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help=(
            "An odd number (e.g. 1, 3, 5, etc.) of model checkpoints. "
            + "Predictions will be used as an ensemble. If a majority of "
            + "models choose a given label then it is chosen."
        ),
    )
    parser.add_argument(
        "--submission-name",
        type=str,
        default="submission.csv",
        help="The path of the .csv file with the predicted labels.",
    )
    args = parser.parse_args()
    if len(args.models) % 2 == 0:
        raise EnsembleError("Must have an odd number of models in the ensemble.")
    return args


def main(args: argparse.Namespace):
    hydra.initialize(version_base="1.3", config_path="../configs")
    cfg = hydra.compose(config_name="test.yaml")

    register_resolvers()

    data_module = hydra.utils.instantiate(cfg.data.constructor)

    predictions = ensemble_predict(
        checkpoints=args.models,
        dataset=data_module.test_dataset,
        model_constructor=cfg.model.constructor_predict,
        no_majority_decision=DecisionWithoutMajority.RANDOM,
        id2label=cfg.data.id2label,
        seed=cfg.seed,
    )

    df = pd.DataFrame.from_dict(
        {
            "Id": range(len(predictions)),
            "Label": [pred.label[1] for pred in predictions],
        }
    )
    if not os.path.exists(cfg.paths.submissions_dir):
        os.makedirs(cfg.paths.submissions_dir)
    df.to_csv(os.path.join(cfg.paths.submissions_dir, args.submission_name), index=False)


if __name__ == "__main__":
    main(parse_args())
