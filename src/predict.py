import argparse
import os
import random
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import datasets
import hydra
import numpy as np
import pandas as pd
import pyrootutils
from omegaconf import DictConfig
from transformers import Trainer

from utils.config import register_resolvers

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class EnsembleError(Exception):
    pass


@dataclass
class Prediction:
    text: str
    label: Optional[Tuple[int, str]]


class DecisionWithoutMajority(Enum):
    RANDOM = 0
    NONE = 1


def ensemble_predict(
    checkpoints: List[str],
    dataset: datasets.Dataset,
    model_constructor: DictConfig,
    id2label: Dict[int, str],
    no_majority_decision=DecisionWithoutMajority.NONE,
    seed=42,
) -> List[Prediction]:
    ensemble_predictions: List[List[int]] = []
    for checkpoint_path in checkpoints:
        # load the model given its checkpoint path
        model = hydra.utils.instantiate(model_constructor)(checkpoint_path)
        trainer = Trainer(model=model)

        # get the most likely label from each model
        predictions = list(np.argmax(trainer.predict(dataset).predictions, axis=-1))
        ensemble_predictions.append(predictions)

    # get the text for each input sample
    prediction_texts: List[str] = [sample["text"] for sample in dataset]

    # for each sample, make a decision based on the predictions of each model
    # in the ensemble; if a simple majority exists, that is the label, otherwise
    # the "no_majority_decision" argument is used to decide what to do next
    predictions: List[Prediction] = []
    for preds, text in zip(zip(*ensemble_predictions), prediction_texts):
        # see if a majority label exists
        majority_label = None
        for label, count in Counter(preds).items():
            if count > len(preds) / 2:
                majority_label = label

        if majority_label is None:
            # the majority label does not exist
            if no_majority_decision == DecisionWithoutMajority.NONE:
                # do not predict anything since we are unsure
                predicted_label = None
            elif no_majority_decision == DecisionWithoutMajority.RANDOM:
                # predict at random; (useful only for kaggle)
                random.seed(seed)
                predicted_label = random.choice(preds)
            else:
                raise ValueError("Unrecognized ensemble decision type.")
        else:
            # the majority label exists so we use it
            predicted_label = majority_label

        if predicted_label is None:
            label = None
        else:
            label = (predicted_label, id2label[predicted_label])
        predictions.append(Prediction(text=text, label=label))

    return predictions


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
        "--prediction-name",
        type=str,
        default="prediction.tsv",
        help="The path of the .tsv file with the predicted labels and input text.",
    )
    args = parser.parse_args()
    if len(args.models) % 2 == 0:
        raise EnsembleError("Must have an odd number of models in the ensemble.")
    return args


def main(args: argparse.Namespace):
    hydra.initialize(version_base="1.3", config_path="../configs")
    cfg = hydra.compose(config_name="predict.yaml")

    register_resolvers()

    data_module = hydra.utils.instantiate(cfg.data.constructor)

    predictions = ensemble_predict(
        checkpoints=args.models,
        dataset=data_module.predict_dataset,
        model_constructor=cfg.model.constructor_predict,
        no_majority_decision=DecisionWithoutMajority.NONE,
        id2label=cfg.data.id2label,
    )

    df = pd.DataFrame.from_dict(
        {
            "label": [pred.label[1] for pred in predictions],
            "text": [pred.text for pred in predictions],
        }
    )
    if not os.path.exists(cfg.paths.predictions_dir):
        os.makedirs(cfg.paths.predictions_dir)
    df.to_csv(os.path.join(cfg.paths.predictions_dir, args.prediction_name), sep="\t", index=False)


if __name__ == "__main__":
    main(parse_args())
