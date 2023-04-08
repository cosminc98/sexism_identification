import csv

import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, Trainer

from constants import BASE_MODEL, SEED
from data.coroseof_datamodule import COROSEOFDataModule


def load_model(checkpoint_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

    trainer = Trainer(model=model)

    return trainer


def main():
    ensamble_list = ["./nitro-robertlarge-nlp-v1.9.0/checkpoint-1952"]

    dm = COROSEOFDataModule(BASE_MODEL, seed=SEED)
    dm.setup(COROSEOFDataModule.STAGE_TEST)

    ensamble_predictions = []
    for model in ensamble_list:
        trainer = load_model(model)

        predictions = trainer.predict(dm.test_dataset)
        ensamble_predictions.append(predictions)

        del trainer

    final_ensamble_prediction = ensamble_predictions[0].predictions

    for i in range(1, len(ensamble_predictions)):
        final_ensamble_prediction = final_ensamble_prediction + ensamble_predictions[i].predictions

    preds = np.argmax(np.array(final_ensamble_prediction), axis=-1)

    with open("./subs/nitro-robertweet-nlp-v2.1.0.csv", "w", newline="") as csvfile:
        data = []
        for i, pred in enumerate(preds):
            data.append([i, COROSEOFDataModule.INT_TO_STR[pred]])

        header = ["Id", "Label"]
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data)


if __name__ == "__main__":
    main()
