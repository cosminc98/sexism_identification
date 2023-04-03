import csv

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
)

from train_tmp import (
    INT_TO_STR,
    STR_TO_INT,
    CustomTrainer,
    compute_metrics,
    data_collator,
    normalize,
    tokenize,
    tokenizer,
)


def load_model(checkpoint_path: str, ds_tok):
    model2 = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        num_labels=5,
        id2label=INT_TO_STR,
        label2id=STR_TO_INT,
        classifier_dropout=0.1,
    )

    training_args_ft = TrainingArguments(
        output_dir=checkpoint_path,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer2 = CustomTrainer(
        model=model2,
        args=training_args_ft,
        train_dataset=ds_tok,
        eval_dataset=ds_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer2


ds_eval = load_dataset("csv", data_files={"data": "../data/test_data.csv"})
ds_eval = ds_eval.rename_column("Text", "text")
ds_eval = ds_eval.rename_column("Id", "id")
ds_eval_data = ds_eval["data"]
ds_eval_data = ds_eval_data.map(lambda batch: normalize(batch), batched=False)
ds_eval_tok = ds_eval_data.map(lambda batch: tokenize(batch), batched=True, batch_size=None)
ds_eval_data[0]


ensamble_list = ["./nitro-robertlarge-nlp-v1.9.0/checkpoint-1952"]

ensamble_predictions = []
for model in ensamble_list:
    trainer = load_model(model, ds_eval_tok)

    predictions = trainer.predict(ds_eval_tok)
    ensamble_predictions.append(predictions)

    del trainer


final_ensamble_prediction = ensamble_predictions[0].predictions

for i in range(1, len(ensamble_predictions)):
    print(i)
    final_ensamble_prediction = final_ensamble_prediction + ensamble_predictions[i].predictions


preds = np.argmax(np.array(final_ensamble_prediction), axis=-1)

df = pd.DataFrame({})

with open(f"./subs/nitro-robertweet-nlp-v2.1.0.csv", "w", newline="") as csvfile:
    data = []
    for i, pred in enumerate(preds):
        data.append([i, INT_TO_STR[pred]])

    header = ["Id", "Label"]
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(data)
