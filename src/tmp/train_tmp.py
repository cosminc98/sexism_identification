import re

import emoji
import torch
import torch.nn as nn
from datasets import concatenate_datasets, load_dataset, load_metric
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_CKPT = "dumitrescustefan/bert-base-romanian-uncased-v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

INT_TO_STR = {0: "descriptive", 1: "direct", 2: "non-offensive", 3: "offensive", 4: "reporting"}
STR_TO_INT = {"descriptive": 0, "direct": 1, "non-offensive": 2, "offensive": 3, "reporting": 4}

seed = 66


def normalize(batch):
    """
    This function should be used before tokenizing the input string.

    Normalizes the input string in the following ways:
    -> Converts from ş to ș, ţ to ț, etc.
    -> Converts @mention to USER, #hashtag to HASHTAG, http... and www... to HTTPURL
    -> Converts emoticons to :emoji_with_long_name:
    -> Replaces :emoji_with_long_name: with emoji_with_long_name and replaces _, : and - with empty string
    -> Removes multiple whitespaces with a single whitespace
    """

    sentence = batch["text"]

    # Make sure it's a string
    sentence = str(sentence)

    # Convert from ş to ș, ţ to ț, etc.
    sentence = re.sub(r"ş", "ș", sentence)
    sentence = re.sub(r"Ş", "Ș", sentence)
    sentence = re.sub(r"ţ", "ț", sentence)
    sentence = re.sub(r"Ţ", "Ț", sentence)

    # Convert @mentions to USER, #hashtags to HASHTAG, http... and www... to HTTPURL
    sentence = re.sub(r"@\S+", "USER", sentence)
    sentence = re.sub(r"#\S+", "HASHTAG", sentence)
    sentence = re.sub(r"http\S+", "HTTPURL", sentence)
    sentence = re.sub(r"www\S+", "HTTPURL", sentence)

    # Convert emoticons to :emoji_with_long_name:
    sentence = emoji.demojize(sentence, delimiters=(" :", ": "))

    # Replace :emoji_with_long_name: with emojiwithlongname
    sentence = re.sub(r":\S+:", lambda x: x.group(0).replace("_", "").replace(":", "").replace("-", ""), sentence)

    # Remove multiple whitespaces with a single whitespace
    sentence = re.sub(r"\s+", " ", sentence)

    return {"text": sentence}


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # choose the predicted class (from an array of probabilites)

    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    bacc = balanced_accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "balanced_accuracy": bacc}


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([26.315, 18.181, 1.265, 9.090, 200.0]).to(device))
        # loss_fct = nn.CrossEntropyLoss(
        #     weight=torch.tensor([15.873, 11.111, 1.538, 5.555, 111.111]).to(device)
        # )  # [, , , , ]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def train(index: int, dataset_tokenized):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CKPT,
        num_labels=5,
        id2label=INT_TO_STR,
        label2id=STR_TO_INT,
        classifier_dropout=0.1,
    )

    training_args = TrainingArguments(
        output_dir=f"nitro-robertlarge-nlp-v1.9.{index}",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    del trainer


def ensamble_train(index=1, index_start=0, index_end=5):
    seeds = []
    for i in range(0, index):
        seeds.append(seed + i)

    for i in range(0, index):
        if i >= index_start:
            print("SEED: ", seed)
            ds_split = ds.train_test_split(test_size=0.2, stratify_by_column="label", seed=seeds[i])
            ds_split = ds_split.map(lambda batch: normalize(batch), batched=False)
            ds_tok_split = ds_split.map(lambda batch: tokenize(batch), batched=True, batch_size=None)
            ds_tok_split.set_format("torch", columns=["input_ids", "attention_mask", "label"])

            train(i, ds_tok_split)
        else:
            ds_split = ds.train_test_split(test_size=0.2, stratify_by_column="label", seed=seed)


if __name__ == "__main__":
    ds = load_dataset("csv", data_files={"data": "../data/train_data.csv"})
    ds = ds.rename_column("Final Labels", "label")
    ds = ds.rename_column("Text", "text")
    ds = ds.remove_columns(["Id"])
    ds = ds["data"]
    ds = ds.class_encode_column("label")

    ensamble_train(1, 0, 1)
