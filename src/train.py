from collections import OrderedDict, defaultdict
from typing import Dict, List, Union

import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from constants import BASE_MODEL, N_LABELS, SEED
from data.coroseof_datamodule import COROSEOFDataModule


class CustomTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        args: TrainingArguments,
        data_collator: DataCollator,
        train_dataset: Dataset,
        eval_dataset: Union[Dataset, Dict[str, Dataset]],
        tokenizer: PreTrainedTokenizerBase,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
        )

        counts: Dict[int, int] = defaultdict(lambda: 0)
        for sample in train_dataset:
            counts[int(sample["label"])] += 1

        counts = OrderedDict(sorted(counts.items()))

        self.imbalance_weights: List[float] = []
        for class_count in counts.values():
            # compute the frequency of this class in the dataset
            frequency = class_count / len(train_dataset)

            # compute the class weight to counteract class imbalance
            self.imbalance_weights.append(1 / frequency)

        self.imbalance_weights = torch.tensor(self.imbalance_weights).to(self.args.device)

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        bacc = balanced_accuracy_score(labels, preds)

        return {"accuracy": acc, "f1": f1, "balanced_accuracy": bacc}

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.imbalance_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train(index: int, data_module: COROSEOFDataModule):
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=N_LABELS,
        id2label=COROSEOFDataModule.INT_TO_STR,
        label2id=COROSEOFDataModule.STR_TO_INT,
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
        train_dataset=data_module.train_dataset,
        eval_dataset=data_module.val_dataset,
        tokenizer=data_module.tokenizer,
        data_collator=data_module.data_collator,
    )

    trainer.train()


def ensamble_train(index=1, index_start=0, index_end=5):
    seeds = []
    for i in range(0, index):
        seeds.append(SEED + i)

    for i in range(0, index):
        dm = COROSEOFDataModule(BASE_MODEL, seed=seeds[i])
        dm.setup(COROSEOFDataModule.STAGE_FIT)

        if i >= index_start:
            train(i, data_module=dm)


if __name__ == "__main__":
    ensamble_train(1, 0, 1)
