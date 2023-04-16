from collections import OrderedDict, defaultdict
from typing import Dict, List, Union

import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from transformers import (
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)


class ImbalancedDatasetTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        args: TrainingArguments,
        data_collator: DataCollator,
        train_dataset: Dataset,
        eval_dataset: Union[Dataset, Dict[str, Dataset]],
        key_dataset_labels: str,
        key_model_labels: str,
        key_model_logits: str,
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

        self.key_dataset_labels = key_dataset_labels
        self.key_model_labels = key_model_labels
        self.key_model_logits = key_model_logits

        counts: Dict[int, int] = defaultdict(lambda: 0)
        for sample in train_dataset:
            counts[int(sample[self.key_dataset_labels])] += 1

        counts = OrderedDict(sorted(counts.items()))

        self.imbalance_weights: List[float] = []
        for class_count in counts.values():
            # compute the frequency of this class in the dataset
            frequency = class_count / len(train_dataset)

            # compute the class weight to counteract class imbalance
            self.imbalance_weights.append(1 / frequency)

        self.imbalance_weights = torch.tensor(self.imbalance_weights).to(self.args.device)

    def compute_metrics(self, pred: EvalPrediction):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        bacc = balanced_accuracy_score(labels, preds)

        return {"accuracy": acc, "f1": f1, "balanced_accuracy": bacc}

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get(self.key_model_labels)
        outputs = model(**inputs)
        logits = outputs.get(self.key_model_logits)
        loss_fct = nn.CrossEntropyLoss(weight=self.imbalance_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
