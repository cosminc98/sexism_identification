import re
from typing import Optional, TypeVar

import datasets
import emoji
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

DatasetType = TypeVar("DatasetType", datasets.Dataset, datasets.DatasetDict)


class COROSEOFDataModule(LightningDataModule):
    STAGE_FIT = "fit"
    STAGE_TEST = "test"
    STAGE_PREDICT = "predict"

    KEY_TEXT = "text"

    def __init__(
        self,
        train_path: str,
        test_path: str,
        key_labels: str,
        tokenizer_name_or_path: str,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        stage: Optional[str] = None,
        seed: int = 666013,
    ):
        super().__init__()
        self.key_labels = key_labels
        self.train_path = train_path
        self.test_path = test_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=True)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        if stage is not None:
            self.setup(stage)

    def setup(self, stage: str):
        if stage == COROSEOFDataModule.STAGE_FIT:
            dataset: datasets.DatasetDict = datasets.load_dataset(
                "csv", data_files={"data": self.train_path}
            )
            dataset = dataset.rename_columns(
                {"Final Labels": self.key_labels, "Text": self.KEY_TEXT}
            )
            dataset = dataset.remove_columns(["Id"])
            dataset: datasets.Dataset = dataset["data"]
            dataset = dataset.class_encode_column(self.key_labels)

            # perform train / validation split
            ds_split = dataset.train_test_split(
                test_size=0.2, stratify_by_column=self.key_labels, seed=self.seed
            )

            ds_split = self.process_dataset(ds_split)
            ds_split.set_format("torch", columns=["input_ids", "attention_mask", self.key_labels])

            self.train_dataset = ds_split["train"]
            self.val_dataset = ds_split["test"]

        elif stage == COROSEOFDataModule.STAGE_TEST:
            dataset = datasets.load_dataset("csv", data_files={"data": self.test_path})
            dataset = dataset.rename_columns({"Text": "text", "Id": "id"})
            dataset = dataset["data"]

            dataset = self.process_dataset(dataset)

            self.test_dataset = dataset

        elif stage == COROSEOFDataModule.STAGE_PREDICT:
            raise NotImplementedError(f'Stage "{stage}" not implemented.')

        else:
            raise ValueError(f'Stage "{stage}" not recognized.')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size)

    def tokenize(self, batch):
        return self.tokenizer(
            batch["text"], padding=True, truncation=True, max_length=self.max_seq_length
        )

    def normalize(self, batch):
        """This function should be used before tokenizing the input string.

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
        sentence = re.sub(
            r":\S+:",
            lambda x: x.group(0).replace("_", "").replace(":", "").replace("-", ""),
            sentence,
        )

        # Remove multiple whitespaces with a single whitespace
        sentence = re.sub(r"\s+", " ", sentence)

        return {"text": sentence}

    def process_dataset(self, dataset: DatasetType) -> DatasetType:
        # normalize text
        dataset = dataset.map(lambda batch: self.normalize(batch), batched=False)

        # transform text into sequence of token ids
        dataset = dataset.map(lambda batch: self.tokenize(batch), batched=True, batch_size=None)

        return dataset
