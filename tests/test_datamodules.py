from pathlib import Path

import pytest
import torch

from src.data.coroseof_datamodule import COROSEOFDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_coroseof_datamodule(batch_size):
    dm = COROSEOFDataModule(
        train_path="./data/ro/train_data.csv",
        test_path="./data/ro/test_data.csv",
        key_labels="labels",
        tokenizer_name_or_path="dumitrescustefan/bert-base-romanian-uncased-v1",
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )

    dm.setup("fit")
    assert dm.train_dataset

    num_datapoints = len(dm.train_dataset)
    assert num_datapoints == 31_206

    batch = next(iter(dm.train_dataloader()))
    x, y = batch["input_ids"], batch["labels"]
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
