from typing import Optional

import hydra
import lightning as L
import pyrootutils
from omegaconf import DictConfig

from trainers.imbalanced_dataset_trainer import ImbalancedDatasetTrainer
from utils.config import register_resolvers

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def train(cfg: DictConfig):
    data_module = hydra.utils.instantiate(cfg.data.constructor)

    model = hydra.utils.instantiate(cfg.model.constructor)

    training_args = hydra.utils.instantiate(cfg.trainer)

    trainer = ImbalancedDatasetTrainer(
        model=model,
        key_dataset_labels=cfg.data.key_labels,
        key_model_labels=cfg.model.key_labels,
        key_model_logits=cfg.model.key_logits,
        args=training_args,
        train_dataset=data_module.train_dataset,
        eval_dataset=data_module.val_dataset,
        tokenizer=data_module.tokenizer,
        data_collator=data_module.data_collator,
    )

    trainer.train()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # add config resolvers such as "len" which allows computing the length
    # of a list from the config in another configuration variable
    register_resolvers()

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    train(cfg)


if __name__ == "__main__":
    main()
