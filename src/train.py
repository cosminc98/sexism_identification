from typing import Optional

import hydra
import lightning as L
import pyrootutils
import torch
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
    return trainer.evaluate()


def get_metric_value(metric_dict: dict, metric_name: Optional[str]) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if metric_name is None:
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name]

    return metric_value


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # add config resolvers such as "len" which allows computing the length
    # of a list from the config in another configuration variable
    register_resolvers()

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # train the model
    metric_dict = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # avoid memory fragmentation in multiruns
    torch.cuda.empty_cache()

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
