import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_train_config(cfg_train: DictConfig):
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data.constructor)
    hydra.utils.instantiate(cfg_train.model.constructor)


def test_test_config(cfg_test: DictConfig):
    assert cfg_test
    assert cfg_test.data
    assert cfg_test.model

    HydraConfig().set_config(cfg_test)

    hydra.utils.instantiate(cfg_test.data.constructor)
    hydra.utils.instantiate(cfg_test.model.constructor_predict)
