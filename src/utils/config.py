from typing import Dict, Iterable
from omegaconf import OmegaConf


def id2label(label_names: Iterable[str]) -> Dict[int, str]:
    return {index: name for index, name in enumerate(sorted(label_names))}


def label2id(label_names: Iterable[str]) -> Dict[str, int]:
    return {name: index for index, name in enumerate(sorted(label_names))}


def register_resolvers():
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    OmegaConf.register_new_resolver("id2label", id2label)
    OmegaConf.register_new_resolver("label2id", label2id)
