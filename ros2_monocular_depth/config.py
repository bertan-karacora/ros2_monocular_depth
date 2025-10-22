"""Global package config module.
Module globals are used for config attributes.
Permanent config settings are set in a top-level config file within the package and indicated by an underscore prefix.

Note: Although generally to be avoided, using global attributes is considered a good option for setting up a python package config that works as a global singleton.
See: https://stackoverflow.com/questions/5055042/whats-the-best-practice-using-a-settings-file-in-python
See: https://stackoverflow.com/questions/30556857/creating-a-static-class-with-no-instances
"""

import importlib.resources as resources
import logging
from pathlib import Path
import pprint
import random

import numpy as np
import torch

import ros2_monocular_depth.libs.utils_io as utils_io

_LOGGER = logging.getLogger(__name__)
_PATH_CONFIG_PACKAGE = str(resources.files(__package__) / "config.yaml")
# Will be overwritten if specified in config.yaml
_SEED_RNGS = 42


def _init():
    apply_config(Path(_PATH_CONFIG_PACKAGE), use_private=True)
    seed_rngs()

    _LOGGER.debug(f"Config initialized")


def get_attributes(use_private: bool = False):
    attributes = {}
    for key, value in globals().items():
        is_attribute = key.isupper()
        is_private = key[0] == "_"

        if is_attribute(key) and (use_private or not is_private(key)):
            attributes[key.lower()] = value

    return attributes


def set_attributes(attributes: dict, use_private: bool = False):
    for key, value in attributes.items():
        key = key.upper() if not use_private else f"_{key.upper()}"

        globals()[key] = value

        _LOGGER.debug(f"Config attribute {key} set to {value}")


def seed_rngs():
    random.seed(_SEED_RNGS)
    np.random.seed(_SEED_RNGS)
    torch.manual_seed(_SEED_RNGS)

    _LOGGER.info(f"RNGs seeded with seed {_SEED_RNGS}")


def dump():
    attributes = get_attributes(use_private=True)

    pprint.pprint(attributes)


def apply_config(path: Path, use_private=False):
    attributes = utils_io.read_yaml(path)
    set_attributes(attributes, use_private=use_private)

    _LOGGER.info(f"Config loaded from {path}")


def save(path_dir: Path):
    attributes = get_attributes(use_private=False)
    path_config = path_dir / "config.yaml"

    utils_io.write_yaml(attributes, path_config)

    _LOGGER.info(f"Config saved to {path_config}")


def apply_config_preset(name: str):
    """Apply config from set of configs integrated in the python package.
    Config name may be prepended by directory names but does not include file suffix.
    """
    path_config = resources.files(__package__) / "configs" / f"{name}.yaml"

    apply_config(path_config)


def apply_config_exp(path_dir_exp: Path):
    path_config = path_dir_exp / "config.yaml"

    apply_config(path_config)


def list_available():
    path_dir_configs = resources.files(__package__) / "configs"
    paths_config = sorted(path_dir_configs.glob("**/*.yaml"))

    names_available = [str(path_config.parent.relative_to(path_dir_configs) / path_config.stem) for path_config in paths_config]

    return names_available


_init()
