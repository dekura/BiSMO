"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-03-23 14:56:21
LastEditTime: 2023-04-06 19:24:46
Contact: cgjcuhk@gmail.com
Description: Litho Main Function
"""
from typing import List, Tuple

import hydra
import pyrootutils

# from lightning import LightningDataModule, LightningModule, Trainer
# from src.models.litho import Source, LensList, TCCList, Mask, AerialList
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils
from src.models.litho import AerialList, LensList, Mask, Source, TCCList

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def litho(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # assert cfg.ckpt_path
    # TODO What should be assert?

    log.info(f"Instantiating Source <{cfg.source._target_}>")
    s: Source = hydra.utils.instantiate(cfg.source)

    log.info(f"Instantiating Lens <{cfg.lens._target_}>")
    o: LensList = hydra.utils.instantiate(cfg.lens)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating TCCList <{cfg.tcc._target_}>")
    t: TCCList = hydra.utils.instantiate(cfg.tcc, source=s, lensList=o)

    log.info(f"Instantiating Mask <{cfg.mask._target_}>")
    m: Mask = hydra.utils.instantiate(cfg.mask)

    log.info(f"Instantiating Aerial and Resist <{cfg.aerial._target_}>")
    a: AerialList = hydra.utils.instantiate(cfg.aerial, mask=m, tccList=t)

    log.info("Starting Litho!")
    a.litho()

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    object_dict = {
        "cfg": cfg,
        "source": s,
        "lenslist": o,
        "logger": logger,
        "TCCList": t,
        "aerial": a,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # metric_dict = trainer.callback_metrics
    # return metric_dict, object_dict
    return {}, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="litho.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    litho(cfg)


if __name__ == "__main__":
    main()
