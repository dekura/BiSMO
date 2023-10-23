"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-10-22 09:37:20
LastEditTime: 2023-10-22 20:36:44
Contact: cgjcuhk@gmail.com
Description: 
"""
from lightning.pytorch.utilities import rank_zero_only

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    # model = object_dict["model"]
    trainer = object_dict["engine"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    # hparams["model"] = cfg["model"]

    # # save number of model parameters
    # hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    # hparams["model/params/trainable"] = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad
    # )
    # hparams["model/params/non_trainable"] = sum(
    #     p.numel() for p in model.parameters() if not p.requires_grad
    # )

    # hparams["data"] = cfg["data"]
    hparams["engine"] = cfg["engine"]
    hparams["source"] = cfg.get("source")
    hparams["mask"] = cfg.get("mask")
    hparams["mo_module"] = cfg.get("module.mo")
    hparams["so_module"] = cfg.get("module.so")


    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")


    # send hparams to all loggers
    # for logger in trainer.loggers:
    trainer.logger.log_hyperparams(hparams)
