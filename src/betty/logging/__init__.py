"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-10-22 19:09:23
LastEditTime: 2023-11-13 09:29:05
Contact: cgjcuhk@gmail.com
Description: 
"""
from logging import Logger
from .logger_tensorboard import TensorBoardLogger
from .logger_wandb import WandBLogger
from .logger_base import get_logger, LoggerBase
from .logger_aim import AimLogger


logger_mapping = {
    "tensorboard": TensorBoardLogger,
    "wandb": WandBLogger,
    "none": LoggerBase,
    "aim": AimLogger
}


def type_check(logger_type):
    assert logger_type in logger_mapping

    if logger_type == "wandb":
        try:
            import wandb
        except ImportError:
            get_logger().warning(
                "[!] WandB is not installed. The default logger will be instead used."
            )
            logger_type = "none"
    elif logger_type == "tensorboard":
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            get_logger().warning(
                "[!] Tensorboard is not installed. The default logger will be instead used."
            )
            logger_type = "none"
    elif logger_type == "aim":
        try:
            import aim
        except ImportError:
            get_logger().warning(
                "[!] Aim logger is not installed. The default logger will be instead used."
            )
            logger_type = "none"

    return logger_type


def logger(
        logger_type="none",
        logger_repo="none",
        ):
    logger_type = type_check(logger_type)
    if logger_type == "aim":
        return logger_mapping[logger_type](repo=logger_repo)
    else:
        return logger_mapping[logger_type]()
