"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-03-23 14:56:21
LastEditTime: 2023-05-19 16:44:27
Contact: cgjcuhk@gmail.com
Description: Litho Main Function
"""
from typing import List, Tuple

import aim
import hydra
import pyrootutils
import torch
import torchvision.transforms as T
from lightning import Callback, LightningDataModule, LightningModule, Trainer
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
from src.models.litho import AbbeLitho, Mask, Source
from src.models.litho.utils import torch_arr_bound

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def abbe(cfg: DictConfig) -> Tuple[dict, dict]:
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

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating Mask <{cfg.mask._target_}>")
    m: Mask = hydra.utils.instantiate(cfg.mask)

    log.info(f"Instantiating Aerial using Abbe model <{cfg.aerial._target_}>")
    abbe: AbbeLitho = hydra.utils.instantiate(cfg.aerial, source=s, mask=m)

    ai, ri = abbe.abbe_litho()

    aerial_path = "/home/gjchen21/projects/smo/SMO-torch/data/ibm_opc_test/AI/t1_0_mask.png"
    resist_path = "/home/gjchen21/projects/smo/SMO-torch/data/ibm_opc_test/RI/t1_0_mask.png"
    aerial_gt = Mask(layout_path=aerial_path, target_path=aerial_path)
    aerial_gt.open_layout()
    aerial_gt.data = aerial_gt.data.to("cuda:0")
    # print(aerial_gt.data)
    # torch_arr_bound(aerial_gt.data, "aerial_gt")
    resist_gt = Mask(layout_path=resist_path, target_path=aerial_path)
    resist_gt.open_layout()
    # torch_arr_bound(resist_gt.data, "resist_gt")
    resist_gt.data = resist_gt.data.to("cuda:0")
    aerial_loss = torch.abs(torch.sum(aerial_gt.data - ai))
    resist_loss = torch.abs(torch.sum(resist_gt.data - ri))
    # print(aerial_loss)
    logger[0].log_metrics({"aerial": aerial_loss, "resist": resist_loss})
    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)
    transform = T.ToPILImage()
    aim_images = [aim.Image(transform(i)) for i in [ai, ri, aerial_gt.data, resist_gt.data]]
    logger[0].experiment.track(
        value=aim_images,
        name="AI and RI",
        context={"sig_out": cfg.source.sigma_out, "sig_in": cfg.source.sigma_in},
    )

    hparams = {}

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")
    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")
    hparams["source"] = cfg.get("source")
    hparams["mask"] = cfg.get("mask")

    if logger:
        log.info("Logging hyperparameters!")
        # utils.log_hyperparameters(object_dict)
        for ll in logger:
            ll.log_hyperparams(hparams)

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="abbe.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    metric_dict, _ = abbe(cfg)

    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
