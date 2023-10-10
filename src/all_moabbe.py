"""
Description: MO with Abbe's Approach.
For 10 testcases.
"""
from typing import Any, List, Dict, Tuple, Optional

import hydra
import pyrootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils
from src.models.litho import Mask, Source

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def mo(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    # assert cfg.ckpt_path
    # TODO What should be assert?

    log.info(f"Instantiating Source <{cfg.source._target_}>")
    s: Source = hydra.utils.instantiate(cfg.source)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating Mask <{cfg.mask._target_}>")
    m: Mask = hydra.utils.instantiate(cfg.mask)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, source=s, mask=m)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    object_dict = {
        "cfg": cfg,
        "source": s,
        "logger": logger,
        "mask": m,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("smo"):
        log.info("Starting SMO training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("inference"):
        log.info("Starting Inference!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")
    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="mo_testcase1.yaml")
def main_testcase1(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)

    metric_dict, _ = mo(cfg)

    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value

@hydra.main(version_base="1.3", config_path="../configs", config_name="mo_testcase2.yaml")
def main_testcase2(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)

    metric_dict, _ = mo(cfg)

    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value

@hydra.main(version_base="1.3", config_path="../configs", config_name="mo_testcase3.yaml")
def main_testcase3(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)

    metric_dict, _ = mo(cfg)

    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value

@hydra.main(version_base="1.3", config_path="../configs", config_name="mo_testcase4.yaml")
def main_testcase4(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)

    metric_dict, _ = mo(cfg)

    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value

@hydra.main(version_base="1.3", config_path="../configs", config_name="mo_testcase5.yaml")
def main_testcase5(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)

    metric_dict, _ = mo(cfg)

    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value

@hydra.main(version_base="1.3", config_path="../configs", config_name="mo_testcase6.yaml")
def main_testcase6(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)

    metric_dict, _ = mo(cfg)

    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value

@hydra.main(version_base="1.3", config_path="../configs", config_name="mo_testcase7.yaml")
def main_testcase7(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)

    metric_dict, _ = mo(cfg)

    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value

@hydra.main(version_base="1.3", config_path="../configs", config_name="mo_testcase8.yaml")
def main_testcase8(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)

    metric_dict, _ = mo(cfg)

    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value

@hydra.main(version_base="1.3", config_path="../configs", config_name="mo_testcase9.yaml")
def main_testcase9(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)

    metric_dict, _ = mo(cfg)

    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value

@hydra.main(version_base="1.3", config_path="../configs", config_name="mo_testcase10.yaml")
def main_testcase10(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)

    metric_dict, _ = mo(cfg)

    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value

if __name__ == "__main__":
    main_testcase1()
    main_testcase2()
    main_testcase3()
    main_testcase4()
    main_testcase5()
    main_testcase6()
    main_testcase7()
    main_testcase8()
    main_testcase9()
    main_testcase10()