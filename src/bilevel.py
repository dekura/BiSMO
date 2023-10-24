from typing import Any, Dict, List, Optional, Tuple

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("eval", eval)

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
from src.betty.configs import Config, EngineConfig
from src.betty.engine import Engine
from src.engine.smo_engine import SMOEngine
from src.models.litho import Mask, Source
from src.models.mo_module import MO_Module
from src.models.so_module import SO_Module
from src.problems.mo import MO
from src.problems.so import SO

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def bismo(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # assert cfg.ckpt_path
    # TODO What should be assert?

    # to save memory on torch.fft
    torch.backends.cuda.cufft_plan_cache[2].max_size = int(cfg.cufft_max_cache_size)

    log.info(f"Instantiating Source <{cfg.source._target_}>")
    s: Source = hydra.utils.instantiate(cfg.source)

    # log.info("Instantiating loggers...")
    # logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating Mask <{cfg.mask._target_}>")
    m: Mask = hydra.utils.instantiate(cfg.mask)

    log.info(f"Instantiating MO model <{cfg.module.mo._target_}>")
    mo_module: MO_Module = hydra.utils.instantiate(cfg.module.mo, source=s, mask=m)

    log.info(f"Instantiating SO model <{cfg.module.so._target_}>")
    so_module: SO_Module = hydra.utils.instantiate(cfg.module.so, source=s, mask=m)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    dataloader = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating outer MO problem <{cfg.problems.mo._target_}>")
    outer: MO = hydra.utils.instantiate(
        cfg.problems.mo, module=mo_module, train_data_loader=dataloader.train_dataloader()
    )

    log.info(f"Instantiating inner SO problem <{cfg.problems.so._target_}>")
    inner: SO = hydra.utils.instantiate(
        cfg.problems.so, module=so_module, train_data_loader=dataloader.train_dataloader()
    )

    log.info(f"Instantiating Engine config <{cfg.engine._target_}>")
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    problems = [outer, inner]
    l2u = {inner: [outer]}
    u2l = {outer: [inner]}
    dependencies = {"l2u": l2u, "u2l": u2l}

    log.info("Instantiating Engine")
    engine = SMOEngine(config=engine_config, problems=problems, dependencies=dependencies)
    engine.run()

    object_dict = {
        "cfg": cfg,
        "source": s,
        "mask": m,
        "mo_module": mo_module,
        "so_module": so_module,
        "engine": engine,
    }

    if engine.logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)
    metric_dict = {}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="bismo.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    metric_dict, object_dict = bismo(cfg)

    # metric_value = utils.get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )
    # # return optimized metric
    return object_dict


if __name__ == "__main__":
    main()
