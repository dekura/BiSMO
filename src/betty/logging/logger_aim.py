import os
from typing import Any, Dict, Optional, Union
from argparse import Namespace
from betty.logging.logger_base import LoggerBase

try:
    import aim

    HAS_AIM = True

except ImportError:
    HAS_AIM = False


from aim.sdk.run import Run
from aim.sdk.repo import Repo
from aim.sdk.utils import clean_repo_path, get_aim_repo_name
from aim.ext.resource.configs import DEFAULT_SYSTEM_TRACKING_INT


class AimLogger(LoggerBase):
    def __init__(
        self,
        repo: Optional[str] = "logs/aim",
        experiment: Optional[str] = None,
        train_metric_prefix: Optional[str] = "train_",
        val_metric_prefix: Optional[str] = "val_",
        test_metric_prefix: Optional[str] = "test_",
        system_tracking_interval: Optional[int] = None,
        log_system_params: Optional[bool] = False,
        capture_terminal_logs: Optional[bool] = True,
        run_name: Optional[str] = None,
        run_hash: Optional[str] = None,
    ):
        super().__init__()

        self._experiment_name = experiment
        self._run_name = run_name
        self._repo_path = repo

        self._train_metric_prefix = train_metric_prefix
        self._val_metric_prefix = val_metric_prefix
        self._test_metric_prefix = test_metric_prefix
        self._system_tracking_interval = None
        self._log_system_params = False
        self._capture_terminal_logs = capture_terminal_logs

        self._run = None
        self._run_hash = run_hash

    @staticmethod
    def _convert_params(params: Union[Dict[str, Any], Namespace]) -> Dict[str, Any]:
        # in case converting from namespace
        if isinstance(params, Namespace):
            params = vars(params)

        if params is None:
            params = {}

        return params

    @property
    def experiment(self) -> Run:
        if self._run is None:
            if self._run_hash:
                self._run = Run(
                    self._run_hash,
                    repo=self._repo_path,
                    system_tracking_interval=self._system_tracking_interval,
                    log_system_params=self._log_system_params,
                    capture_terminal_logs=self._capture_terminal_logs,
                )
                if self._run_name is not None:
                    self._run.name = self._run_name
            else:
                self._run = Run(
                    repo=self._repo_path,
                    experiment=self._experiment_name,
                    system_tracking_interval=self._system_tracking_interval,
                    log_system_params=self._log_system_params,
                    capture_terminal_logs=self._capture_terminal_logs,
                )
                self._run_hash = self._run.hash
        return self._run

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]):
        params = self._convert_params(params)

        # Handle OmegaConf object
        try:
            from omegaconf import OmegaConf
        except ModuleNotFoundError:
            pass
        else:
            # Convert to primitives
            if OmegaConf.is_config(params):
                params = OmegaConf.to_container(params, resolve=True)

        for key, value in params.items():
            self.experiment.set(("hparams", key), value, strict=False)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        # assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        metric_items: Dict[str:Any] = {k: v for k, v in metrics.items()}

        if "epoch" in metric_items:
            epoch: int = metric_items.pop("epoch")
        else:
            epoch = None

        for k, v in metric_items.items():
            name = k
            context = {}
            if self._train_metric_prefix and name.startswith(self._train_metric_prefix):
                name = name[len(self._train_metric_prefix) :]
                context["subset"] = "train"
            elif self._test_metric_prefix and name.startswith(self._test_metric_prefix):
                name = name[len(self._test_metric_prefix) :]
                context["subset"] = "test"
            elif self._val_metric_prefix and name.startswith(self._val_metric_prefix):
                name = name[len(self._val_metric_prefix) :]
                context["subset"] = "val"
            self.experiment.track(v, name=name, step=step, epoch=epoch, context=context)

    def log(self, metrics: Dict[str, float], tag=None, step: Optional[int] = None):
        # assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        metric_items: Dict[str:Any] = {k: v for k, v in metrics.items()}

        if "epoch" in metric_items:
            epoch: int = metric_items.pop("epoch")
        else:
            epoch = None

        for k, v in metric_items.items():
            name = k
            context = {}
            if self._train_metric_prefix and name.startswith(self._train_metric_prefix):
                name = name[len(self._train_metric_prefix) :]
                context["subset"] = "train"
            elif self._test_metric_prefix and name.startswith(self._test_metric_prefix):
                name = name[len(self._test_metric_prefix) :]
                context["subset"] = "test"
            elif self._val_metric_prefix and name.startswith(self._val_metric_prefix):
                name = name[len(self._val_metric_prefix) :]
                context["subset"] = "val"
            self.experiment.track(v, name=name, step=step, epoch=epoch, context=context)


    def finalize(self, status: str = "") -> None:
        # super().finalize(status)
        if self._run:
            self._run.close()
            del self._run
            self._run = None

    def __del__(self):
        self.finalize()

    @property
    def save_dir(self) -> str:
        repo_path = clean_repo_path(self._repo_path) or Repo.default_repo_path()
        return os.path.join(repo_path, get_aim_repo_name())

    @property
    def name(self) -> str:
        return self._experiment_name

    @property
    def version(self) -> str:
        return self.experiment.hash
