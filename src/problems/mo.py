"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-10-22 13:05:39
LastEditTime: 2023-11-20 10:48:13
Contact: cgjcuhk@gmail.com
Description: the definition of Source optimization problem.
"""

import aim
import torch
import torch.nn as nn
import torchvision.transforms as T

from src.betty.configs import Config, EngineConfig
from src.betty.logging.logger_aim import AimLogger
from src.betty.problems import ImplicitProblem


class MO(ImplicitProblem):
    def __init__(
        self,
        config: Config,
        module: nn.Module,
        optimizer_cfg: torch.optim.Optimizer,
        scheduler_cfg: torch.optim.lr_scheduler,
        train_data_loader=None,
        name: str = "MO",
        weight_l2: float = 1000,
        weight_pvb: float = 3000,
        vis_in_train: bool = False,
        device: str = "cuda:0",
    ):
        super().__init__(
            name,
            config,
            module,
            # train_data_loader,
            # optimizer,
            # scheduler,
        )
        self.module = module
        self.train_data_loader = train_data_loader
        self.sigmoid_source = nn.Sigmoid()
        self.criterion = nn.MSELoss()
        self.weight_l2 = weight_l2
        self.weight_pvb = weight_pvb
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.vis_in_train = vis_in_train
        # self.device = torch.device(device)
        self.device_id = device
        self.min_loss = 1e10
        self.min_pvb = 1e10
        self.min_l2 = 1e10

    def update_source_value(self, source_params):
        if self.SO.module.source_acti == "cosine":
            self.source_value = (1 + torch.cos(source_params)) / 2
        elif self.SO.module.source_acti == "sigmoid":
            self.source_value = self.sigmoid_source(
                self.SO.module.source_sigmoid_steepness * source_params
            )
        else:
            self.source_value = (1 + torch.cos(source_params)) / 2

    def forward(self):
        inner_source_param = self.SO.module.source_params
        self.update_source_value(inner_source_param)
        mo_intensity2D_list, mo_RI_list = self.module(self.source_value)
        return mo_intensity2D_list, mo_RI_list

    def get_mins(self, loss, l2, pvb):
        if loss.clone().detach().item() < self.min_loss:
            self.min_loss = loss.clone().detach().item()
        if l2.clone().detach().item() < self.min_l2:
            self.min_l2 = l2.clone().detach().item()
        if pvb.clone().detach().item() < self.min_pvb:
            self.min_pvb = pvb.clone().detach().item()

    def training_step(self, batch):
        AIlist, RIlist = self.forward()
        # binary RI, torch.where creates a tensor with require_grad = False
        RI_min = torch.where(RIlist[0] > 0.5, 1.0, 0.0).float()
        RI_norm = torch.where(RIlist[1] > 0.5, 1.0, 0.0).float()
        RI_max = torch.where(RIlist[2] > 0.5, 1.0, 0.0).float()
        target = self.module.mask.target_data.float()

        # RI_pvb = torch.where(RI_min != RI_max, 1.0, 0.0).float()

        l2 = self.criterion(RIlist[1], target)
        # pvb = self.criterion(RIlist[1], RIlist[0]) + self.criterion(RIlist[1], RIlist[2])
        pvb = self.criterion(target, RIlist[0]) + self.criterion(target, RIlist[2])
        loss = l2 * self.weight_l2 + pvb * self.weight_pvb

        l2_val = (RI_norm - self.module.mask.target_data).abs().sum()
        # pvb_val = (RI_norm - RI_min).abs().sum() + (RI_norm - RI_max).abs().sum()
        other_pvb_val = (RI_max - RI_min).abs().sum()

        self.get_mins(loss, l2_val, other_pvb_val)
        self.log(
            {
                "train/loss": loss.clone().detach(),
                "train/l2": l2_val.detach().clone(),
                "train/pvb": other_pvb_val.detach().clone(),
                "min/loss": self.min_loss,
                "min/l2": self.min_l2,
                "min/pvb": self.min_pvb,
            },
            global_step=None,
        )

        if self.vis_in_train:
            if self.is_rank_zero():
                mask_moed = torch.where(self.module.mask_value > 0.5, 1.0, 0.0).float()
                transform = T.ToPILImage()
                aim_images = [
                    aim.Image(transform(i))
                    for i in [
                        self.source_value.clone().detach(),
                        mask_moed.clone().detach(),
                        self.module.mask_value.detach().clone(),
                        RI_norm.detach().clone(),
                        self.module.mask.target_data.clone().detach(),
                    ]
                ]
                self.logger.experiment.track(
                    value=aim_images,
                    name=f"train epoch {self._count}",
                    step=self._count,
                    context={"train epoch": self._count},
                )

        return {"loss": loss}

    def configure_optimizer(self):
        optimizer = self.optimizer_cfg(params=self.module.parameters())
        return optimizer

    def configure_scheduler(self):
        scheduler = self.scheduler_cfg(optimizer=self.optimizer)
        return scheduler

    def configure_device(self, device):
        self.device = torch.device(self.device_id)
