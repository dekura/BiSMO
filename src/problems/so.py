"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-10-22 13:05:39
LastEditTime: 2023-10-22 17:04:23
Contact: cgjcuhk@gmail.com
Description: the defination of Source optimization problem.
"""


import torch
import torch.nn as nn
from torch import optim

from src.betty.configs import Config, EngineConfig
from src.betty.problems import ImplicitProblem




class SO(ImplicitProblem):
    def __init__(self,
                config: Config,
                module: nn.Module,
                optimizer_cfg: torch.optim.Optimizer,
                # scheduler_cfg: torch.optim.lr_scheduler,
                train_data_loader=None,
                name: str = "SO",
                weight_l2: float = 1000,
                weight_pvb: float = 8000,
                ):
        super().__init__(
            name,
            config,
            module,
            # train_data_loader,
            # optimizer,
            # scheduler,
        )
        self.train_data_loader = train_data_loader
        self.sigmoid_mask = nn.Sigmoid()
        self.criterion = nn.MSELoss()
        self.weight_l2 = weight_l2
        self.weight_pvb = weight_pvb
        self.optimizer_cfg = optimizer_cfg
        # self.scheduler_cfg = scheduler_cfg

    def update_mask_value(self, mask_params):
        if self.MO.module.mask_acti == 'sigmoid':
            # mask after activation func
            self.mask_value = self.sigmoid_mask(
                self.MO.module.mask_sigmoid_steepness * mask_params
            )
        else:
            self.mask_value = self.sigmoid_mask(
                self.MO.module.mask_sigmoid_steepness * mask_params
            )


    def forward(self):
        outer_mask_param = self.MO.module.mask_params
        self.update_mask_value(outer_mask_param)
        so_intensity2D_list, so_RI_list = self.module(self.mask_value)
        return so_intensity2D_list, so_RI_list

    def training_step(self, batch):
        AIlist, RIlist = self.forward()
        # binary RI, torch.where creates a tensor with require_grad = False
        RI_min = torch.where(RIlist[0] > 0.5, 1.0, 0.0).float()
        RI_norm = torch.where(RIlist[1] > 0.5, 1.0, 0.0).float()
        RI_max = torch.where(RIlist[2] > 0.5, 1.0, 0.0).float()

        RI_pvb = torch.where(RI_min != RI_max, 1.0, 0.0).float()

        l2 = self.criterion(RIlist[1], self.module.mask.target_data.float())
        pvb = self.criterion(RIlist[1], RIlist[0]) + self.criterion(RIlist[1], RIlist[2])
        loss = l2 * self.weight_l2 + pvb * self.weight_pvb


        l2_val = (RI_norm - self.module.mask.target_data).abs().sum()
        pvb_val = (RI_norm - RI_min).abs().sum() + (RI_norm - RI_max).abs().sum()
        other_pvb_val = (RI_max - RI_min).abs().sum()

        return loss

    def configure_optimizer(self):
        optimizer = self.optimizer_cfg(params=self.module.parameters())
        return optimizer

    # def configure_scheduler(self):
    #     scheduler = self.scheduler_cfg(optimizer=self.optimizer)
    #     return scheduler