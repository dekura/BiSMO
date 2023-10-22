"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-10-22 13:05:39
LastEditTime: 2023-10-22 17:02:16
Contact: cgjcuhk@gmail.com
Description: the defination of Source optimization problem.
"""


import torch
import torch.nn as nn

from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem




class MO(ImplicitProblem):
    def __init__(self,
                config: Config,
                module: nn.Module,
                optimizer_cfg: torch.optim.Optimizer,
                train_data_loader=None,
                # scheduler_cfg: torch.optim.lr_scheduler,
                name: str = "MO",
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
        self.module = module
        self.train_data_loader = train_data_loader
        self.sigmoid_source = nn.Sigmoid()
        self.criterion = nn.MSELoss()
        self.weight_l2 = weight_l2
        self.weight_pvb = weight_pvb
        self.optimizer_cfg = optimizer_cfg
        # self.scheduler_cfg = scheduler_cfg

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