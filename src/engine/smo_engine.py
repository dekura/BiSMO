"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-10-23 15:16:50
LastEditTime: 2023-10-23 15:39:13
Contact: cgjcuhk@gmail.com
Description: the SMO engine for visulization
"""

import aim
import torch
import torchvision.transforms as T

from betty.engine import Engine
from betty.configs import Config, EngineConfig


class SMOEngine(Engine):
    @torch.no_grad()
    def validation(self):
        AIlist, RIlist = self.MO.forward()
        # binary RI, torch.where creates a tensor with require_grad = False
        # RI_min = torch.where(RIlist[0] > 0.5, 1.0, 0.0).float()
        RI_norm = torch.where(RIlist[1] > 0.5, 1.0, 0.0).float()
        # RI_max = torch.where(RIlist[2] > 0.5, 1.0, 0.0).float()
        mask_moed = torch.where(self.MO.module.mask_value > 0.5, 1.0, 0.0).float()
        self.logger.info("[Validation] save images to aim logger")
        transform = T.ToPILImage()
        aim_images = [
            aim.Image(transform(i))
            for i in [
                self.MO.source_value.clone().detach(),
                mask_moed.detach().clone(),
                self.MO.module.mask_value.detach().clone(),
                RI_norm.detach().clone(),
                self.MO.module.mask.target_data.clone().detach(),
            ]
        ]
        self.logger.experiment.track(
            value=aim_images,
            name=f"val epoch {self.global_step}",
            step=self.global_step,
            context={"val epoch": self.global_step},
        )


