from typing import Any

import aim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from aim.sdk.adapters.pytorch_lightning import AimLogger
from lightning import LightningModule
from torchmetrics import MeanMetric

from src.models.litho.gds_mask import Mask
from src.models.litho.source import Source
from src.models.litho.utils import torch_arr_bound

SOURCE_RELAX_SIGMOID_STEEPNESS = 8
PHOTORISIST_SIGMOID_STEEPNESS = 50
TARGET_INTENSITY = 0.31
LOW_LIGHT_THRES = 1e-3


class SMOLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        source: Source,
        mask: Mask,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        source_acti: str = "cosine",
        visual_in_val: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["source", "mask"])

        # loss function
        self.criterion = nn.MSELoss()

        # source
        self.s = source
        self.s.update()
        # mask
        self.mask = mask
        self.mask.openGDS()
        self.mask.maskfft()
        self.mask.data = self.mask.data.to(torch.float32)
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # activation
        self.sigmoid_source = nn.Sigmoid()

        # init source params
        # hyper-parameters
        self.init_freq_domain()
        self.init_source_params()

    def sigmoid_resist(
        self, aerial, steepness=PHOTORISIST_SIGMOID_STEEPNESS, target_intensity=TARGET_INTENSITY
    ):
        return torch.sigmoid(steepness * (aerial - target_intensity))

    def init_freq_domain(self):
        # hyper-parameters
        self.gnum = self.s.gnum
        self.fnum = self.s.fnum

        self.x_gridnum = self.mask.x_gridnum
        self.y_gridnum = self.mask.y_gridnum
        self.x1 = int(self.x_gridnum // 2 - self.fnum)
        self.x2 = int(self.x_gridnum // 2 + self.fnum + 1)
        self.y1 = int(self.y_gridnum // 2 - self.gnum)
        self.y2 = int(self.y_gridnum // 2 + self.gnum + 1)

        normalized_period = self.x_gridnum / (self.s.wavelength / self.s.na)
        mask_fm = (torch.arange(self.x1, self.x2) - self.x_gridnum // 2) / normalized_period

        mask_gm = (torch.arange(self.y1, self.y2) - self.y_gridnum // 2) / normalized_period

        self.mask_fm, self.mask_gm = torch.meshgrid(mask_fm, mask_gm, indexing="xy")
        self.mask_fg2m = self.mask_fm.pow(2) + self.mask_gm.pow(2)

    def init_source_params(self):
        # [-1, 1]
        self.source_params = nn.Parameter(torch.zeros(self.s.data.shape))

        # for sigmoid
        if self.hparams.source_acti == "sigmoid":
            self.source_params.data[torch.where(self.s.data > 0.5)] = 2 - 0.2
            self.source_params.data.sub_(0.9)
        elif self.hparams.source_acti == "cosine":
            self.source_params.data[torch.where(self.s.data > 0.5)] = 0.1
            self.source_params.data[torch.where(self.s.data <= 0.5)] = torch.pi - 0.1
        else:
            # default cosine
            self.source_params.data[torch.where(self.s.data > 0.5)] = 0.1
            self.source_params.data[torch.where(self.s.data <= 0.5)] = torch.pi - 0.1

    def update_source_value(self):
        if self.hparams.source_acti == "cosine":
            self.source_value = (1 + torch.cos(self.source_params)) / 2
        elif self.hparams.source_acti == "sigmoid":
            self.source_value = self.sigmoid_source(
                SOURCE_RELAX_SIGMOID_STEEPNESS * self.source_params
            )
        else:
            self.source_value = (1 + torch.cos(self.source_params)) / 2

    def get_valid_source(self):
        fx = self.s.fx
        fy = self.s.fy

        size_x, size_y = fx.shape[0], fx.shape[1]
        fx1d = torch.reshape(fx, (size_x * size_y, 1))

        size_x, size_y = fy.shape[0], fy.shape[1]
        fy1d = torch.reshape(fy, (size_x * size_y, 1))

        self.source_fx1d = fx1d
        self.source_fy1d = fy1d

        size_x, size_y = (
            self.s.data.shape[0],
            self.s.data.shape[1],
        )
        self.simple_source_value = torch.reshape(self.source_value, (size_x * size_y, 1))
        high_light_mask = self.simple_source_value.ge(LOW_LIGHT_THRES)

        self.simple_source_value = torch.masked_select(self.simple_source_value, high_light_mask)
        self.simple_source_fx1d = torch.masked_select(self.source_fx1d, high_light_mask)
        self.simple_source_fy1d = torch.masked_select(self.source_fy1d, high_light_mask)
        self.simple_source_fxy2 = self.simple_source_fx1d.pow(2) + self.simple_source_fy1d.pow(2)
        self.source_weight = torch.sum(self.simple_source_value)

    def cal_pupil(self, FX, FY):
        R = torch.sqrt(FX**2 + FY**2)
        H = R.clone().detach()
        H = torch.where(H > 1.0, 0.0, 1.0)
        R[R > 1.0] = 0.0
        W = torch.zeros(R.shape, dtype=torch.complex64)
        self.pupil_fdata = H * torch.exp(-1j * 2 * (torch.pi) * W)

    def forward(self):
        self.update_source_value()
        self.get_valid_source()
        intensity2D = torch.zeros(self.mask.data.shape, dtype=torch.float32)

        print(f"total source number: {self.simple_source_value.shape[0]}")
        for i in range(self.simple_source_value.shape[0]):
            rho2 = (
                self.mask_fg2m
                + 2
                * (
                    self.simple_source_fx1d[i] * self.mask_fm
                    + self.simple_source_fy1d[i] * self.mask_gm
                )
                + self.simple_source_fxy2[i]
            )

            valid_source_mask = rho2.le(1)
            f_calc = torch.masked_select(self.mask_fm, valid_source_mask)
            g_calc = torch.masked_select(self.mask_gm, valid_source_mask)
            self.cal_pupil(f_calc, g_calc)

            valid_mask_fdata = torch.masked_select(
                self.mask.fdata[self.y1 : self.y2, self.x1 : self.x2], valid_source_mask
            )
            tempHAber = valid_mask_fdata * self.pupil_fdata

            e_field = torch.zeros(self.mask.fdata.shape, dtype=torch.complex64)

            ExyzFrequency = torch.zeros(rho2.shape, dtype=torch.complex64)
            ExyzFrequency[valid_source_mask] = tempHAber

            e_field[self.y1 : self.y2, self.x1 : self.x2] = ExyzFrequency

            AA = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(e_field)))
            AA = torch.abs(AA * torch.conj(AA))
            AA = self.simple_source_value[i] * AA
            intensity2D += AA
        self.intensity2D = intensity2D / self.source_weight
        self.RI = self.sigmoid_resist(self.intensity2D)

        return self.intensity2D, self.RI

    def model_step(self):
        AI, RI = self.forward()
        loss = self.criterion(RI, self.mask.data)
        return loss, AI, RI

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, _ = self.model_step()

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        # return loss or backpropagation will fail
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, AI, RI = self.model_step()
        self.val_loss(loss)
        self.log(
            "val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        # for aim logger
        if self.hparams.visual_in_val:
            if self.global_rank == 0:
                if isinstance(self.logger, AimLogger):
                    transform = T.ToPILImage()
                    aim_images = [
                        aim.Image(transform(i))
                        for i in [AI, RI, self.source_value.clone().detach()]
                    ]
                    self.logger.experiment.track(
                        value=aim_images, name=f"AI and RI in epoch {self.current_epoch}"
                    )

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SMOLitModule(None, None, None)
