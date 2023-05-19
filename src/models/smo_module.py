from pathlib import Path
from typing import Any, Optional

import aim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as U
from aim.sdk.adapters.pytorch_lightning import AimLogger
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MeanMetric

from src.models.litho.img_mask import Mask
from src.models.litho.source import Source
from src.models.litho.utils import torch_arr_bound


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
        source_acti: str = "sigmoid",
        source_sigmoid_steepness: float = 8,
        lens_n_liquid: float = 1.44,
        lens_reduction: float = 0.25,
        target_intensity: float = 0.225,
        low_light_thres: float = 1e-3,
        visual_in_val: bool = True,
        resist_sigmoid_steepness: float = 30,
        save_img_folder: str = "./data/soed",
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
        self.mask.open_layout()
        self.mask.maskfft()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        # activation
        self.sigmoid_source = nn.Sigmoid()

        self.init_source_params()

    def sigmoid_resist(self, aerial):
        return torch.sigmoid(
            self.hparams.resist_sigmoid_steepness * (aerial - self.hparams.target_intensity)
        )

    def init_freq_domain_on_device(self):
        device = self.device
        # hyper-parameters
        self.gnum = self.s.gnum
        self.fnum = self.s.fnum

        self.x_gridnum = self.mask.x_gridnum
        self.y_gridnum = self.mask.y_gridnum
        self.x1 = int(self.x_gridnum // 2 - self.fnum)
        self.x2 = int(self.x_gridnum // 2 + self.fnum + 1)
        self.y1 = int(self.y_gridnum // 2 - self.gnum)
        self.y2 = int(self.y_gridnum // 2 + self.gnum + 1)

        normalized_period_x = self.x_gridnum / (self.s.wavelength / self.s.na)
        normalized_period_y = self.y_gridnum / (self.s.wavelength / self.s.na)
        mask_fm = (torch.arange(self.x1, self.x2) - self.x_gridnum // 2) / normalized_period_x
        mask_fm = mask_fm.to(device)

        mask_gm = (torch.arange(self.y1, self.y2) - self.y_gridnum // 2) / normalized_period_y
        mask_gm = mask_gm.to(device)

        self.mask_fm, self.mask_gm = torch.meshgrid(mask_fm, mask_gm, indexing="xy")
        self.mask_fg2m = self.mask_fm.pow(2) + self.mask_gm.pow(2)

        # for intensity norm
        self.norm_spectrum_calc = normalized_period_x * normalized_period_y
        self.dfmdg = 1 / self.norm_spectrum_calc

        # source part
        self.s_fx = self.s.fx.to(device)
        self.s_fy = self.s.fy.to(device)
        self.source_fx1d = torch.reshape(self.s_fx, (-1, 1))
        self.source_fy1d = torch.reshape(self.s_fy, (-1, 1))

        # load mask data to device
        self.mask.data = self.mask.data.to(torch.float32).to(device)
        self.mask.fdata = self.mask.fdata.to(device)

        # load target data to device
        if hasattr(self.mask, "target_data"):
            self.mask.target_data = self.mask.target_data.to(torch.float32).to(device)
        else:
            self.mask.target_data = self.mask.data.detach().clone()

    def init_source_params(self):
        # [-1, 1]
        self.source_params = nn.Parameter(torch.zeros(self.s.data.shape))

        # for sigmoid
        if self.hparams.source_acti == "sigmoid":
            self.source_params.data[torch.where(self.s.data > 0.5)] = 2 - 0.02
            self.source_params.data.sub_(0.99)
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
                self.hparams.source_sigmoid_steepness * self.source_params
            )
        else:
            self.source_value = (1 + torch.cos(self.source_params)) / 2

    def get_valid_source(self):
        self.simple_source_value = torch.reshape(self.source_value, (-1, 1))
        high_light_mask = self.simple_source_value.ge(self.hparams.low_light_thres)

        self.simple_source_value = torch.masked_select(self.simple_source_value, high_light_mask)
        self.simple_source_fx1d = torch.masked_select(self.source_fx1d, high_light_mask)
        self.simple_source_fy1d = torch.masked_select(self.source_fy1d, high_light_mask)
        self.simple_source_fxy2 = self.simple_source_fx1d.pow(2) + self.simple_source_fy1d.pow(2)
        self.source_weight = torch.sum(self.simple_source_value)

    def cal_pupil(self, FX, FY):
        R = torch.sqrt(FX**2 + FY**2)  # rho
        fgSquare = torch.square(R)
        NA = self.s.na
        n_liquid = self.hparams.lens_n_liquid
        M = self.hparams.lens_reduction
        obliquityFactor = torch.sqrt(
            torch.sqrt(
                (1 - (M**2 * NA**2) * fgSquare) / (1 - ((NA / n_liquid) ** 2) * fgSquare)
            )
        )
        # no aberrations
        return obliquityFactor * (1 + 0j)

    def get_norm_intensity(self):
        norm_pupil_fdata = self.cal_pupil(self.simple_source_fx1d, self.simple_source_fy1d)
        norm_tempHAber = self.norm_spectrum_calc * norm_pupil_fdata
        norm_ExyzFrequency = norm_tempHAber.view(-1, 1)
        norm_Exyz = torch.fft.fftshift(torch.fft.fft(norm_ExyzFrequency))
        norm_IntensityCon = torch.abs(norm_Exyz * torch.conj(norm_Exyz))
        norm_total_intensity = torch.matmul(
            self.simple_source_value.view(-1, 1).T, norm_IntensityCon
        )
        norm_IntensityTemp = self.hparams.lens_n_liquid * (self.dfmdg**2) * norm_total_intensity
        norm_Intensity = norm_IntensityTemp / self.source_weight
        self.norm_Intensity = norm_Intensity.detach()

    def on_fit_start(self):
        # we need to move the freq to device
        self.init_freq_domain_on_device()

    def forward(self):
        self.update_source_value()
        self.get_valid_source()
        self.get_norm_intensity()
        intensity2D = torch.zeros(self.mask.data.shape, dtype=torch.float32, device=self.device)

        # print(self.simple_source_value.shape[0])
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
            f_calc = (
                torch.masked_select(self.mask_fm, valid_source_mask) + self.simple_source_fx1d[i]
            )
            g_calc = (
                torch.masked_select(self.mask_gm, valid_source_mask) + self.simple_source_fy1d[i]
            )

            pupil_fdata = self.cal_pupil(f_calc, g_calc)

            valid_mask_fdata = torch.masked_select(
                self.mask.fdata[self.y1 : self.y2, self.x1 : self.x2], valid_source_mask
            )

            tempHAber = valid_mask_fdata * pupil_fdata

            ExyzFrequency = torch.zeros(rho2.shape, dtype=torch.complex64, device=self.device)
            ExyzFrequency[valid_source_mask] = tempHAber

            e_field = torch.zeros(self.mask.fdata.shape, dtype=torch.complex64, device=self.device)
            e_field[self.y1 : self.y2, self.x1 : self.x2] = ExyzFrequency

            AA = torch.fft.fftshift(torch.fft.ifft2(e_field))
            AA = torch.abs(AA * torch.conj(AA))
            AA = self.simple_source_value[i] * AA
            intensity2D += AA
        self.intensity2D = intensity2D / self.source_weight
        self.intensity2D = self.intensity2D / self.norm_Intensity
        self.RI = self.sigmoid_resist(self.intensity2D)
        return self.intensity2D, self.RI

    def model_step(self):
        AI, RI = self.forward()
        loss = self.criterion(RI, self.mask.target_data)
        return loss, AI, RI

    def training_step(self, batch: Any, batch_idx: int):
        loss, AI, _ = self.model_step()

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        # return loss or backpropagation will fail
        binary_AI = torch.where(AI.detach() > self.hparams.target_intensity, 1, 0)
        l2_error = (self.mask.target_data - binary_AI).abs().sum()
        self.log("train/l2", l2_error, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, AI, RI = self.model_step()

        self.val_loss(loss)
        self.log(
            "val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            "simple_s_num",
            torch.tensor(self.simple_source_value.shape[0]),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        # for aim logger

        binary_AI = torch.where(AI.detach() > self.hparams.target_intensity, 1, 0)
        l2_error = (self.mask.target_data - binary_AI).abs().sum()
        self.log("val/l2", l2_error, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if self.hparams.visual_in_val:
            if self.global_rank == 0:
                if isinstance(self.logger, AimLogger):
                    transform = T.ToPILImage()
                    aim_images = [
                        aim.Image(transform(i))
                        for i in [
                            self.source_value.clone().detach(),
                            self.mask.data,
                            self.mask.target_data,
                            binary_AI.to(torch.float32),
                            RI,
                            AI,
                        ]
                    ]
                    self.logger.experiment.track(
                        value=aim_images,
                        name=f"AI and RI in epoch {self.current_epoch}",
                        step=self.global_step,
                        context={"epoch": self.current_epoch},
                    )

    def test_step(self, batch: Any, batch_idx: int):
        _, AI, RI = self.model_step()
        source_type = self.s.type
        save_img_folder = Path(self.hparams.save_img_folder) / self.mask.dataset_name
        RI_folder = save_img_folder / f"{source_type}_RI"
        RI_folder.mkdir(parents=True, exist_ok=True)
        SO_folder = save_img_folder / f"{source_type}_SO"
        SO_folder.mkdir(parents=True, exist_ok=True)
        mask_folder = save_img_folder / f"{source_type}_mask"
        mask_folder.mkdir(parents=True, exist_ok=True)

        binary_AI = torch.where(AI.detach() > self.hparams.target_intensity, 1, 0)
        binary_source = torch.where(self.source_params > 0, 1, 0)
        binary_AI_path = RI_folder / self.mask.mask_name
        binary_source_path = SO_folder / self.mask.mask_name
        binary_mask_path = mask_folder / self.mask.mask_name
        U.save_image(binary_AI.to(torch.float32), binary_AI_path)
        U.save_image(binary_source.to(torch.float32), binary_source_path)
        U.save_image(self.mask.data, binary_mask_path)

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
