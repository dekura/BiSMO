"""
Description: SO with Abbe's Approach.
In SO, mask.data / fdata is un-changing,
while s.data / self.source_value is changing.
"""

from pathlib import Path
from typing import Any, Optional

import aim
import cv2
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


class SOLitModule(LightningModule):
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
        dose_list: list = [0.98, 1.00, 1.02],
        source_acti: str = "sigmoid",
        source_type: str = "annular",
        source_sigmoid_steepness: float = 8,
        source_sigmoid_tr: float = 0.0,
        lens_n_liquid: float = 1.44,
        lens_reduction: float = 0.25,
        resist_intensity: float = 0.225,
        low_light_thres: float = 1e-3,
        visual_in_val: bool = True,
        resist_sigmoid_steepness: float = 30,
        weight_l2: float = 1000.00,
        weight_pvb: float = 3000.00,
        save_img_folder: str = "./data/soed",
    ) -> None:
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

        # dose list
        self.dose_list = dose_list

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # activation
        self.sigmoid_source = nn.Sigmoid()

        # init
        self.init_source_params()

    def sigmoid_resist(self, aerial) -> torch.Tensor:
        return torch.sigmoid(
            self.hparams.resist_sigmoid_steepness * (aerial - self.hparams.resist_intensity)
        )

    def init_freq_domain_on_device(self) -> None:
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
        self.s.data = self.s.data.to(device)

        # load mask data to device
        self.mask.data = self.mask.data.to(torch.float32).to(device)
        self.mask.fdata = self.mask.fdata.to(device)

        # load target data to device
        if hasattr(self.mask, "target_data"):
            self.mask.target_data = self.mask.target_data.to(torch.float32).to(device)
        else:
            self.mask.target_data = self.mask.data.detach().clone()

    def init_source_params(self) -> None:
        # [-1, 1]
        self.source_params = nn.Parameter(self.s.data.float())

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

    def update_source_value(self) -> None:
        if self.hparams.source_acti == "cosine":
            self.source_value = (1 + torch.cos(self.source_params)) / 2
        elif self.hparams.source_acti == "sigmoid":
            self.source_value = self.sigmoid_source(
                self.hparams.source_sigmoid_steepness
                * (self.source_params - self.hparams.source_sigmoid_tr)
            )
        else:
            self.source_value = (1 + torch.cos(self.source_params)) / 2

    def cal_pupil(self, FX, FY) -> torch.Tensor:
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

    def forward(self) -> tuple[list, list]:
        self.update_source_value()
        # get_valid_source
        self.source_data = torch.reshape(self.source_value, (-1, 1))
        high_light_mask = self.source_data.ge(self.hparams.low_light_thres)

        self.source_data = torch.masked_select(self.source_data, high_light_mask)
        self.simple_source_fx1d = torch.masked_select(self.source_fx1d, high_light_mask)
        self.simple_source_fy1d = torch.masked_select(self.source_fy1d, high_light_mask)
        self.simple_source_fxy2 = self.simple_source_fx1d.pow(2) + self.simple_source_fy1d.pow(2)
        self.source_weight = torch.sum(self.source_data)
        norm_pupil_fdata = self.cal_pupil(self.simple_source_fx1d, self.simple_source_fy1d)

        # get_norm_intensity
        norm_tempHAber = self.norm_spectrum_calc * norm_pupil_fdata
        norm_ExyzFrequency = norm_tempHAber.view(-1, 1)
        norm_Exyz = torch.fft.fftshift(torch.fft.fft(norm_ExyzFrequency))
        norm_IntensityCon = torch.abs(norm_Exyz * torch.conj(norm_Exyz))
        norm_total_intensity = torch.matmul(self.source_data.view(-1, 1).T, norm_IntensityCon)
        norm_IntensityTemp = self.hparams.lens_n_liquid * (self.dfmdg**2) * norm_total_intensity
        norm_Intensity = norm_IntensityTemp / self.source_weight
        self.norm_Intensity = norm_Intensity.detach()

        # obtain intensity
        self.intensity2D_list = []
        self.RI_list = []

        # 1. calculate pupil_fdata
        self.mask_fvalue_min = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(self.mask.data.float() * self.dose_list[0]))
        )
        self.mask_fvalue_norm = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(self.mask.data.float() * self.dose_list[1]))
        )
        self.mask_fvalue_max = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(self.mask.data.float() * self.dose_list[2]))
        )
        mask_fvalue = [self.mask_fvalue_min, self.mask_fvalue_norm, self.mask_fvalue_max]
        for fvalue in mask_fvalue:
            intensity2D = torch.zeros(
                self.mask.target_data.shape, dtype=torch.float32, device=self.device
            )
            for i in range(self.source_data.shape[0]):
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
                    torch.masked_select(self.mask_fm, valid_source_mask)
                    + self.simple_source_fx1d[i]
                )
                g_calc = (
                    torch.masked_select(self.mask_gm, valid_source_mask)
                    + self.simple_source_fy1d[i]
                )

                pupil_fdata = self.cal_pupil(f_calc, g_calc)

                # 2. calculate mask
                valid_mask_fdata = torch.masked_select(
                    fvalue[self.y1 : self.y2, self.x1 : self.x2], valid_source_mask
                )

                tempHAber = valid_mask_fdata * pupil_fdata

                # 3. calculate intensity
                ExyzFrequency = torch.zeros(rho2.shape, dtype=torch.complex64, device=self.device)
                ExyzFrequency[valid_source_mask] = tempHAber

                e_field = torch.zeros(fvalue.shape, dtype=torch.complex64, device=self.device)
                e_field[self.y1 : self.y2, self.x1 : self.x2] = ExyzFrequency

                AA = torch.fft.fftshift(torch.fft.ifft2(e_field))
                AA = torch.abs(AA * torch.conj(AA))
                AA = self.source_data[i] * AA
                intensity2D += AA
            normed_intensity2D = intensity2D / self.source_weight / self.norm_Intensity
            self.intensity2D_list.append(normed_intensity2D)
            self.RI_list.append(self.sigmoid_resist(normed_intensity2D))

        return self.intensity2D_list, self.RI_list

    def on_fit_start(self) -> None:
        # we need to move the freq to device
        self.init_freq_domain_on_device()

    def model_step(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        AIlist, RIlist = self.forward()
        # binary RI, torch.where creates a tensor with require_grad = False
        RI_min = torch.where(RIlist[0] > 0.5, 1.0, 0.0).float()
        RI_norm = torch.where(RIlist[1] > 0.5, 1.0, 0.0).float()
        RI_max = torch.where(RIlist[2] > 0.5, 1.0, 0.0).float()

        RI_pvb = torch.where(RI_min != RI_max, 1.0, 0.0).float()

        l2 = self.criterion(RIlist[1], self.mask.target_data.float())
        pvb = self.criterion(RIlist[1], RIlist[0]) + self.criterion(RIlist[1], RIlist[2])
        loss = l2 * self.hparams.weight_l2 + pvb * self.hparams.weight_pvb

        # l2 in 1e-3, pvb in 1e-5
        l2_val = (RI_norm - self.mask.target_data).abs().sum()
        pvb_val = (RI_norm - RI_min).abs().sum() + (RI_norm - RI_max).abs().sum()
        other_pvb_val = (RI_max - RI_min).abs().sum()

        # for training, I use sig(AI) to calculate loss.
        # for testing and validation, I use real RI to get pvb and l2.
        return l2_val, pvb_val, other_pvb_val, loss, AIlist[1], RI_norm, RI_pvb

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        l2, pvb, other_pvb, loss, _, _, _ = self.model_step()

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        # return loss or backpropagation will fail

        l2_error = l2.detach().clone()
        pvb_error = pvb.detach().clone()
        other_pvb_error = other_pvb.detach().clone()
        self.log("train/l2", l2_error, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/pvb", pvb_error, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(
            "train/other_pvb",
            other_pvb_error,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        # return {"loss": loss}
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        l2, pvb, other_pvb, loss, AI, RI, RI_pvb = self.model_step()

        self.val_loss(loss)
        self.log(
            "val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            "simple_s_num",
            torch.tensor(self.source_data.shape[0]).float(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        # for aim logger
        l2_error = l2.detach().clone()
        pvb_error = pvb.detach().clone()
        other_pvb_error = other_pvb.detach().clone()
        binary_AI = RI.detach().clone()
        vis_pvb = RI_pvb.detach().clone()

        self.log("val/l2", l2_error, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val/pvb", pvb_error, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(
            "val/other_pvb",
            other_pvb_error,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        if self.hparams.visual_in_val:
            if self.global_rank == 0:
                if isinstance(self.logger, AimLogger):
                    transform = T.ToPILImage()
                    aim_images = [
                        aim.Image(transform(i))
                        for i in [
                            self.s.data.clone().detach(),
                            self.mask.data,
                            self.mask.target_data,
                            binary_AI.to(torch.float32),
                            # RI,
                            vis_pvb,
                            AI.clone().detach(),
                        ]
                    ]
                    self.logger.experiment.track(
                        value=aim_images,
                        name=f"AI and RI in epoch {self.current_epoch}",
                        step=self.global_step,
                        context={"epoch": self.current_epoch},
                    )

    def test_step(self, batch: Any, batch_idx: int) -> None:
        _, _, _, _, AI, RI, RI_pvb = self.model_step()
        save_img_folder = Path(self.hparams.save_img_folder) / self.mask.dataset_name
        AI_folder = save_img_folder / "AI"
        AI_folder.mkdir(parents=True, exist_ok=True)

        RI_folder = save_img_folder / "RI"
        RI_folder.mkdir(parents=True, exist_ok=True)

        # mask_folder = save_img_folder / f"mask"
        # mask_folder.mkdir(parents=True, exist_ok=True)

        pvb_folder = save_img_folder / "pvb"
        pvb_folder.mkdir(parents=True, exist_ok=True)

        source_folder = save_img_folder / "source"
        source_folder.mkdir(parents=True, exist_ok=True)

        RI_soed = RI.detach().clone()
        AI_soed = AI.detach().clone()
        RI_pvb_soed = RI_pvb.detach().clone()
        sourece = torch.where(self.source_params > 0.0, 1, 0)

        AI_soed_path = AI_folder / self.mask.mask_name
        RI_soed_path = RI_folder / self.mask.mask_name
        RI_pvb_soed_path = pvb_folder / self.mask.mask_name
        source_soed_path = source_folder / self.mask.mask_name
        # mask_path = mask_folder / self.mask.mask_name

        U.save_image(AI_soed.to(torch.float32), AI_soed_path)
        U.save_image(RI_soed.to(torch.float32), RI_soed_path)
        U.save_image(RI_pvb_soed.to(torch.float32), RI_pvb_soed_path)
        U.save_image(sourece.to(torch.float32), source_soed_path)
        # U.save_image(self.mask.data, mask_path)

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
    _ = SOLitModule(None, None, None)
