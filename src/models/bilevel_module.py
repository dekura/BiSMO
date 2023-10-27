import sys

sys.path.append(".")

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

from src.models.litho.aerial import AerialList
from src.models.litho.img_mask import Mask
from src.models.litho.lens import LensList
from src.models.litho.source import Source
from src.models.litho.tcc import TCCList
from src.models.litho.utils import torch_arr_bound


class SMOLitModule(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)
    """

    def __init__(
        self,
        source: Source,
        mask: Mask,
        # lens: LensList,
        dose_list: list = [
            0.98,
            1.00,
            1.02,
        ],
        # dose_coff: list = [1, 1, 1,],
        resist_tRef: int = 0.06,
        weight_pvb: float = 1.0,
        weight_l2: float = 1.0,
        # tcc: TCCList,
        # aerial: AerialList,
        # optimizer: torch.optim.Optimizer,
        # scheduler: torch.optim.lr_scheduler,
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

        # source
        self.source = source
        self.source.update()
        self.source.ifft()

        # lens / pupil
        # lens_list.focusList = [-50, 0, 50]
        # lens_list.focusCoef = [0.5, 1, 0.5]
        # self.lens = lens
        # self.lens.focusList = [0.0]
        # self.lens.focusCoef = [1.0]
        # self.lens.calculate()

        # tcc
        # self.tcc = TCCList(self.source, self.lens)
        # self.tcc.calculate()

        # mask
        self.mask = mask
        self.mask.open_layout()
        self.mask.maskfft()

        # AI and RI, new
        # self.aerial = AerialList(self.mask, self.tcc, Any)
        # a.image.doseList = [0.98, 1, 1.02]
        # a.image.doseCoef = [1, 1, 1]
        # self.aerial.image.doseList = [0.98, 1, 1.02,]
        # self.aerial.image.doseCoef = [1, 1, 1,]
        # a.litho()
        # self.aerial.litho()

        # doselist
        self.resist_tRef = resist_tRef
        self.dose_list = dose_list
        # self.dose_coff = dose_coff
        self.weight_pvb = weight_pvb
        self.weight_l2 = weight_l2

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        # activation
        self.sigmoid_source = nn.Sigmoid()

        # loss function
        # self.criterion =
        # length = len(self.aerial.image.focusList)
        # lengthD = len(self.aerial.image.doseList)

        # for ii in range(length):
        # pvb, self.aerial.image.RIList is a torch
        # pvb = torch.sum((self.aerial.image.RIList[ii][0] - self.aerial.image.RIList[ii][1]) ** 2) + torch.sum((self.aerial.image.RIList[ii][2] - self.aerial.image.RIList[ii][1]) ** 2)
        # mse
        # l2 = torch.sum((self.aerial.image.RIList[ii][1] - self.mask.data).abs())
        # print('pvb: ', pvb.sum(), 'l2', l2)

        # self.criterion = nn.MSELoss()
        # self.criterion = nn.SmoothL1Loss + pvb + l2
        self.criterion = nn.SmoothL1Loss()

        self.init_source_params()

    # RI = sigmoid(resist_sigmoid_steepness * (AI - target_intensity))
    def sigmoid_resist(self, aerial):
        return torch.sigmoid(
            self.hparams.resist_sigmoid_steepness * (aerial - self.hparams.target_intensity)
        )

    def init_freq_domain_on_device(self):
        """
        na: float = 1.35,
        wavelength: float = 193.0,
        maskxpitch: float = 1280.0,
        maskypitch: float = 1280.0,
        sigma_out: float = 0.95,
        sigma_in: float = 0.63,
        smooth_deta: float = 0.03,
        source_type: str = "annular",
        shiftAngle: float = math.pi / 4,
        openAngle: float = math.pi / 16,

        193 / (1280 * 1.35) = 0.112

        fnum / gnum = 2 / 0.112 = 17.906 -> 18

        self.detaf = self.wavelength / (self.maskxpitch * self.na)
        self.detag = self.wavelength / (self.maskypitch * self.na)
        self.fnum = int(torch.ceil(torch.tensor(2 / self.detaf, dtype=torch.float64)))
        self.gnum = int(torch.ceil(torch.tensor(2 / self.detag, dtype=torch.float64)))

        - 18 * 0.112

        fx = torch.linspace(-self.fnum * self.detaf, self.fnum * self.detaf, 2 * self.fnum + 1)
        fy = torch.linspace(-self.gnum * self.detag, self.gnum * self.detag, 2 * self.gnum + 1)

        form a meshgrid

        FX, FY = torch.meshgrid(fx, fy, indexing="xy")
        """

        device = self.device
        # hyper-parameters, int: 18
        self.gnum = self.source.gnum
        self.fnum = self.source.fnum

        self.x_gridnum = self.mask.x_gridnum
        self.y_gridnum = self.mask.y_gridnum
        self.x1 = int(self.x_gridnum // 2 - self.fnum)
        self.x2 = int(self.x_gridnum // 2 + self.fnum + 1)
        self.y1 = int(self.y_gridnum // 2 - self.gnum)
        self.y2 = int(self.y_gridnum // 2 + self.gnum + 1)

        normalized_period_x = self.x_gridnum / (self.source.wavelength / self.source.na)
        normalized_period_y = self.y_gridnum / (self.source.wavelength / self.source.na)
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
        self.s_fx = self.source.fx.to(device)
        self.s_fy = self.source.fy.to(device)
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
        # [-1, 1], learnable
        self.source_params = nn.Parameter(torch.zeros(self.source.data.shape))

        # for sigmoid
        if self.hparams.source_acti == "sigmoid":
            self.source_params.data[torch.where(self.source.data > 0.5)] = 2 - 0.02
            self.source_params.data.sub_(0.99)
        elif self.hparams.source_acti == "cosine":
            self.source_params.data[torch.where(self.source.data > 0.5)] = 0.1
            self.source_params.data[torch.where(self.source.data <= 0.5)] = torch.pi - 0.1
        else:
            # default cosine
            self.source_params.data[torch.where(self.source.data > 0.5)] = 0.1
            self.source_params.data[torch.where(self.source.data <= 0.5)] = torch.pi - 0.1

    # activate source
    def update_source_value(self):
        if self.hparams.source_acti == "cosine":
            self.source_value = (1 + torch.cos(self.source_params)) / 2
        elif self.hparams.source_acti == "sigmoid":
            self.source_value = self.sigmoid_source(
                self.hparams.source_sigmoid_steepness * self.source_params
            )
        else:
            self.source_value = (1 + torch.cos(self.source_params)) / 2

    # change source to annular or other types
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
        NA = self.source.na
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
        # real RI,
        # instead self.hparams.target_intensity to
        # self.resist_tRef
        # self.RI = (self.RI >= self.hparams.target_intensity).to(torch.float64)
        # threshold, substrate
        self.RIlist = []
        for ii in range(self.dose_list):
            resist_t = ii * self.hparams.target_intensity
            self.RIlist.append((self.RI >= resist_t).to(torch.float64))
        return self.intensity2D, self.RI, self.RIlist

    def model_step(self):
        AI, RI, RIlist = self.forward()
        # loss, pvb, l2
        pvb = (torch.sum((RIlist[0] - RI) ** 2) + torch.sum((RIlist[2] - RI) ** 2)).requires_grad_(
            True
        )
        l2 = self.criterion(RI, self.mask.target_data)
        # origin loss
        # loss = self.criterion(RI, self.mask.target_data)
        # true loss
        loss = (pvb * self.weight_pvb) + (l2 * self.weight_l2)
        return loss, l2, pvb, AI, RI

    def training_step(self, batch: Any, batch_idx: int):
        loss, l2, pvb, _, _ = self.model_step()

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        # return loss or backpropagation will fail
        # binary_AI = torch.where(AI.detach() > self.hparams.target_intensity, 1, 0)
        # l2_error = (self.mask.target_data - binary_AI).abs().sum()
        # l2_error = (self.mask.target_data - RI).abs().sum()
        self.log("train/l2", l2, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.log("train/pvb", pvb, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, _, _, AI, RI = self.model_step()

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
        _, _, _, AI, RI = self.model_step()
        source_type = self.source.type
        save_img_folder = Path(self.hparams.save_img_folder) / self.mask.dataset_name
        RI_folder = save_img_folder / f"{source_type}_RI"
        RI_folder.mkdir(parents=True, exist_ok=True)
        SO_folder = save_img_folder / f"{source_type}_SO"
        SO_folder.mkdir(parents=True, exist_ok=True)
        mask_folder = save_img_folder / f"{source_type}_mask"
        mask_folder.mkdir(parents=True, exist_ok=True)

        # save images
        # binary_AI = torch.where(AI.detach() > self.hparams.target_intensity, 1, 0)
        # true binary RI
        binary_AI = RI
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
    # _ = SMOLitModule(None, None, None)
    mask_path = "data/soed/ibm_opc_test/mask/t1_0_mask.png"
    m = Mask(layout_path=mask_path, target_path=mask_path)
    s = Source(
        source_type="annular",
        maskxpitch=1280,
        maskypitch=1280,
        sigma_in=0.63,
        sigma_out=0.95,
    )

    o = LensList(
        nLiquid=1.44,
        wavelength=193.0,
        defocus=0.0,
        maskxpitch=1280,
        maskypitch=1280,
        na=1.35,
    )
    _ = SMOLitModule(source=s, mask=m, lens=o)
