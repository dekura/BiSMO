'''
Description: Random SMO with Abbe's Approach.
'''

from pathlib import Path
from typing import Any, Optional

import aim
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.utils as U
from aim.sdk.adapters.pytorch_lightning import AimLogger
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MeanMetric

from src.models.litho.img_mask import Mask
from src.models.litho.source import Source

from src.models.modules.so import SO_Module
from src.models.modules.mo import MO_Module


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
        so_optimizer: torch.optim.Optimizer,
        so_scheduler: torch.optim.lr_scheduler,
        mo_optimizer: torch.optim.Optimizer,
        mo_scheduler: torch.optim.lr_scheduler,
        dose_list: list = [0.98, 1.00, 1.02],
        source_acti: str = "sigmoid",
        mask_acti: str = "sigmoid",
        source_type: str = 'annular',
        mask_sigmoid_steepness: float = 9,
        mask_sigmoid_tr: float = 0.5,
        source_sigmoid_steepness: float = 8,
        lens_n_liquid: float = 1.44,
        lens_reduction: float = 0.25,
        resist_intensity: float = 0.225,
        low_light_thres: float = 1e-3,
        visual_in_val: bool = True,
        resist_sigmoid_steepness: float = 30,
        weight_l2: float = 1000.00,
        weight_pvb: float = 3000.00,
        mo_frequency: float = 50,
        so_frequency: float = 10,
        save_img_folder: str = "./data/smoed",
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["source", "mask"])
        # loss function
        self.criterion = nn.MSELoss()

        # source
        self.s = source
        self.s.update()

        # mask, init mask.data, mask.fdata, no need for init_mask_params
        self.mask = mask
        self.mask.open_layout()
        self.mask.maskfft()

        # dose list
        self.dose_list = dose_list

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # init
        self.SO = self.init_SO()
        self.MO = self.init_MO()

    def init_SO(self):
        return SO_Module(self.s, 
                         self.mask, 
                         self.hparams.source_acti, 
                         self.hparams.source_sigmoid_steepness, 
                         self.hparams.resist_sigmoid_steepness, 
                         self.hparams.resist_intensity, 
                         self.dose_list, 
                         self.hparams.lens_n_liquid, 
                         self.hparams.lens_reduction, 
                         self.hparams.low_light_thres, 
                         self.device
                         )

    def init_MO(self):
        return MO_Module(self.s, 
                         self.mask, 
                         self.hparams.mask_acti, 
                         self.hparams.mask_sigmoid_steepness, 
                         self.hparams.resist_sigmoid_steepness, 
                         self.hparams.resist_intensity, 
                         self.dose_list, 
                         self.hparams.lens_n_liquid, 
                         self.hparams.lens_reduction, 
                         self.hparams.low_light_thres, 
                         self.device
                         )

    def model_step(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        switch_flag = self.current_epoch % (self.hparams.mo_frequency + self.hparams.so_frequency)
        self.mo_flag = switch_flag < self.hparams.mo_frequency
        self.so_flag = switch_flag < (self.hparams.mo_frequency + self.hparams.mo_frequency)
        AIlist = []
        RIlist = []

        if self.mo_flag:
            self.SO.update_source_value()
            AIlist, RIlist = self.MO.forward(self.SO.source_value.detach().clone())
        elif self.so_flag:
            self.MO.update_mask_value()
            AIlist, RIlist = self.SO.forward(self.MO.mask_value.detach().clone())
        else:
            # default MO.
            self.SO.update_source_value()
            AIlist, RIlist = self.MO.forward(self.SO.source_value.detach().clone())


        RI_min = torch.where(RIlist[0] > 0.5, 1.0, 0.0).float()
        RI_norm = torch.where(RIlist[1] > 0.5, 1.0, 0.0).float()
        RI_max = torch.where(RIlist[2] > 0.5, 1.0, 0.0).float()

        RI_pvb = torch.where(RI_min != RI_max, 1.0, 0.0).float()

        l2 = self.criterion(RIlist[1], self.mask.target_data.float())
        pvb = self.criterion(RIlist[1], RIlist[0]) + self.criterion(RIlist[1], RIlist[2])
        loss = l2 * self.hparams.weight_l2 + pvb * self.hparams.weight_pvb

        l2_val = (RI_norm - self.mask.target_data).abs().sum()
        pvb_val = (RI_norm - RI_min).abs().sum() + (RI_norm - RI_max).abs().sum()
        other_pvb_val = (RI_max - RI_min).abs().sum()

        return l2_val, pvb_val, other_pvb_val, loss, AIlist[1], RI_norm, RI_pvb

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        so_opt, mo_opt = self.optimizers(use_pl_optimizer=True)
        so_opt = so_opt.optimizer
        mo_opt = mo_opt.optimizer
        so_sch, mo_sch = self.lr_schedulers()
        l2, pvb, other_pvb, loss, _, _, _ = self.model_step()

        if self.mo_flag:
            mo_opt.zero_grad()
            self.manual_backward(loss)
            mo_opt.step()
            # mo_sch.step(self.trainer.callback_metrics['train/loss'])
        elif self.so_flag:
            so_opt.zero_grad()
            self.manual_backward(loss)
            so_opt.step()
            # so_sch.step(self.trainer.callback_metrics['train/loss'])

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
        self.log("train/other_pvb", other_pvb_error, on_step=False, on_epoch=True, prog_bar=False, logger=True)
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
            torch.tensor(self.s.data.shape[0] * self.s.data.shape[1]).float(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        # for aim logger
        l2_error = l2.detach().clone()
        pvb_error = pvb.detach().clone()
        other_pvb_error = other_pvb.detach().clone()

        self.log("val/l2", l2_error, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val/pvb", pvb_error, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val/other_pvb", other_pvb_error, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        masked_smoed = torch.where(self.MO.mask_value > 0.5, 1.0, 0.0).float()

        source_smoed = torch.where(self.SO.source_value > 0.5, 1.0, 0.0).float()

        if self.hparams.visual_in_val:
            if self.global_rank == 0:
                if isinstance(self.logger, AimLogger):
                    transform = T.ToPILImage()
                    aim_images = [
                        aim.Image(transform(i))
                        for i in [
                            self.SO.source_value.detach().clone(),
                            source_smoed.detach().clone(),
                            self.mask.target_data.float().detach().clone(),
                            masked_smoed.detach().clone(),
                            RI.detach().clone(),
                            RI_pvb.detach().clone(),
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
        AI_folder = save_img_folder / f"AI"
        AI_folder.mkdir(parents=True, exist_ok=True)
        
        RI_folder = save_img_folder / f"RI"
        RI_folder.mkdir(parents=True, exist_ok=True)

        pvb_folder = save_img_folder / f"pvb"
        pvb_folder.mkdir(parents=True, exist_ok=True)

        source_folder = save_img_folder / f"source"
        source_folder.mkdir(parents=True, exist_ok=True)

        gray_source_folder = save_img_folder / f"gray_source"
        gray_source_folder.mkdir(parents=True, exist_ok=True)

        masked_folder = save_img_folder / f"masked"
        masked_folder.mkdir(parents=True, exist_ok=True)

        RI_smoed = RI.detach().clone()
        AI_smoed = AI.detach().clone()
        RI_pvb_smoed = RI_pvb.detach().clone()
        masked_smoed = torch.where(self.MO.mask_value > 0.5, 1, 0)
        sourece = torch.where(self.SO.source_value > 0.5, 1, 0)
        gray_source = self.SO.source_value.detach().clone()

        AI_smoed_path = AI_folder / self.mask.mask_name
        RI_smoed_path = RI_folder / self.mask.mask_name
        RI_pvb_smoed_path = pvb_folder / self.mask.mask_name
        masked_smoed_path = masked_folder / self.mask.mask_name
        source_smoed_path = source_folder / self.mask.mask_name
        gray_source_path = gray_source_folder / self.mask.mask_name

        U.save_image(AI_smoed.to(torch.float32), AI_smoed_path)
        U.save_image(RI_smoed.to(torch.float32), RI_smoed_path)
        U.save_image(RI_pvb_smoed.to(torch.float32), RI_pvb_smoed_path)
        U.save_image(masked_smoed.to(torch.float32), masked_smoed_path)
        U.save_image(sourece.to(torch.float32), source_smoed_path)
        U.save_image(gray_source.to(torch.float32), gray_source_path)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        so_optimizer = self.hparams.so_optimizer(params=self.SO.parameters())
        mo_optimizer = self.hparams.mo_optimizer(params=self.MO.parameters())
        if self.hparams.so_scheduler is not None and self.hparams.mo_scheduler is not None:
            so_scheduler = self.hparams.so_scheduler(optimizer=so_optimizer)
            mo_scheduler = self.hparams.mo_scheduler(optimizer=mo_optimizer)
            return (
                {
                    "optimizer": so_optimizer,
                    "lr_scheduler": {
                        "scheduler": so_scheduler,
                        "monitor": "train/loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                },
                {
                    "optimizer": mo_optimizer,
                    "lr_scheduler": {
                        "scheduler": mo_scheduler,
                        "monitor": "train/loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                },
            )

        return ({"optimizer": so_optimizer}, {"optimizer": mo_optimizer})

if __name__ == "__main__":
    _ = SMOLitModule(None, None, None)
