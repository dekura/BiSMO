'''
Author: Hongquan
begin: tcclist, mask
end: AI, RI
'''

import sys
sys.path.append('.')

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

from src.models.litho.tcc import TCCList
from src.models.litho.lens import LensList
from src.models.litho.img_mask import Mask
from src.models.litho.source import Source
from src.models.litho.utils import torch_arr_bound
from src.models.litho.aerial import AerialList

class SMOLitModule(LightningModule):
    '''
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)
    '''
    def __init__(
            self, 
            source: Source, 
            lens: LensList, 
            mask: Mask, 
            weight_pvb: float = 1.0, 
            weight_l2:float = 1.0, 
            dose_list: list = [0.98, 1.00, 1.02,], 
            resist_tRef: float = 0.06, 
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["source", "mask"])

        # loss func
        self.criterion = nn.MSELoss()

        # activation func
        self.mask_acti = nn.Sigmoid()
        
        # weight of loss
        self.weight_pvb = weight_pvb
        self.weight_l2 = weight_l2
        
        # source
        self.source = source
        self.source.update()

        # mask
        self.mask = mask
        self.mask.open_layout()
        self.mask.maskfft()

        # lens
        self.lens = lens
        self.lens.focusList = [0.0]
        self.lens.focusCoef = [1.0]
        self.lens.calculate()
        
        # dose list
        self.dose_list = dose_list
        self.resist_tRef = resist_tRef

        # for averaging loss
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def cal_tcc(self):
        self.tcc = TCCList(self.source, self.lens)
        # tcc = TCCList(self.source, self.lens)
        # return TCCList(self.source, self.lens)

    def sigmoid_resist(self, aerial):
        return torch.sigmoid(
            self.hparams.resist_sigmoid_steepness * (aerial - self.hparams.target_intensity)
        )

    def init_freq_domain_on_device(self):
        device = self.device
        # hyper-parameters
        # self.gnum = self.source.gnum
        # self.fnum = self.source.fnum
        # self.source.fx.to(device)

        # lens, source
        self.source.detaf = self.source.detaf.to(device)
        self.source.detag = self.source.detag.to(device)

        self.lens.detaf = self.lens.detaf.to(device)
        self.lens.detag = self.lens.detag.to(device)

        # load source data
        self.source.data = self.source.data.to(device)
        self.source.data = self.source.mdata.to(device)

        # load lens
        self.lens.fDataList = self.lens.fDataList.to(device)
        self.lens.fDataList = self.lens.sDataList.to(device)
        
        # load mask data to device
        # mask has 3 datas.
        self.mask.data = self.mask.data.to(torch.float32).to(device)
        self.mask.fdata = self.mask.fdata.to(device)

        # load target data to device
        if hasattr(self.mask, "target_data"):
            self.mask.target_data = self.mask.target_data.to(torch.float32).to(device)
        else:
            self.mask.target_data = self.mask.data.detach().clone()

    def init_mask_params(self):
        self.mask_params = nn.Parameter(torch.zeros(self.mask.data.shape))
        self.mask_params.data = self.mask.target_data

        # if self.hparams.mask_acti == "sigmoid":
            # self.mask_params = torch.sigmoid()
            # self.mask_params.data[torch.where(self.s.data > 0.5)] = 2 - 0.02
            # self.mask_params.data.sub_(0.99)

    def update_mask_value(self):
        # Filter
        self.filter = torch.zeros([self.mask.data.shape[0] * 2, self.mask.data.shape[1] * 2], dtype=torch.float32, device=self.device)
        self.filter[self.mask.data.shape[0] * 0.5 : self.mask.data.shape[0] * 1.5, \
                     self.mask.data.shape[1] * 0.5 : self.mask.data.shape[1] * 1.5] = 1
    
        if self.hparams.mask_acti == "sigmoid":
            self.mask_value = self.mask_acti(
                self.hparams.mask_sigmoid_steepness * self.mask_params
            )
        elif self.hparams.mask_acti == "cosine":
            self.mask_value = (1 + torch.cos(self.mask_params)) / 2
        else:
            self.mask_value = self.mask_acti(
                self.hparams.mask_sigmoid_steepness * self.mask_params
            )

    def forward(self):
        self.tcc = TCCList(self.source, self.lens)
        self.update_mask_value()
        self.aerial = AerialList(mask = self.mask_params, 
                                 tccList = self.tcc, 
                                #  resist_tRef = self.resist_tRef, 
                                resist_tRef = self.hparams.target_intensity, 
                                 doseList = self.dose_list, 
                                 )
        # return AI, RI
        self.AIlist = self.aerial.image.AIList
        self.RIlist = self.aerial.image.RIList
        return self.AIlist, self.RIlist
        # pass

    def model_step(self):
        AI, RI = self.forward()
        # pvb, l2
        pvb = (torch.sum((RI[0] - RI[1]) ** 2) + torch.sum((RI[2] - RI[1]) ** 2)).requires_grad_(True)
        l2 = self.criterion(RI[1], self.mask.target_data)
        # loss
        loss = (pvb * self.weight_pvb) + (l2 * self.weight_l2)
        return loss, AI, RI
        # pass

    def training_step(self, batch: Any, batch_idx: int):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def configure_optimizers(self):
        pass

if __name__ == "__main__":
    SMOLitModule(None, None, None)