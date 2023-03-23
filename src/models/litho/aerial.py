"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2022-10-01 15:53:02
LastEditTime: 2023-03-23 16:21:28
Contact: gjchen21@cse.cuhk.edu.hk
Description:  use litho model to get the aerial image.
"""
from pathlib import Path

import torch

from src.models.litho.config import PATH
from src.models.litho.image import ImageHopkins, ImageHopkinsList
from src.models.litho.lens import LensList
from src.models.litho.mask import Mask
from src.models.litho.source import Source
from src.models.litho.tcc import TCCList, TCCDB
from src.models.litho.utils import arr_bound, save_img_from_01np, show_img

class Aerial:
    def __init__(self, m, t):
        self.image = ImageHopkins(m, t)
        self.xsize = self.image.mask.x_gridnum
        self.ysize = self.image.mask.y_gridnum

    def litho(self):
        self.mask_init()
        self.image.mask.maskfft()
        self.image.calAI()
        self.image.calRI()


class AerialList(Aerial):
    def __init__(
            self,
            mask: Mask,
            tccList: TCCList):
        self.image = ImageHopkinsList(mask, tccList)
        self.xsize = self.image.mask.x_gridnum
        self.ysize = self.image.mask.y_gridnum

    def litho(self):
        # I delete maskfft() here, and it did not influence.
        # because the mask already ffted.
        self.image.AIList = []
        self.image.RIList = []
        self.image.calculate()

    def show_AI(self, show=True, save=False):
        length = len(self.image.focusList)
        for ii in range(length):
            AI = self.image.AIList[ii]
            if show:
                show_img(AI, f"AIList[{ii}]")
                arr_bound(AI, f"AIList[{ii}]")
            if save:
                save_img_from_01np(AI, f"images/AI/AIList_{ii}.png")

    def show_RI(self, show=True, save=False):
        length = len(self.image.focusList)
        lengthD = len(self.image.doseList)
        for ii in range(length):
            for jj in range(lengthD):
                RI = self.image.RIList[ii][jj]
                if show:
                    show_img(RI, f"RIList[{ii}][{jj}]")
                    arr_bound(RI, f"RIList[{ii}][{jj}]")
                if save:
                    save_img_from_01np(RI, f"images/RI/RIList_{ii}{jj}.png")


if __name__ == "__main__":

    # a = time.time()
    m = Mask()
    m.x_range = [-1024.0, 1024.0]
    m.y_range = [-1024.0, 1024.0]
    m.x_gridsize = 1
    m.y_gridsize = 1
    m_path = "NOR2_X2.gds"
    m.openGDS(PATH.gdsdir / m_path, 11, 0.1, pixels_per_um=100)
    # arr_bound(m.data, "mask.data")
    # this line is required to initialize the mask.
    m.maskfft()
    # arr_bound(m.spat_part, "mask.spat_part")
    # arr_bound(m.freq_part, "mask.freq_part")
    # arr_bound(m.fdata, "mask.fdata")

    # save_img_from_01np(m.data, f'images/mask/{m_path}.png')

    # save m data first.
    # show_img(m.data, "mask image data")
    # arr_bound(m.data, "mask image bound")

    # '**************************************************'
    # we can save/load the TCC to disk here.
    # '**************************************************'

    # load tcc from db
    tcc_db = get_tcc("./litho/db/tcclist")

    print("Calculating Aerial image and resist image")
    a = AerialList(m, tcc_db)
    a.image.resist_a = 100
    a.image.resist_tRef = 0.12

    # a.image.doseList = [0.9, 1, 1.1]
    # a.image.doseCoef = [0.3, 1, 0.3]
    a.image.doseList = [1]
    a.image.doseCoef = [1]
    a.litho()
    # a.show_AI(show=False, save=True)
    a.show_AI(show=True, save=True)
    # # a.show_RI(show=False, save=True)
    a.show_RI(show=True, save=True)
