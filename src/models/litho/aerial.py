"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2022-10-01 15:53:02
LastEditTime: 2023-04-06 19:30:22
Contact: gjchen21@cse.cuhk.edu.hk
Description:  use litho model to get the aerial image.
"""

from typing import Any

from src.models.litho.config import PATH
from src.models.litho.gds_mask import Mask
from src.models.litho.image import ImageHopkins, ImageHopkinsList
from src.models.litho.tcc import TCCDB, TCCList
from src.models.litho.utils import save_img_from_01torch, show_img, torch_arr_bound


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
        tccList: TCCList,
        vis: Any,
        resist_a: float = 80,
        resist_t: float = 0.6,
        resist_tRef: float = 0.12,
        doseList: list = [1.0],
        doseCoef: list = [1.0],
    ):
        self.image = ImageHopkinsList(mask, tccList)
        self.image.resist_a = resist_a
        self.image.resist_t = resist_t
        self.image.resist_tRef = resist_tRef
        self.image.doseList = doseList
        self.image.doseCoef = doseCoef
        self.xsize = self.image.mask.x_gridnum
        self.ysize = self.image.mask.y_gridnum

        self.vis = vis

    def litho(self):
        # I delete maskfft() here, and it did not influence.
        # because the mask already ffted.
        self.image.AIList = []
        self.image.RIList = []
        self.image.calculate()
        self.show_AI()
        self.show_RI()

    def show_AI(self):
        length = len(self.image.focusList)
        for ii in range(length):
            AI = self.image.AIList[ii]
            if self.vis.aerial_show:
                show_img(AI, f"AIList[{ii}]")
                torch_arr_bound(AI, f"AIList[{ii}]")
            if self.vis.aerial_save:
                save_img_from_01torch(AI, f"{self.vis.aerial_save_path}/AIList_{ii}.png")

    def show_RI(self):
        length = len(self.image.focusList)
        lengthD = len(self.image.doseList)
        for ii in range(length):
            for jj in range(lengthD):
                RI = self.image.RIList[ii][jj]
                if self.vis.resist_show:
                    show_img(RI, f"RIList[{ii}][{jj}]")
                    torch_arr_bound(RI, f"RIList[{ii}][{jj}]")
                if self.vis.resist_save:
                    save_img_from_01torch(RI, f"{self.vis.resist_save_path}/RIList_{ii}{jj}.png")


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

    # save_img_from_01torch(m.data, f'images/mask/{m_path}.png')

    # save m data first.
    # show_img(m.data, "mask image data")
    # arr_bound(m.data, "mask image bound")

    # '**************************************************'
    # we can save/load the TCC to disk here.
    # '**************************************************'

    # load tcc from db
    # tcc_db = get_tcc("./litho/db/tcclist")

    # print("Calculating Aerial image and resist image")
    # a = AerialList(m, tcc_db)
    # a.image.resist_a = 100
    # a.image.resist_tRef = 0.12

    # # a.image.doseList = [0.9, 1, 1.1]
    # # a.image.doseCoef = [0.3, 1, 0.3]
    # a.image.doseList = [1]
    # a.image.doseCoef = [1]
    # a.litho()
    # # a.show_AI(show=False, save=True)
    # a.show_AI(show=True, save=True)
    # # # a.show_RI(show=False, save=True)
    # a.show_RI(show=True, save=True)
