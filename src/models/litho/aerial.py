"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2022-10-01 15:53:02
LastEditTime: 2022-11-02 04:30:01
Contact: gjchen21@cse.cuhk.edu.hk
Description:  use litho model to get the aerial image.
"""
from pathlib import Path

import torch

from litho.config import PATH
from litho.image import ImageHopkins, ImageHopkinsList
from litho.lens import LensList
from litho.mask import Mask
from litho.source import Source
from litho.tcc import TCC_db, TCCList
from litho.utils import arr_bound, save_img_from_01np, show_img

# import scipy.signal as sg

TCC_DB_NAME = "./litho/db/tcclist"


def get_tcc(db_path: str = None):
    db_ext = [".db", ".dat"]
    db_path_ext = [Path(db_path + item) for item in db_ext]

    # if have db, load and return
    for p in db_path_ext:
        if p.exists():
            tdb = TCC_db()
            tdb.load_db(db_path)
            print(f"load tcc from {db_path}")
            return tdb

    # else, calculate tcc and save to db.
    # first create folder.
    # if not Path(db_path).parent.exists():
    Path(db_path).parent.mkdir(exist_ok=True)
    s = Source()
    s.na = 1.35
    # s.maskxpitch = m.x_range[1] - m.x_range[0]
    # s.maskypitch = m.y_range[1] - m.y_range[0]
    s.maskxpitch = 2048
    s.maskypitch = 2048
    s.type = "annular"
    s.sigma_in = 0.6
    s.sigma_out = 0.9
    s.smooth_deta = 0.00
    s.shiftAngle = 0
    s.update()
    s.ifft()

    o = LensList()
    o.na = s.na
    o.maskxpitch = s.maskxpitch
    o.maskypitch = s.maskypitch
    o.focusList = [0]
    o.focusCoef = [1]
    o.calculate()

    print("Calculating TCC and SVD kernels")
    t = TCCList(s, o)
    t.order = 30
    t.calculate()
    print("calculate tcc directly")
    # then save it
    tdb = TCC_db()
    tdb.save_db(t, db_path)
    return t


def load_tcc_from_net(tcc_db_path: str = None, net_pred_path: str = None):
    tdb = TCC_db()
    tdb.load_db(tcc_db_path)
    print(f"load tcc from {tcc_db_path}")
    # print(f"original kernel")
    print(tdb.kernelList[0][1, 0, :])
    print(f"loaded kerner from {net_pred_path}")
    if torch.cuda.is_available():
        pred = torch.load(net_pred_path)
    else:
        pred = torch.load(net_pred_path, map_location=torch.device("cpu"))
    # print(pred.shape)
    pred = pred.view(29, 19, 7)
    pred = pred * 1e-5
    # print(pred.shape)
    print(pred[1, 0, :])
    print("change kernel[0]")
    tdb.kernelList[0] = pred.clone().detach().numpy()
    print(tdb.kernelList[0][1, 0, :])
    return tdb


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
    def __init__(self, mask, tccList):
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

    # load tcc from predicted db
    # tcc_db = load_tcc_from_net(
    #     tcc_db_path="./litho/db/tcclist",
    #     net_pred_path="./litho/kernels/pred_min_3_i_17438.pt",
    # )

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
