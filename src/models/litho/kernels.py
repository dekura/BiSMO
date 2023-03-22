"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2022-10-01 15:53:02
LastEditTime: 2022-10-03 16:45:29
Contact: gjchen21@cse.cuhk.edu.hk
Description:  use litho model to get the aerial image.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from litho.image import ImageHopkins, ImageHopkinsList
from litho.lens import LensList
from litho.source import Source
from litho.tcc import TCC_db, TCCList

# import scipy.signal as sg

TCC_DB_NAME = "./litho/db/tcclist"


def save_img_from_01np(np_arr, file_path):
    img = np_arr * 255
    img = Image.fromarray(img)
    if img.mode == "F":
        img = img.convert("L")
    if not Path(file_path).parent.exists():
        Path(file_path).parent.mkdir(parents=True)
    img.save(f"{file_path}")


def show_img(arr, name):
    plt.figure()
    plt.title(f"{name}")
    plt.imshow(arr, cmap="hot", interpolation="none")
    plt.show()


def img_bound(arr, name):
    print(arr.shape)
    print(f"{name} max: {np.max(arr)}")
    print(f"{name} min: {np.min(arr)}")


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
    s.maskxpitch = 600
    s.maskypitch = 1000
    s.type = "annular"
    s.sigma_in = 0.7
    s.sigma_out = 0.9
    s.smooth_deta = 0.00
    s.shiftAngle = 0
    s.update()
    s.ifft()

    o = LensList()
    o.na = s.na
    o.maskxpitch = s.maskxpitch
    o.maskypitch = s.maskypitch
    o.focusList = [0, 80]
    o.focusCoef = [1, 0.5]
    o.calculate()

    print("Calculating TCC and SVD kernels")
    t = TCCList(s, o)
    t.calculate()
    print("calculate tcc directly")
    # then save it
    tdb = TCC_db()
    tdb.save_db(t, db_path)
    return t


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
                img_bound(AI, f"AIList[{ii}]")
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
                    img_bound(RI, f"RIList[{ii}][{jj}]")
                if save:
                    save_img_from_01np(RI, f"images/RI/RIList_{ii}{jj}.png")


if __name__ == "__main__":

    # a = time.time()
    # m = Mask()
    # m.x_range = [-300.0, 300.0]
    # m.y_range = [-300.0, 300.0]
    # m.x_gridsize = 1
    # m.y_gridsize = 1
    # m_path = "NOR2_X2.gds"
    # m.openGDS(PATH.gdsdir / m_path, 11, 0.1)
    # # this line is required to initialize the mask.
    # m.maskfft()
    # save_img_from_01np(m.data, f'images/mask/{m_path}.png')

    # save m data first.
    # show_img(m.data, "mask image data")
    # img_bound(m.data, "mask image bound")

    # '**************************************************'
    # we can save/load the TCC to disk here.
    # '**************************************************'

    # load tcc from db
    # tcc_db = get_tcc("./litho/db/tcclist")
    # print("Calculating Aerial image and resist image")
    # # print(tcc_db)
    # # print(tcc_db.kernelList)
    # # 2 set of kernels here
    # # print(tcc_db.kernelList[0].shape)
    # # print(tcc_db.coefList[0].shape)
    # print(tcc_db.kernelList[0][0,0, 1])
    # print(tcc_db.coefList[0][1])
    # print(tcc_db.coefList[0][1].dtype)
    # print(tcc_db.kernelList[0][:,:, 1].shape)
    # print(tcc_db.kernelList[0][:,:, 1].dtype)
    # torch_kernels = torch.from_numpy(tcc_db.kernelList[0])
    # torch_coefs = torch.from_numpy(tcc_db.coefList[0])
    # torch.save(torch_kernels, './litho/kernels/kernels.pt')
    # torch.save(torch_coefs, './litho/kernels/coefs.pt')

    torch_n_kernels = torch.load("./litho/kernels/kernels.pt")
    torch_n_coefs = torch.load("./litho/kernels/coefs.pt")
    print(torch_n_kernels * 100000)
    print(torch_n_kernels.shape)
    # print(torch_n_kernels[1,12,:])
    print(torch_n_coefs)
    # print(torch_kernels[0,0,1])
    # a = AerialList(m, tcc_db)
    # a.image.resist_a = 100
    # a.image.resist_tRef = 0.6
    # a.image.doseList = [0.9, 1, 1.1]
    # a.image.doseCoef = [0.3, 1, 0.3]
    # a.litho()
    # # a.show_AI(show=False, save=True)
    # a.show_AI(show=True, save=True)
    # # a.show_RI(show=False, save=True)
    # a.show_RI(show=True, save=True)
