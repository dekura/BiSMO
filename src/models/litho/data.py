"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2022-10-01 15:53:02
LastEditTime: 2022-11-16 22:58:22
Contact: gjchen21@cse.cuhk.edu.hk
Description:  use litho model to get the aerial image.
"""
import argparse
import timeit
from pathlib import Path

import torch
from tqdm import tqdm

from litho.consts import IMAGE_WH, TCC_ORDER
from litho.image import ImageHopkins, ImageHopkinsList
from litho.lens import LensList
from litho.mask import Mask
from litho.source import Source
from litho.tcc import TCC_db, TCCList
from litho.utils import arr_bound, save_img_from_01np, show_img


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
    s.maskxpitch = IMAGE_WH
    s.maskypitch = IMAGE_WH
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
    t.order = TCC_ORDER
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

    def show_AI(self, show=True, save=False, save_name=None):
        length = len(self.image.focusList)
        for ii in range(length):
            AI = self.image.AIList[ii]
            if show:
                show_img(AI, f"AI[{ii}]_{save_name}")
                arr_bound(AI, f"arr{ii}_{save_name}")
            if save:
                save_img_from_01np(AI, f"{save_name}")

    def show_RI(self, show=True, save=False, save_name=None):
        length = len(self.image.focusList)
        lengthD = len(self.image.doseList)
        for ii in range(length):
            for jj in range(lengthD):
                RI = self.image.RIList[ii][jj]
                if show:
                    show_img(RI, f"RIList[{ii}][{jj}]_{save_name}")
                    arr_bound(RI, f"RIList[{ii}][{jj}]_{save_name}")
                if save:
                    save_img_from_01np(RI, f"{save_name}")


if __name__ == "__main__":

    # a = time.time()
    # '**************************************************'
    # we can save/load the TCC to disk here.
    # '**************************************************'
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--p", default="A", type=str, help="the paramters number")
    args = parser.parse_args()
    alpha = args.p

    tcc_db = get_tcc("./litho/db/tcclist")
    gds_folder = Path("./NanGateLibGDS")
    match_str = f"{alpha}*.gds"
    gds_paths = list(gds_folder.glob(match_str))
    # print(gds_paths)

    for ps in tqdm(gds_paths):
        print(ps)
        gds_name = ps.name
        for layer in [9, 10, 11]:
            if layer == 10:
                layer_name = "via"
            else:
                layer_name = "metal"

            m = Mask()
            m.openGDS(ps, layer, 0.1, pixels_per_um=100)
            for idx, img in enumerate(m.mask_groups):
                start = timeit.default_timer()
                print(
                    f"{match_str}: process {str(ps)} layer {layer} mask {idx + 1}/{len(m.mask_groups)}"
                )
                m.data = img
                # this line is required to initialize the mask.
                m.maskfft()
                m_name = f"{gds_name}_l{layer}_{idx}.png"
                save_img_from_01np(m.data, f"images/{layer_name}/mask/{m_name}")
                # show_img(m.data, "mask image data")
                a = AerialList(m, tcc_db)
                a.image.resist_tRef = 0.1
                a.image.doseList = [1]
                a.image.doseCoef = [1]
                a.litho()
                end = timeit.default_timer()
                print(end - start)
                a.show_AI(
                    show=False, save=True, save_name=f"images/{layer_name}/AI/{m_name}"
                )
                a.show_RI(
                    show=False, save=True, save_name=f"images/{layer_name}/RI/{m_name}"
                )
