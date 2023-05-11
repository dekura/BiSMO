"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-05-06 16:00:12
LastEditTime: 2023-05-06 17:32:27
Contact: cgjcuhk@gmail.com
Description: generate the dataset for discrete diffusion model
"""
from pathlib import Path

import numpy as np

# import os, sys
import pyrootutils
import torch

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# sys.path.append()

from src.models.litho.img_mask import Mask

source_types = ["annular", "dipole", "quasar"]
folder_types = ["RI", "SO", "mask"]
dataset_name = "iccad13"


def gen_dict(mask_path, source_path, source_img_path, source_class):
    """
    source_path: the source shape after SMO
    source_img_path: the source shape before SMO
    """
    return {
        "source_img": load_source(source_img_path),
        "class": source_class,
        "mask": load_mask(mask_path),  # 2 \times 51 \times 51
        "gt": load_source(source_path),  # 1 \times 51 \times  51
    }


def load_mask(png_path):
    x_gridnum = 2048
    y_gridnum = 2048
    fnum = 28
    gnum = 28

    x1 = int(x_gridnum // 2 - fnum)
    x2 = int(x_gridnum // 2 + fnum + 1)
    y1 = int(y_gridnum // 2 - gnum)
    y2 = int(y_gridnum // 2 + gnum + 1)
    m = Mask(layout_path=png_path, target_path=png_path)
    m.open_layout()
    m.maskfft()
    fdata = m.fdata[y1:y2, x1:x2]
    r = fdata.real
    i = fdata.imag
    trans = torch.stack((r, i))
    print(f"mask shape: {trans.shape}")
    return trans.numpy()


def load_source(source_path):
    s = Mask(layout_path=source_path, target_path=source_path)
    s.open_layout()
    data = s.data.unsqueeze(0)
    print(f"source shape: {data.shape}")
    return data.numpy()


def get_source_class(source_type):
    print(source_type)
    return source_types.index(source_type)


def save_npz():
    save_img_folder = "/home/gjchen21/projects/smo/SMO-ICCAD23/data/soed/ibm_opc_test/"
    source_gt_folder = "/home/gjchen21/projects/smo/SMO-ICCAD23/data/source_gt/"
    save_img_folder = Path(save_img_folder)
    source_gt_folder = Path(source_gt_folder)

    all_data = []

    for source_type in source_types:
        RI_folder = save_img_folder / f"{source_type}_RI"
        RI_folder.mkdir(parents=True, exist_ok=True)
        SO_folder = save_img_folder / f"{source_type}_SO"
        SO_folder.mkdir(parents=True, exist_ok=True)
        mask_folder = save_img_folder / f"{source_type}_mask"
        mask_folder.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            i = i + 1
            name = f"t{i}_0_mask.png"
            mask_path = mask_folder / name
            source_path = SO_folder / name
            source_img_path = source_gt_folder / f"{source_type}.png"
            source_class = get_source_class(source_type)
            d = gen_dict(mask_path, source_path, source_img_path, source_class)
            all_data.append(d)

    npz_save_path = "/home/gjchen21/projects/smo/SMO-ICCAD23/data/npz/smo_iccad13.npz"
    np.savez(npz_save_path, all_data)


def load_npz():
    npz_save_path = "/home/gjchen21/projects/smo/SMO-ICCAD23/data/npz/smo_iccad13.npz"
    npzfile = np.load(npz_save_path, allow_pickle=True)
    print(npzfile["arr_0"])


if __name__ == "__main__":
    # save_npz()
    load_npz()
