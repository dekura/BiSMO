"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2022-10-17 11:50:53
LastEditTime: 2023-03-23 11:15:02
Contact: cgjcuhk@gmail.com
Description: some utils for image loading.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rich import print as rprint


def save_img_from_01np(np_arr, file_path):
    img = (np_arr * 255).astype(np.uint8)
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


def arr_bound(arr, name):
    rprint(f"\n=============[yellow]{name}[/yellow]================")
    rprint(arr)
    rprint(f"[yellow]{name}[/yellow].shape: {arr.shape}")
    rprint(f"[yellow]{name}[/yellow].dtype: {arr.dtype}")
    rprint(f"[yellow]{name}[/yellow] [red]sum: {np.sum(arr)}")
    if arr.dtype == "complex128":
        rprint(f"[yellow]{name}.real[/yellow] sum: {np.sum(arr.real)}")
        rprint(f"[yellow]{name}.imag[/yellow] sum: {np.sum(arr.imag)}")
        rprint(f"[yellow]{name}.real[/yellow] max: {np.max(arr.real)}")
        rprint(f"[yellow]{name}.real[/yellow] min: {np.min(arr.real)}")
        rprint(f"[yellow]{name}.imag[/yellow] max: {np.max(arr.imag)}")
        rprint(f"[yellow]{name}.imag[/yellow] min: {np.min(arr.imag)}")
    else:
        rprint(f"[yellow]{name}[/yellow] max: {np.max(arr)}")
        rprint(f"[yellow]{name}[/yellow] min: {np.min(arr)}")
    rprint(f"==============[yellow]{name}[/yellow]======================\n")


def plot(ndarray):
    """plots Transversed image, with origin (0,0) at the lower left corner"""
    import matplotlib.pyplot as plt

    plt.imshow(ndarray.T, origin="lower")
    plt.show()
