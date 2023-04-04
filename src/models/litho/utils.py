"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2022-10-17 11:50:53
LastEditTime: 2023-04-04 11:47:48
Contact: cgjcuhk@gmail.com
Description: some utils for image loading.
"""
from pathlib import Path
import torch
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
    # rprint(arr)
    min_wh = 7
    if arr.shape[0] >= min_wh and len(list(arr.shape)) > 1:
        lefti = arr.shape[0] // 2 - 4
        rprint(f"\n*************[yellow]{name}[{lefti}:{lefti+min_wh},{lefti}:{lefti+min_wh}][/yellow]*************")
        rprint(arr[lefti:lefti+min_wh,lefti:lefti+min_wh])
        rprint(f"*************[yellow]{name}[{lefti}:{lefti+7},{lefti}:{lefti+7}][/yellow]*************\n")
    else:
        rprint(arr)
    rprint(f"[yellow]{name}[/yellow].shape: {arr.shape}")
    rprint(f"[yellow]{name}[/yellow].dtype: {arr.dtype}")
    rprint(f"[yellow]{name}[/yellow] [red]sum[/red]: {np.sum(arr)}")
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


def torch_arr_bound(arr, name):
    rprint(f"\n=============[yellow]{name}[/yellow]================")
    # rprint(arr)
    min_wh = 7
    if arr.shape[0] >= min_wh and len(list(arr.shape)) > 1:
        lefti = arr.shape[0] // 2 - 4
        rprint(f"\n*************[yellow]{name}[{lefti}:{lefti+min_wh},{lefti}:{lefti+min_wh}][/yellow]*************")
        rprint(arr[lefti:lefti+min_wh,lefti:lefti+min_wh])
        rprint(f"*************[yellow]{name}[{lefti}:{lefti+7},{lefti}:{lefti+7}][/yellow]*************\n")
    else:
        rprint(arr)
    rprint(f"[yellow]{name}[/yellow].shape: {arr.shape}")
    rprint(f"[yellow]{name}[/yellow].dtype: {arr.dtype}")
    rprint(f"[yellow]{name}[/yellow] [red]sum[/red]: {torch.sum(arr)}")
    if arr.dtype == torch.complex128:
        rprint(f"[yellow]{name}.real[/yellow] sum: {torch.sum(arr.real)}")
        rprint(f"[yellow]{name}.imag[/yellow] sum: {torch.sum(arr.imag)}")
        rprint(f"[yellow]{name}.real[/yellow] max: {torch.max(arr.real)}")
        rprint(f"[yellow]{name}.real[/yellow] min: {torch.min(arr.real)}")
        rprint(f"[yellow]{name}.imag[/yellow] max: {torch.max(arr.imag)}")
        rprint(f"[yellow]{name}.imag[/yellow] min: {torch.min(arr.imag)}")
    else:
        rprint(f"[yellow]{name}[/yellow] max: {torch.max(arr)}")
        rprint(f"[yellow]{name}[/yellow] min: {torch.min(arr)}")
    rprint(f"==============[yellow]{name}[/yellow]======================\n")


def delta_np_torch(arr: np.array, tarr: torch.tensor):
    t_np_arr = torch.from_numpy(arr)
    rprint("=============[yellow]The delta is : [/yellow]=============")
    rprint(f"{torch.sum(t_np_arr - tarr)}\n")


def plot(ndarray):
    """plots Transversed image, with origin (0,0) at the lower left corner"""
    import matplotlib.pyplot as plt

    plt.imshow(ndarray.T, origin="lower")
    plt.show()
