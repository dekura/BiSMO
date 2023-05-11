"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-04-30 20:11:19
LastEditTime: 2023-04-30 23:56:51
Contact: cgjcuhk@gmail.com
Description:
"""
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T


class Mask:
    """Binary Mask.

    Read from image.
    """

    def __init__(
        self,
        layout_path: str,
        target_path: str,
        dataset_name: str = "iccad13",
    ):
        self.dataset_name = dataset_name
        self.layout_path = layout_path
        self.target_path = target_path
        self.mask_name = Path(layout_path).name
        self.target_name = Path(target_path).name

        """
        Process calculation
        """
        # self.openGDS()
        # self.maskfft()

    def open_layout(self):
        img = Image.open(self.layout_path)
        img = img.convert("L")
        transform = T.PILToTensor()
        data = transform(img) // 255
        self.data = data[0]
        self.y_gridnum = self.data.shape[0]
        self.x_gridnum = self.data.shape[1]

        target = Image.open(self.target_path)
        target = target.convert("L")
        target_data = transform(target) // 255
        self.target_data = target_data[0]

    # use the fftw packages
    def maskfft(self):
        self.fdata = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(self.data)))


if __name__ == "__main__":
    png_path = (
        "/home/gjchen21/phd/projects/smo/SMO-ICCAD23-torch/data/ibm_opc_test/mask/t1_0_mask.png"
    )
    m = Mask(layout_path=png_path, target_path=png_path)
    m.open_layout()
    from utils import torch_arr_bound

    torch_arr_bound(m.data, "m.data")
