"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-04-30 20:11:19
LastEditTime: 2023-11-13 09:15:13
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
        # self.open_layout()
        # self.maskfft()

    def open_layout(self, resize=False):
        img = Image.open(self.layout_path)
        img = img.convert("L")
        target = Image.open(self.target_path)
        target = target.convert("L")

        transform = T.PILToTensor()
        data = transform(img) // 255
        target_data = transform(target) // 255
        if resize:
            # resize_transform = T.Resize((1024, 1024))
            resize_transform = T.Resize((2048, 2048))
            data = resize_transform(data)
            target_data = resize_transform(target_data)
        self.data = data[0]
        self.y_gridnum = self.data.shape[0]
        self.x_gridnum = self.data.shape[1]
        self.target_data = target_data[0]


        # 3 channels


        # 1 channel

    # use the fftw packages
    def maskfft(self):
        self.fdata = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(self.data)))


if __name__ == "__main__":
    png_path = (
        "data/soed/ibm_opc_test/mask/t1_0_mask.png"
    )
    m = Mask(layout_path=png_path, target_path=png_path)
    m.open_layout()
    # print(m.data.max())
    from utils import torch_arr_bound

    torch_arr_bound(m.data, "m.data")
