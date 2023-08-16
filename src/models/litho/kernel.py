"""
Author: Hongquan
Contact: fallenhhq@gmail.com
Description: load kernel from ./kernel.
m.data, m.fdata, m.sdata are used.
"""
import torch
import torch.nn as nn

# used in simple_ilt.py
# one config - kernel path
# torch.Size([24, 35, 35])
class Kernel:
    def __init__(self, basedir="./kernel", defocus=False, conjuncture=False, combo=False, device=torch.device("cuda")):
        self._basedir = basedir
        self._defocus = defocus
        self._conjuncture = conjuncture
        self._combo = combo
        self._device = device

        self._kernels = torch.load(self._kernel_file(), map_location=device).permute(2, 0, 1)
        self._scales = torch.load(self._scale_file(), map_location=device)

        self._knx, self._kny = self._kernels.shape[:2]

    @property
    def kernels(self): 
        return self._kernels
        
    @property
    def scales(self): 
        return self._scales

    def _kernel_file(self):
        filename = ""
        if self._defocus:
            filename = "defocus" + filename
        else:
            filename = "focus" + filename
        if self._conjuncture:
            filename = "ct_" + filename
        if self._combo:
            filename = "combo_" + filename
        filename = self._basedir + "/kernels/" + filename + ".pt"
        return filename

    def _scale_file(self):
        filename = self._basedir + "/scales/"
        if self._combo:
            return filename + "combo.pt"
        else:
            if self._defocus:
                return filename + "defocus.pt"
            else:
                return filename + "focus.pt"


if __name__ == '__main__':
    k = Kernel()
    print(k.kernels.shape)
