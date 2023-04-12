"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-04-12 15:29:06
LastEditTime: 2023-04-12 22:53:32
Contact: cgjcuhk@gmail.com
Description: 
"""
"""
"""
import numpy as np
import pyfftw
import scipy as sci
import shelve
import time
import torch
# from src.models.litho.lens import LensList
# from src.models.litho.source import Source

from lens_torch import LensList
from source_torch import Source
from mask_torch import Mask

from utils import torch_arr_bound, show_img

class ICC:
    """
    The illumination-cross coefficient.
    """

    def __init__(self, source, lens, mask):
        self.s = source
        self.s.update()
        self.s.ifft()

        self.lens = lens
        self.lens.update()
        self.lens.calPupil()
        self.lens.calPSF()

        self.mask = mask
        self.mask.maskfft()

        self.order = 7
        self.psf = lens.data

        self.norm = self.mask.y_gridnum * self.mask.x_gridnum
        self.x1 = int(self.mask.x_gridnum // 2 - self.lens.fnum)
        self.x2 = int(self.mask.x_gridnum // 2 + self.lens.fnum + 1)
        self.y1 = int(self.mask.y_gridnum // 2 - self.lens.gnum)
        self.y2 = int(self.mask.y_gridnum // 2 + self.lens.gnum + 1)
        self.x_gridnum = self.mask.x_gridnum
        self.y_gridnum = self.mask.y_gridnum
        self.gnum = self.lens.gnum
        self.fnum = self.lens.fnum

        self.aerial = torch.zeros((self.y_gridnum, self.x_gridnum))
        self.icc2d_spat_part = torch.zeros((self.y_gridnum, self.x_gridnum), dtype=torch.complex128)
        self.icc2d_freq_part = torch.zeros((self.y_gridnum, self.x_gridnum), dtype=torch.complex128)
        # self.finalAI = torch.zeros((self.y_gridnum, self.x_gridnum))


    def calMutualIntensity(self):
        self.gnum, self.fnum = self.s.data.shape
        J = torch.zeros((self.gnum, self.fnum, self.gnum, self.fnum), dtype=torch.float64)
        for ii in range(self.gnum):
            for jj in range(self.fnum):
                J[:, :, ii, jj] = self.s.spatMutualData.real[
                    (self.gnum - ii - 1) : (2 * self.gnum - ii - 1),
                    (self.fnum - jj - 1) : (2 * self.fnum - jj - 1),
                ]
        self.jsource = torch.reshape(J, (self.gnum * self.fnum, self.gnum, self.fnum))
        # self.jsource = torch.reshape(J, (self.gnum, self.fnum, self.gnum * self.fnum))
        # self.jsource = torch.reshape(self.jsource, (self.gnum * self.fnum * self.gnum * self.fnum, 1))

    def calSpatICC(self):
        abbe2d_freq = torch.zeros((self.mask.y_gridnum, self.mask.x_gridnum), dtype=torch.complex128)
        abbe2d_freq[self.y1 : self.y2, self.x1 : self.x2] = self.psf  * self.mask.fdata[self.y1 : self.y2, self.x1 : self.x2]
        # self.icc2d = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(abbe2d_freq)))
        self.icc2d = abbe2d_freq
        self.icc2d = torch.abs(self.icc2d * self.icc2d)
        # self.icc2d = torch.abs(abbe2d_freq * torch.conj(abbe2d_freq))
        # self.icc2d =  torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(self.icc2d)))
        # self.icc2d = torch.real(self.icc2d)
        # abbe2d_spat = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(abbe2d_freq)))
        # self.abbe2d_spat = abbe2d_spat
        # self.icc2d = torch.abs(abbe2d_spat * torch.conj(abbe2d_spat))

    # def icc2dfft(self):
    #     self.icc2d_spat_part[:] = torch.fft.ifftshift(self.abbe2d_spat)
    #     self.icc2d_freq_part = torch.fft.fftn(self.icc2d_spat_part)
    #     self.icc2d_fdata = torch.fft.fftshift(self.icc2d_freq_part)
    #     self.icc2d_fdata = torch.abs(self.icc2d_fdata * self.icc2d_fdata)


    def calculate(self):
        self.calSpatICC()
        self.calMutualIntensity()
        AI = torch.zeros((self.mask.y_gridnum, self.mask.x_gridnum))
        # self.finalAI = torch.sum(self.jsource) * self.icc2d
        for i in range(self.gnum * self.fnum):
            e_field = torch.zeros(
                (self.mask.y_gridnum, self.mask.x_gridnum), dtype=torch.float32
            )
            e_field[self.y1 : self.y2, self.x1 : self.x2] = (
                self.jsource[i, :, :] * self.icc2d[self.y1 : self.y2, self.x1 : self.x2]
            )
            AA = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(e_field)))
            AA = torch.real(AA)
            AI += AA
        self.finalAI = AI
        # self.icc2dfft()
        # self.finalAI_freq = torch.zeros(
        #     (self.mask.y_gridnum, self.mask.x_gridnum), dtype=torch.complex128
        # )
        # for i in range(self.gnum * self.fnum):
        #     self.finalAI_freq[self.y1 : self.y2, self.x1 : self.x2] += \
        #         self.jsource[:, :, i] * self.icc2d[self.y1 : self.y2, self.x1 : self.x2]
        # self.finalAI = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(self.finalAI_freq)))
        # self.finalAI = torch.real(self.finalAI)
        # # self.AI = self.jsource * self.icc2d

if __name__ == "__main__":
    s = Source()
    s.type = "annular"
    s.na = 1.35
    s.maskxpitch = 2048
    s.maskypitch = 2048
    s.sigma_out = 0.9
    s.sigma_in = 0.6
    s.smooth_deta = 0
    s.shiftAngle = 0
    s.update()
    s.ifft()
    print(s.data.size())


    o = LensList()
    o.maskxpitch = 2048
    o.maskypitch = 2048
    o.na = 1.35
    o.focusList = [0.0]
    o.focusCoef = [1.0]
    # o.calculate()

    gds_path = "/home/gjchen21/phd/projects/smo/SMO-ICCAD23/data/NanGateLibGDS/NOR2_X2.gds"
    m = Mask(gds_path, 11)
    m.openGDS()
    # m.maskfft()

    icc = ICC(s, o, m)
    icc.calculate()

    torch_arr_bound(icc.icc2d, "icc.icc2d")
    show_img(icc.icc2d, "icc.icc2d")
    torch_arr_bound(icc.jsource, "icc.jsource")
    torch_arr_bound(icc.finalAI, "icc.finalAI")
    show_img(icc.finalAI, "icc.finalAI")
    ## jsource we have, jsource is real-number matrix.