"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-04-12 15:29:06
LastEditTime: 2023-04-20 16:01:56
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
import copy
import torch
# from src.models.litho.lens import LensList
# from src.models.litho.source import Source

from lens_torch import LensList
from source_torch import Source
from mask_torch import Mask
from zernike_torch import zerniken
from utils import torch_arr_bound, show_img

class ICC:
    """
    The illumination-cross coefficient.
    """

    def __init__(self, source, lens, mask):
        self.s = source
        self.s.update()
        self.s.ifft()
        self.s.simple_source()

        self.lens = lens
        self.lens.update()
        self.lens.calPupil()
        self.lens.calPSF()

        self.mask = mask
        self.mask.maskfft()

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


    # def calMutualIntensity(self):
    #     self.gnum, self.fnum = self.s.data.shape
    #     J = torch.zeros((self.gnum, self.fnum, self.gnum, self.fnum), dtype=torch.float64)
    #     for ii in range(self.gnum):
    #         for jj in range(self.fnum):
    #             J[:, :, ii, jj] = self.s.spatMutualData.real[
    #                 (self.gnum - ii - 1) : (2 * self.gnum - ii - 1),
    #                 (self.fnum - jj - 1) : (2 * self.fnum - jj - 1),
    #             ]
    #     self.jsource = torch.reshape(J, (self.gnum * self.fnum, self.gnum, self.fnum))
        # self.jsource = torch.reshape(J, (self.gnum, self.fnum, self.gnum * self.fnum))
        # self.jsource = torch.reshape(self.jsource, (self.gnum * self.fnum * self.gnum * self.fnum, 1))

    # def calSpatICC(self):
        # abbe2d_freq = torch.zeros((self.mask.y_gridnum, self.mask.x_gridnum), dtype=torch.complex128)
        # abbe2d_freq[self.y1 : self.y2, self.x1 : self.x2] = self.psf  * self.mask.fdata[self.y1 : self.y2, self.x1 : self.x2]
        # self.icc2d = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(abbe2d_freq)))
        # self.icc2d = abbe2d_freq
        # self.icc2d = torch.abs(self.icc2d * self.icc2d)
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


    def calPupil(self, FX, FY):
        R = torch.sqrt(FX ** 2 + FY ** 2)
        TH = torch.arctan2(FY, FX)
        H = copy.deepcopy(R)
        H = torch.where(H > 1.0, 0.0, 1.0)
        R[R > 1.0] = 0.0

        W = torch.zeros(R.shape, dtype=torch.complex128)

        for ii in range(len(self.lens.Zn)):
            W = W + zerniken(self.lens.Zn[ii], R, TH) * self.lens.Cn[ii]

        na = self.s.na
        if na < 1:
            W = W + self.lens.defocus / self.s.wavelength * (
                torch.sqrt(1 - (na ** 2) * (R ** 2)) - 1
            )
        elif na >= 1:
            # W = W + self.defocus/self.wavelength*\
            #         (torch.sqrt(self.nLiquid**2-(self.na**2)*(R**2))-self.nLiquid)
            W = W + (na ** 2) / (2 * self.s.wavelength) * self.lens.defocus * (R ** 2)
        self.pupil_fdata = H * torch.exp(-1j * 2 * (torch.pi) * W)

    def calculate(self):
        normalized_period = self.x_gridnum / (self.s.wavelength/self.s.na)
        mask_fm = (torch.arange(self.x1, self.x2) - self.x_gridnum // 2) / normalized_period
        mask_gm = (torch.arange(self.y1, self.y2) - self.y_gridnum // 2) / normalized_period
        mask_gm, mask_fm = torch.meshgrid(mask_fm, mask_gm, indexing='xy')
        # print(mask_gm)
        # mask_fg2m = mask
        mask_fg2m = mask_fm.pow(2) + mask_gm.pow(2)

        self.s.simple_mdata = self.s.simple_mdata[100:]
        self.s.simple_fx = self.s.simple_fx[100:]
        self.s.simple_fy = self.s.simple_fy[100:]

        sourceX = self.s.simple_fx
        sourceY = self.s.simple_fy
        sourceXY2 = sourceX.pow(2) + sourceY.pow(2)

        intensity2D = torch.zeros((
            self.mask.fdata.shape[0],
            self.mask.fdata.shape[1],
            self.s.simple_mdata.shape[0]
            ), dtype=torch.float32)
        # print(sourceX)
        weight = torch.sum(self.s.simple_mdata)
        print("start")
        for i in range(self.s.simple_mdata.shape[0]):
            print(i)
            if i>100:
                break
            rho2 = mask_fg2m + 2 * (sourceX[i] * mask_fm + sourceY[i] * mask_gm) + sourceXY2[i]
            valid_source_mask = rho2.le(1)
            f_calc = torch.masked_select(mask_fm, valid_source_mask)
            g_calc = torch.masked_select(mask_gm, valid_source_mask)
            valid_mask_fdata = torch.masked_select(self.mask.fdata[self.y1 : self.y2, self.x1 : self.x2], valid_source_mask)

            self.calPupil(f_calc, g_calc)
            tempHAber = valid_mask_fdata * self.pupil_fdata

            e_field = torch.zeros(
                self.mask.fdata.shape,
                dtype=torch.complex128)

            ExyzFrequency = torch.zeros(rho2.shape, dtype=torch.complex128)
            ExyzFrequency[valid_source_mask] = tempHAber

            e_field[self.y1 : self.y2, self.x1 : self.x2] = ExyzFrequency
            # Exyz_Partial = torch.fft.ifft2(ExyzFrequency)
            # intensityTemp = intensityTemp + Exyz_Partial.real.pow(2) + Exyz_Partial.imag.pow(2)
            AA = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(e_field)))
            AA = torch.abs(AA * torch.conj(AA))
            intensity2D[:,:, i] = AA
        intensity2D = torch.reshape(self.s.simple_mdata, (1, 1, -1)) * intensity2D
        intensity2D = (torch.sum(intensity2D, dim=2))
        self.intensity2D =  intensity2D / weight

if __name__ == "__main__":
    s = Source()
    s.type = "dipole"
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
    m.maskfft()

    icc = ICC(s, o, m)
    icc.calculate()



    torch_arr_bound(icc.intensity2D, "icc.intensity2D")
    show_img(icc.intensity2D, "icc.intensity2D")
    # torch_arr_bound(icc.icc2d, "icc.icc2d")
    # show_img(icc.icc2d, "icc.icc2d")
    # torch_arr_bound(icc.jsource, "icc.jsource")
    # torch_arr_bound(icc.finalAI, "icc.finalAI")
    # show_img(icc.finalAI, "icc.finalAI")
    ## jsource we have, jsource is real-number matrix.