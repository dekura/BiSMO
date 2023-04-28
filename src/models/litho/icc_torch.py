"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-04-12 15:29:06
LastEditTime: 2023-04-28 03:43:51
Contact: cgjcuhk@gmail.com
Description:
"""
# from src.models.litho.lens import LensList
# from src.models.litho.source import Source


import copy
import torch
from source_torch import Source
from mask_torch import Mask
from utils import torch_arr_bound, show_img
class ICC:
    """
    The illumination-cross coefficient.
    """

    def __init__(self, source, mask):
        self.s = source
        self.s.update()
        # self.s.ifft()
        print(f"s.data.shape: {self.s.data.shape}")
        self.s.simple_source()

        self.mask = mask
        self.mask.maskfft()

        # self.psf = lens.data


        self.gnum = self.s.gnum
        self.fnum = self.s.fnum

        self.x1 = int(self.mask.x_gridnum // 2 - self.fnum)
        self.x2 = int(self.mask.x_gridnum // 2 + self.fnum + 1)
        self.y1 = int(self.mask.y_gridnum // 2 - self.gnum)
        self.y2 = int(self.mask.y_gridnum // 2 + self.gnum + 1)

        self.x_gridnum = self.mask.x_gridnum
        self.y_gridnum = self.mask.y_gridnum


    def calPupil(self, FX, FY):
        R = torch.sqrt(FX ** 2 + FY ** 2)
        # TH = torch.arctan2(FY, FX)
        # H = copy.deepcopy(R)
        H = R.detach().clone()
        H = torch.where(H > 1.0, 0.0, 1.0)
        R[R > 1.0] = 0.0

        W = torch.zeros(R.shape, dtype=torch.complex64)

        # for ii in range(len(self.lens.Zn)):
        #     W = W + zerniken(self.lens.Zn[ii], R, TH) * self.lens.Cn[ii]

        # na = self.s.na
        # if na < 1:
        #     W = W + self.lens.defocus / self.s.wavelength * (
        #         torch.sqrt(1 - (na ** 2) * (R ** 2)) - 1
        #     )
        # elif na >= 1:
        #     # W = W + self.defocus/self.wavelength*\
        #     #         (torch.sqrt(self.nLiquid**2-(self.na**2)*(R**2))-self.nLiquid)
        #     W = W + (na ** 2) / (2 * self.s.wavelength) * \
        #         self.lens.defocus * (R ** 2)
        #     # print(f"W: {W}")
        self.pupil_fdata = H * torch.exp(-1j * 2 * (torch.pi) * W)
        # torch_arr_bound(self.pupil_fdata, "self.pupil_fdata")

    def calculate(self):
        normalized_period = self.x_gridnum / (self.s.wavelength/self.s.na)
        mask_fm = (torch.arange(self.x1, self.x2) -
                   self.x_gridnum // 2) / normalized_period
        mask_gm = (torch.arange(self.y1, self.y2) -
                   self.y_gridnum // 2) / normalized_period
        mask_fm, mask_gm = torch.meshgrid(mask_fm, mask_gm, indexing='xy')

        mask_fg2m = mask_fm.pow(2) + mask_gm.pow(2)

        total_source = self.s.simple_mdata.shape[0]

        # total_source = 20
        print(f"totol source number : {total_source}")
        self.s.simple_mdata = self.s.simple_mdata[:total_source]
        self.s.simple_fx = self.s.simple_fx[:total_source]
        self.s.simple_fy = self.s.simple_fy[:total_source]

        sourceX = self.s.simple_fx
        sourceY = self.s.simple_fy
        sourceXY2 = sourceX.pow(2) + sourceY.pow(2)

        # too large to direct use
        # intensity2D = torch.zeros((
        #     self.mask.fdata.shape[0],
        #     self.mask.fdata.shape[1],
        #     self.s.simple_mdata.shape[0]
        # ), dtype=torch.float32)

        weight = torch.sum(self.s.simple_mdata)
        intensity2D = torch.zeros(self.mask.fdata.shape, dtype=torch.float32)
        for i in range(self.s.simple_mdata.shape[0]):
            # print(i)
            # torch_arr_bound(mask_fm, f"mask_fm[{i}]")
            rho2 = mask_fg2m + 2 * \
                (sourceX[i] * mask_fm + sourceY[i] * mask_gm) + sourceXY2[i]
            valid_source_mask = rho2.le(1)

            # torch_arr_bound(valid_source_mask, f"valid_source_mask {i}")
            f_calc = torch.masked_select(mask_fm, valid_source_mask)
            # torch_arr_bound(f_calc, f"f_calc[{i}]")
            g_calc = torch.masked_select(mask_gm, valid_source_mask)
            # torch_arr_bound(g_calc, f"g_calc[{i}]")
            valid_mask_fdata = torch.masked_select(
                self.mask.fdata[self.y1: self.y2, self.x1: self.x2], valid_source_mask)

            self.calPupil(f_calc, g_calc)

            # torch_arr_bound(self.pupil_fdata, f"self.pupil fdata on source[{i}]")
            tempHAber = valid_mask_fdata * self.pupil_fdata

            e_field = torch.zeros(
                self.mask.fdata.shape,
                dtype=torch.complex64)

            ExyzFrequency = torch.zeros(rho2.shape, dtype=torch.complex64)
            ExyzFrequency[valid_source_mask] = tempHAber

            e_field[self.y1: self.y2, self.x1: self.x2] = ExyzFrequency
            AA = torch.fft.fftshift(
                torch.fft.ifft2(torch.fft.ifftshift(e_field)))
            AA = torch.abs(AA * torch.conj(AA))
            AA = self.s.simple_mdata[i] * AA
            intensity2D += AA

        self.intensity2D = intensity2D / weight
        self.RI = torch.where(self.intensity2D >= 0.225, 1, 0)

"""
TODO:
1. Update Source and pupil pitch. (can be different from the mask)
2. Read layout from image.
3. Implement the SMO algorithm.
"""
if __name__ == "__main__":
    gds_max_x = 1000
    gds_max_y = 1000
    MASK_W = 1000
    MASK_H = 1000

    source_w = 1280
    source_h = 1280

    s = Source()
    # s.type = "coventional"
    s.type = "annular"
    # s.type = "quasar"
    # s.type = "dipole"
    s.na = 1.35
    s.maskxpitch = source_w
    s.maskypitch = source_h
    s.sigma_out = 0.7
    s.sigma_in = 0.5
    s.smooth_deta = 0
    s.shiftAngle = 0
    # s.update()
    # s.ifft()
    # print(s.data.size())

    # o = LensList()
    # o.maskxpitch = source_w
    # o.maskypitch = source_h
    # o.na = 1.35
    # o.focusList = [0.0]
    # o.focusCoef = [1.0]
    # # o.calculate()

    gds_path = "/home/gjchen21/phd/projects/smo/SMO-ICCAD23/data/NanGateLibGDS/NOR2_X2.gds"
    # img_path = "/home/gjchen21/phd/projects/smo/SMO-ICCAD23-torch/data/ibm_opc_test/mask/t1_0_mask.png"
    m = Mask(layout_path=gds_path, layername=11, xmax=gds_max_x,
            ymax=gds_max_y, maskxpitch=MASK_W, maskypitch=MASK_H)
    # m.x_range = [ ]
    # m.open_img()
    m.openGDS()
    m.maskfft()

    icc = ICC(s, m)
    icc.calculate()

    torch_arr_bound(icc.intensity2D, "icc.intensity2D")
    # show_img(icc.intensity2D, "icc.intensity2D")
    show_img(m.data, "m.data")
    show_img(icc.RI, "icc.RI")
    # torch_arr_bound(icc.icc2d, "icc.icc2d")
    # show_img(icc.icc2d, "icc.icc2d")
    # torch_arr_bound(icc.jsource, "icc.jsource")
    # torch_arr_bound(icc.finalAI, "icc.finalAI")
    # show_img(icc.finalAI, "icc.finalAI")
    # jsource we have, jsource is real-number matrix.

