"""
NOTE: This is for Scalar Pupil Assumption, not Jones Pupil
LensList is a Lens container, used for, e.g., robust mask synthesis
"""

import copy

import torch

from src.models.litho.zernike import zerniken


class Lens:
    """Model lens
    Args:
        na : 1.35
        nLiquid : 1.414
        wavelength (nm) : 193.0
        defocus (nm) : 0.0
        maskxpitch (nm): 1000
        maskypitch (nm): 1000

    """

    def __init__(
        self,
        na: float = 1.35,
        nLiquid: float = 1.414,
        wavelength: float = 193.0,
        defocus: float = 0.0,
        maskxpitch: float = 1000,
        maskypitch: float = 1000,
    ):
        self.na = na
        self.nLiquid = nLiquid
        self.wavelength = wavelength
        self.defocus = defocus
        self.maskxpitch = maskxpitch
        self.maskypitch = maskypitch
        self.Zn = [9]
        self.Cn = [0.0]

    def update(self):
        self.detaf = self.wavelength / (self.maskxpitch * self.na)
        self.detag = self.wavelength / (self.maskypitch * self.na)
        self.fnum = int(torch.ceil(torch.tensor(2 / self.detaf)))
        self.gnum = int(torch.ceil(torch.tensor(2 / self.detag)))

    def calPupil(self, shiftx=0, shifty=0):
        fx = torch.linspace(-self.fnum * self.detaf, self.fnum * self.detaf, 2 * self.fnum + 1)
        fy = torch.linspace(-self.gnum * self.detag, self.gnum * self.detag, 2 * self.gnum + 1)
        FX, FY = torch.meshgrid(fx - shiftx, fy - shifty, indexing="xy")

        R = torch.sqrt(FX**2 + FY**2)
        TH = torch.arctan2(FY, FX)
        H = copy.deepcopy(R)
        H = torch.where(H > 1.0, 0.0, 1.0)
        R[R > 1.0] = 0.0

        W = torch.zeros((2 * self.gnum + 1, 2 * self.fnum + 1), dtype=torch.complex128)

        for ii in range(len(self.Zn)):
            W = W + zerniken(self.Zn[ii], R, TH) * self.Cn[ii]

        if self.na < 1:
            W = W + self.defocus / self.wavelength * (
                torch.sqrt(1 - (self.na**2) * (R**2)) - 1
            )
        elif self.na >= 1:
            # W = W + self.defocus/self.wavelength*\
            #         (torch.sqrt(self.nLiquid**2-(self.na**2)*(R**2))-self.nLiquid)
            W = W + (self.na**2) / (2 * self.wavelength) * self.defocus * (R**2)
        self.fdata = H * torch.exp(-1j * 2 * (torch.pi) * W)

    def calPSF(self):
        normlize = 1  # self.detaf * self.detag
        self.data = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(self.fdata))) * normlize


class LensList(Lens):
    """List of lens."""

    def __init__(
        self,
        na: float = 1.35,
        nLiquid: float = 1.414,
        wavelength: float = 193.0,
        defocus: float = 0.0,
        maskxpitch: float = 1000,
        maskypitch: float = 1000,
    ):
        super().__init__(na, nLiquid, wavelength, defocus, maskxpitch, maskypitch)
        self.focusList = [0.0]
        self.focusCoef = [1.0]
        self.fDataList = []
        self.sDataList = []

        """
        Process calculation
        """
        self.calculate()

    def calculate(self):
        self.update()
        for ii in self.focusList:
            self.defocus = ii
            self.calPupil()
            self.fDataList.append(self.fdata)
            self.calPSF()
            self.sDataList.append(self.data)


if __name__ == "__main__":
    lens_list = LensList()
    lens_list.na = 0.85
    lens_list.focusList = [-50, 0, 50]
    lens_list.focusCoef = [0.5, 1, 0.5]
    lens_list.calculate()

    lens = Lens()
    lens.na = 0.85
    lens.update()
    lens.calPupil()
    lens.calPSF()
