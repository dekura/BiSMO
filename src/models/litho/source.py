"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2023-03-29 15:45:14
LastEditTime: 2023-04-02 14:41:33
Contact: cgjcuhk@gmail.com


Description:

The real value for source.spatMutualData is used.
"""

import math
import torch
from torch.special import erf

def Edeta(deta, x):
    if deta != 0:
        g = 0.5 * (1 + erf(x / deta))
        return g
    else:
        g = torch.zeros(x.shape, dtype=torch.float64)
        g[x >= 0] = 1
        return g


class Source:
    """
    Source.data is used for Abbe fomulation
    Source.mdata is used for Hopkins fomulation, Mutual Intensity, TCC calculation

    """

    def __init__(
        self,
        na: float = 1.35,
        wavelength: float = 193.0,
        maskxpitch: float = 2000.0,
        maskypitch: float = 2000.0,
        sigma_out:  float = 0.8,
        sigma_in:   float = 0.6,
        smooth_deta: float = 0.03,
        source_type: str = "annular",
        shiftAngle: float =  math.pi / 4,
        openAngle:  float = math.pi / 16,
    ):
        self.na = na
        self.wavelength = wavelength
        self.maskxpitch = maskxpitch
        self.maskypitch = maskypitch

        self.sigma_out = sigma_out
        self.sigma_in = sigma_in
        self.smooth_deta = smooth_deta
        self.shiftAngle = shiftAngle
        self.openAngle = openAngle
        self.type = source_type

    def update(self):
        self.detaf = self.wavelength / (self.maskxpitch * self.na)
        self.detag = self.wavelength / (self.maskypitch * self.na)
        self.fnum = int(torch.ceil(torch.tensor(2 / self.detaf, dtype=torch.float64)))
        self.gnum = int(torch.ceil(torch.tensor(2 / self.detag, dtype=torch.float64)))

        fx = torch.linspace(
            -self.fnum * self.detaf, self.fnum * self.detaf, 2 * self.fnum + 1
        )
        fy = torch.linspace(
            -self.gnum * self.detag, self.gnum * self.detag, 2 * self.gnum + 1
        )
        FX, FY = torch.meshgrid(fx, fy, indexing="xy")

        r = torch.sqrt(FX ** 2 + FY ** 2)
        theta = torch.arctan2(FY, FX)
        theta[r > 1] = 0
        r[r > 1] = 0
        s0 = torch.sqrt(FX ** 2 + FY ** 2)
        s0[s0 <= 1] = 1
        s0[s0 > 1] = 0

        self.r = r
        self.s0 = s0
        self.theta = theta
        self.fx = FX
        self.fy = FY

        if self.type == "conventional":
            s = Edeta(self.smooth_deta, self.sigma_out - self.r) * self.s0
            self.data = s
        elif self.type == "annular":
            s = (
                Edeta(self.smooth_deta, self.sigma_out - self.r)
                * Edeta(self.smooth_deta, self.r - self.sigma_in)
                * self.s0
            )
            self.data = s
        elif self.type == "quasar":
            s = (
                Edeta(self.smooth_deta, self.sigma_out - self.r)
                * Edeta(self.smooth_deta, self.r - self.sigma_in)
                * (
                    Edeta(
                        self.smooth_deta,
                        self.openAngle
                        - torch.abs(1.0 * math.pi -
                                 torch.abs(self.shiftAngle - self.theta)),
                    )
                    + Edeta(
                        self.smooth_deta,
                        self.openAngle
                        - torch.abs(0.5 * math.pi -
                                 torch.abs(self.shiftAngle - self.theta)),
                    )
                    + Edeta(
                        self.smooth_deta,
                        self.openAngle
                        - torch.abs(-0.5 * math.pi -
                                 torch.abs(self.shiftAngle - self.theta)),
                    )
                    + Edeta(
                        self.smooth_deta,
                        self.openAngle
                        - torch.abs(-0.0 * math.pi -
                                 torch.abs(self.shiftAngle - self.theta)),
                    )
                )
                * self.s0
            )
            self.data = s
        elif self.type == "dipole":
            s = (
                Edeta(self.smooth_deta, self.sigma_out - self.r)
                * Edeta(self.smooth_deta, self.r - self.sigma_in)
                * (
                    Edeta(
                        self.smooth_deta,
                        self.openAngle - torch.abs(self.shiftAngle - self.theta),
                    )
                    + Edeta(
                        self.smooth_deta,
                        self.openAngle
                        - torch.abs(math.pi - torch.abs(self.shiftAngle - self.theta)),
                    )
                )
                * self.s0
            )
            self.data = s
        else:
            s = (
                Edeta(self.smooth_deta, self.sigma_out - self.r)
                * Edeta(self.smooth_deta, self.r - self.sigma_in)
                * self.s0
            )
            self.data = s

    def ifft(self):
        fx = torch.linspace(
            -self.fnum * self.detaf, self.fnum * self.detaf, 4 * self.fnum + 1
        )
        fy = torch.linspace(
            -self.gnum * self.detag, self.gnum * self.detag, 4 * self.gnum + 1
        )
        FX, FY = torch.meshgrid(fx, fy, indexing="xy")

        r = torch.sqrt(FX ** 2 + FY ** 2)
        theta = torch.arctan2(FY, FX)
        theta[r > 1] = 0
        r[r > 1] = 0
        s0 = torch.sqrt(FX ** 2 + FY ** 2)
        s0 = torch.where(s0 > 1.0, 0.0, 1.0)

        self.r = r
        self.s0 = s0
        self.theta = theta
        self.fx = FX
        self.fy = FY

        if self.type == "conventional":
            s = Edeta(self.smooth_deta, self.sigma_out - self.r) * self.s0
            self.mdata = s
        elif self.type == "annular":
            s = (
                Edeta(self.smooth_deta, self.sigma_out - self.r)
                * Edeta(self.smooth_deta, self.r - self.sigma_in)
                * self.s0
            )
            self.mdata = s
        elif self.type == "quasar":
            s = (
                Edeta(self.smooth_deta, self.sigma_out - self.r)
                * Edeta(self.smooth_deta, self.r - self.sigma_in)
                * (
                    Edeta(
                        self.smooth_deta,
                        self.openAngle
                        - torch.abs(1.0 * math.pi -
                                torch.abs(self.shiftAngle - self.theta)),
                    )
                    + Edeta(
                        self.smooth_deta,
                        self.openAngle
                        - torch.abs(0.5 * math.pi -
                                torch.abs(self.shiftAngle - self.theta)),
                    )
                    + Edeta(
                        self.smooth_deta,
                        self.openAngle
                        - torch.abs(-0.5 * math.pi -
                                torch.abs(self.shiftAngle - self.theta)),
                    )
                    + Edeta(
                        self.smooth_deta,
                        self.openAngle
                        - torch.abs(-0.0 * math.pi -
                                torch.abs(self.shiftAngle - self.theta)),
                    )
                )
                * self.s0
            )
            self.mdata = s
        elif self.type == "dipole":
            s = (
                Edeta(self.smooth_deta, self.sigma_out - self.r)
                * Edeta(self.smooth_deta, self.r - self.sigma_in)
                * (
                    Edeta(
                        self.smooth_deta,
                        self.openAngle - torch.abs(self.shiftAngle - self.theta),
                    )
                    + Edeta(
                        self.smooth_deta,
                        self.openAngle
                        - torch.abs(math.pi - torch.abs(self.shiftAngle - self.theta)),
                    )
                )
                * self.s0
            )
            self.mdata = s
        else:
            s = (
                Edeta(self.smooth_deta, self.sigma_out - self.r)
                * Edeta(self.smooth_deta, self.r - self.sigma_in)
                * self.s0
            )
            self.mdata = s
        normlize = 1  # self.detaf * self.detag
        self.spatMutualData = (
            torch.fft.fftshift(torch.fft.ifft2(
                torch.fft.ifftshift(self.mdata))) * normlize
        )


if __name__ == "__main__":
    s = Source()
    s.type = "annular"
    s.sigma_in = 0.6
    s.sigma_out = 0.8
    s.smooth_deta = 0
    s.update()
    s.ifft()

