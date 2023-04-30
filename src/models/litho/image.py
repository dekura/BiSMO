import torch
import torch.nn.functional as F

from src.models.litho.gds_mask import Mask
from src.models.litho.tcc import TCCList


class ImageHopkins:
    """ImageHopkinsList is a container, used for, e.g., robust mask synthesis This is for Scalar
    Assumption, not Vector."""

    def __init__(self, mask, tcc):
        self.resist_a = 80  # Resist model parameter: sharpness
        self.resist_t = 0.6  # Resist model parameter: threshold
        self.tcc = tcc  # TCC
        self.mask = mask  # Mask
        self.order = tcc.order  # TCC Order
        self.kernels = tcc.kernels  # Kernels
        self.coefs = tcc.coefs  # Coefs

        self.norm = self.mask.y_gridnum * self.mask.x_gridnum
        self.x1 = int(self.mask.x_gridnum // 2 - self.tcc.s.fnum)
        self.x2 = int(self.mask.x_gridnum // 2 + self.tcc.s.fnum + 1)
        self.y1 = int(self.mask.y_gridnum // 2 - self.tcc.s.gnum)
        self.y2 = int(self.mask.y_gridnum // 2 + self.tcc.s.gnum + 1)

        self.spat_part = torch.zeros(
            (self.mask.y_gridnum, self.mask.x_gridnum), dtype=torch.complex64
        )
        self.freq_part = torch.zeros(
            (self.mask.y_gridnum, self.mask.x_gridnum), dtype=torch.complex64
        )

    def calAIold(self):  # much faster than calAIold(), however some bugs here.
        AI_freq_dense = torch.zeros(
            (self.mask.y_gridnum, self.mask.x_gridnum), dtype=torch.complex64
        )
        AI_freq_sparse = torch.zeros(
            (int(self.y2 - self.y1), int(self.x2 - self.x1)), dtype=torch.complex64
        )
        for ii in range(self.order):
            self.x1 = int(self.x1)
            self.x2 = int(self.x2)
            self.y1 = int(self.y1)
            self.y2 = int(self.y2)
            e_field = (
                self.kernels[:, :, ii] * self.mask.fdata[self.y1 : self.y2, self.x1 : self.x2]
            )
            e_field_conj = (
                torch.conj(torch.rot90(self.kernels[:, :, ii], 2))
                * self.mask.fdata[self.y1 : self.y2, self.x1 : self.x2]
            )
            AA = F.conv2d(
                e_field.view(1, *e_field.shape),
                e_field_conj.view(1, 1, *e_field_conj.shape),
                padding="same",
            )
            AA = AA.squeeze()
            AI_freq_sparse += self.coefs[ii] * AA
        AI_freq_dense[self.y1 : self.y2, self.x1 : self.x2] = AI_freq_sparse

        self.freq_part[:] = torch.fft.ifftshift(AI_freq_dense)
        self.spat_part = torch.fft.ifft2(self.freq_part)
        self.AI = torch.real(torch.fft.fftshift(self.spat_part)) / self.norm

    def calAI(self):
        AI = torch.zeros((self.mask.y_gridnum, self.mask.x_gridnum))
        for ii in range(self.order):
            e_field = torch.zeros(
                (self.mask.y_gridnum, self.mask.x_gridnum), dtype=torch.complex64
            )
            e_field[self.y1 : self.y2, self.x1 : self.x2] = (
                self.kernels[:, :, ii] * self.mask.fdata[self.y1 : self.y2, self.x1 : self.x2]
            )
            AA = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(e_field)))
            AI += self.coefs[ii] * torch.abs(AA * torch.conj(AA))
        self.AI = AI / self.order

    # def calRI(self):
    # self.RI = 1 / (1 + torch.exp(-self.resist_a * (self.AI - self.resist_t)))

    def calRI(self):
        self.RI = (self.AI >= self.resist_t).to(torch.float32)


class ImageHopkinsList(ImageHopkins):
    def __init__(self, mask: Mask, tccList: TCCList):
        self.resist_a = 80
        self.resist_t = 0.6
        self.resist_tRef = 0.12
        self.doseList = [1.0]
        self.doseCoef = [1.0]
        self.mask = mask
        self.tcc = tccList
        self.order = tccList.order
        self.kernelList = tccList.kernelList
        self.coefList = tccList.coefList
        self.focusList = tccList.focusList
        self.focusCoef = tccList.focusCoef
        self.AIList = []
        self.RIList = []

        self.norm = self.mask.y_gridnum * self.mask.x_gridnum
        self.x1 = self.mask.x_gridnum // 2 - self.tcc.s.fnum
        self.x2 = self.mask.x_gridnum // 2 + self.tcc.s.fnum + 1
        self.y1 = self.mask.y_gridnum // 2 - self.tcc.s.gnum
        self.y2 = self.mask.y_gridnum // 2 + self.tcc.s.gnum + 1

        self.spat_part = torch.zeros(
            (self.mask.y_gridnum, self.mask.x_gridnum), dtype=torch.complex64
        )
        self.freq_part = torch.zeros(
            (self.mask.y_gridnum, self.mask.x_gridnum), dtype=torch.complex64
        )

    def calculate(self):
        length = len(self.focusList)
        for ii in range(length):
            self.kernels = self.kernelList[ii]
            self.coefs = self.coefList[ii]
            self.calAI()
            self.AIList.append(self.AI)
            self.RIList.append([])
            for jj in self.doseList:
                self.resist_t = self.resist_tRef * jj
                # print("resist: ", self.resist_t)
                self.calRI()
                self.RIList[ii].append(self.RI)


if __name__ == "__main__":
    from source import Source

    from src.models.litho.gds_mask import Mask

    mp = [
        [
            [-1, 6],
            [-1, 2],
            [1, 2],
            [1, 1],
            [6, 1],
            [6, 0],
            [0, 0],
            [0, 1],
            [-2, 1],
            [-2, 6],
            [-1, 6],
        ],
        [
            [6, -1],
            [6, -2],
            [1, -2],
            [1, -3],
            [4, -3],
            [4, -6],
            [3, -6],
            [3, -4],
            [0, -4],
            [0, -1],
            [6, -1],
        ],
    ]
    m = Mask("AND2_X4.gds", 10)
    m.x_range = [-300.0, 300.0]
    m.y_range = [-400.0, 300.0]
    m.x_gridsize = 1.0
    m.y_gridsize = 1.0
    m.CD = 40
    m.polygons = mp
    m.poly2mask()
    m.smooth()
    m.maskfft()

    """nominal ILT setting"""
    # s = Source()
    # s.na = 1.35
    # s.maskxpitch = 600.0
    # s.maskypitch = 800.0
    # s.type = "annular"
    # s.sigma_in = 0.6
    # s.sigma_out = 0.8
    # s.update()
    # s.ifft()

    # o = Lens()
    # o.na = s.na
    # o.maskxpitch = s.maskxpitch
    # o.maskypitch = s.maskypitch
    # o.update()
    # o.calPupil()
    # o.calPSF()

    # t = TCC(s, o)
    # t.calMutualIntensity()
    # t.calSpatTCC()
    # t.svd()

    # i = ImageHopkins(m, t)
    # i.calAI()

    """robust ILT setting"""
    from lens import LensList
    from tcc import TCCList

    s = Source()
    s.na = 1.25
    s.maskxpitch = 600.0
    s.maskypitch = 1000
    s.type = "annular"
    s.sigma_in = 0.5
    s.sigma_out = 0.8
    s.update()
    s.ifft()

    o = LensList()
    o.na = s.na
    o.maskxpitch = s.maskxpitch
    o.maskypitch = s.maskypitch
    o.focusList = [-50, 0, 50]
    o.focusCoef = [0.5, 1, 0.5]
    o.calculate()

    t = TCCList(s, o)
    t.calculate()

    i = ImageHopkinsList(m, t)
    i.doseList = [0.95, 1, 1.05]
    i.doseCoef = [0.5, 1, 0.5]
    i.calculate()
