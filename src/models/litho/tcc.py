"""
"""
import shelve
from pathlib import Path

import numpy as np
import pyfftw
import scipy as sci

from litho.consts import IMAGE_WH, TCC_ORDER
from litho.lens import LensList
from litho.source import Source


class TCC:
    """ """

    def __init__(self, source, lens):
        self.s = source
        self.s.update()
        self.s.ifft()

        self.lens = lens
        self.lens.update()
        self.lens.calPupil()
        self.lens.calPSF()

        self.order = 7
        self.psf = lens.data

    def calMutualIntensity(self):
        self.gnum, self.fnum = self.s.data.shape
        J = np.zeros((self.gnum, self.fnum, self.gnum, self.fnum), dtype=np.complex)
        for ii in range(self.gnum):
            for jj in range(self.fnum):
                J[:, :, ii, jj] = self.s.spatMutualData.real[
                    (self.gnum - ii - 1) : (2 * self.gnum - ii - 1),
                    (self.fnum - jj - 1) : (2 * self.fnum - jj - 1),
                ]
        self.jsource = np.reshape(J, (self.gnum * self.fnum, self.gnum * self.fnum))

    def calSpatTCC(self):
        H = np.reshape(self.psf, (self.psf.size, 1))
        self.tcc2d = (
            self.jsource * np.dot(H, H.transpose()) / self.s.detaf / self.s.detag
        )

    def svd(self):
        self.spat_part = pyfftw.empty_aligned(
            (self.gnum, self.fnum, self.gnum, self.fnum), dtype="complex128"
        )
        self.freq_part = pyfftw.empty_aligned(
            (self.gnum, self.fnum, self.gnum, self.fnum), dtype="complex128"
        )
        self.fft_svd = pyfftw.FFTW(self.spat_part, self.freq_part, axes=(0, 1, 2, 3))

        tcc4d = self.tcc2d.reshape((self.gnum, self.fnum, self.gnum, self.fnum))
        self.spat_part[:] = np.fft.ifftshift(tcc4d)
        self.fft_svd()
        tcc4df = np.fft.fftshift(self.freq_part)
        # tcc4df = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(tcc4d)))

        tcc2df = tcc4df.reshape((self.gnum * self.fnum, self.gnum * self.fnum))

        # U,S,V = np.linalg.svd(tcc2df)
        # arr_bound(tcc2df, 'tcc2df')
        U, S, V = sci.sparse.linalg.svds(tcc2df, self.order)  # faster than svd
        self.coefs = S[0 : self.order]
        self.kernels = np.zeros((self.gnum, self.fnum, self.order), dtype=np.complex)
        for ii in range(self.order):
            self.kernels[:, :, ii] = np.reshape(U[:, ii], (self.gnum, self.fnum))


class TCCList(TCC):
    def __init__(self, source, lensList):
        self.s = source
        self.PSFList = lensList.sDataList
        self.order = 7
        self.focusList = lensList.focusList
        self.focusCoef = lensList.focusCoef
        self.kernelList = []
        self.coefList = []

    def calculate(self):
        self.calMutualIntensity()
        for ii in self.PSFList:
            self.psf = ii
            self.calSpatTCC()
            self.svd()
            self.coefList.append(self.coefs)
            self.kernelList.append(self.kernels)


class TCC_db:
    def __init__(self):
        self.s = None
        self.PSFList = []
        self.order = None
        self.focusList = []
        self.focusCoef = []
        self.kernelList = []
        self.coefList = []

    def save_db(self, tcclist, db_path):
        self.s = tcclist.s
        self.PSFList = tcclist.PSFList
        self.order = tcclist.order
        self.focusList = tcclist.focusList
        self.focusCoef = tcclist.focusCoef
        self.kernelList = tcclist.kernelList
        self.coefList = tcclist.coefList

        db = shelve.open(f"{db_path}")
        db["TCCList"] = self
        db.close()
        print(f"tcc list saved to {db_path}")

    def load_db(self, db_path):
        db = shelve.open(f"{db_path}")
        tcclist = db["TCCList"]
        self.s = tcclist.s
        self.PSFList = tcclist.PSFList
        self.order = tcclist.order
        self.focusList = tcclist.focusList
        self.focusCoef = tcclist.focusCoef
        self.kernelList = tcclist.kernelList
        self.coefList = tcclist.coefList
        print(f"Load tcc list from {db_path}")


def get_tcc(db_path: str = None):
    db_ext = [".db", ".dat"]
    db_path_ext = [Path(db_path + item) for item in db_ext]

    # if have db, load and return
    for p in db_path_ext:
        if p.exists():
            tdb = TCC_db()
            tdb.load_db(db_path)
            print(f"load tcc from {db_path}")
            return tdb

    # else, calculate tcc and save to db.
    # first create folder.
    # if not Path(db_path).parent.exists():
    Path(db_path).parent.mkdir(exist_ok=True)
    s = Source()
    s.na = 1.35
    # s.maskxpitch = m.x_range[1] - m.x_range[0]
    # s.maskypitch = m.y_range[1] - m.y_range[0]
    s.maskxpitch = IMAGE_WH
    s.maskypitch = IMAGE_WH
    s.type = "annular"
    s.sigma_in = 0.6
    s.sigma_out = 0.9
    s.smooth_deta = 0.00
    s.shiftAngle = 0
    s.update()
    s.ifft()

    o = LensList()
    o.na = s.na
    o.maskxpitch = s.maskxpitch
    o.maskypitch = s.maskypitch
    o.focusList = [0]
    o.focusCoef = [1]
    o.calculate()

    print("Calculating TCC and SVD kernels")
    t = TCCList(s, o)
    t.order = TCC_ORDER
    t.calculate()
    print("calculate tcc directly")
    # then save it
    tdb = TCC_db()
    tdb.save_db(t, db_path)
    return t


if __name__ == "__main__":
    # s = Source()
    # s.type = "annular"
    # s.sigma_in = 0.5
    # s.sigma_out = 0.8
    # s.na = 1.35
    # s.update()
    # s.ifft()

    # o = LensList()
    # o.maskxpitch = s.maskxpitch
    # o.maskypitch = s.maskypitch
    # o.na = s.na
    # o.focusList = [-50, 0, 50]
    # o.focusCoef = [0.5, 1, 0.5]
    # o.calculate()

    # tcc = TCCList(s, o)
    # tcc.calculate()
    tcc_db = get_tcc("./litho/db/tcclist")
    # arr_bound(tcc_db.kernelList[0], "kernel_list")
    # arr_bound(tcc_db.kernelList[0][0], "kernel_list[0]")
    # arr_bound(tcc_db.coefList[0], "coef_list")
    # tccdb = TCC_db()
    # # tccdb.save_db(tcc, './litho/db/tcclist')
    # tccdb.load_db("./litho/db/tcclist")
    # # print(tccdb.s)
    # # print(tccdb.PSFList)
    # print(tccdb.order)
    # cl = tccdb.coefList[0][::-1]
    # for i in range(len(cl)):
    #     print(f"{cl[i]},")
