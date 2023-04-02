import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
from PIL import Image, ImageDraw
import torch.nn.functional as F
from pathlib import Path
# from src.models.litho.gdsii.library import Library
from gdsii.library import Library
from utils import torch_arr_bound

class Mask:
    """

    Binary Mask

    Args:
        x/ymax: for the computing area
        x/y_gridsize: the simulated size of the area. Different value are supported. 2nm
        CD: used for method poly2mask, 45nm

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt

        from litho.config import PATH
        from litho.mask import Mask

        m = Mask()
        m.x_range = [-300.0, 300.0]
        m.y_range = [-300.0, 300.0]
        m.x_gridsize = 10
        m.y_gridsize = 10
        m.openGDS(PATH.gdsdir / "AND2_X4.gds", 10)
        m.maskfft()
        m.smooth()

        plt.imshow(
            m.data,
            extent=(m.x_range[0], m.x_range[1], m.y_range[0], m.y_range[1]),
            cmap="hot",
            interpolation="none",
        )
        plt.figure()
        plt.imshow(
            m.sdata,
            extent=(m.x_range[0], m.x_range[1], m.y_range[0], m.y_range[1]),
            cmap="hot",
            interpolation="none",
        )
        plt.show()

    """

    def __init__(
            self,
            gds_path: str,
            layername: int,
            boundary: float = 0.16,
            pixels_per_um: int = 10,
            xmax=500,
            ymax=500,
            x_gridsize=1,
            y_gridsize=1,
            CD=45,
            ):
        self.x_range = [-xmax, xmax]  # nm
        self.y_range = [-ymax, ymax]
        self.x_gridsize = x_gridsize  # nm
        self.y_gridsize = y_gridsize
        self.CD = CD
        self.gds_path = gds_path
        self.layername = layername
        self.boundary = boundary
        self.pixels_per_um = pixels_per_um

    def poly2mask(self):
        """Get Pixel-based Mask Image from Polygon Data
        The Poylgon Data Form are sensitive
        Similar to poly2mask in Matlab
        """
        self.x_gridnum = int((self.x_range[1] - self.x_range[0]) / self.x_gridsize)
        self.y_gridnum = int((self.y_range[1] - self.y_range[0]) / self.y_gridsize)
        img = Image.new("L", (self.x_gridnum, self.y_gridnum), 0)

        self.perimeter = 0.0
        for ii in self.polygons:
            pp = np.array(ii) * self.CD  # polygon
            polygonlen = len(pp)
            self.perimeter += np.sum(np.abs(pp[0:-1] - pp[1:polygonlen]))
            pp[:, 0] = (pp[:, 0] - self.x_range[0]) / self.x_gridsize
            pp[:, 1] = (pp[:, 1] - self.y_range[0]) / self.y_gridsize
            vetex_list = list(pp)
            polygon = [tuple(y) for y in vetex_list]
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)

        self.data = torch.from_numpy(np.array(img))
        # self.data = np.float64(self.data)
        self.spat_part = torch.zeros((self.y_gridnum, self.x_gridnum), dtype=torch.complex128)

        self.freq_part = torch.zeros((self.y_gridnum, self.x_gridnum), dtype=torch.complex128)



    #TODO: how to handle the gds path
    def openGDS(self):
        gdsdir = self.gds_path
        layername = self.layername
        boundary = self.boundary
        pixels_per_um = self.pixels_per_um
        with open(gdsdir, "rb") as stream:
            lib = Library.load(stream)

        a = lib.pop(0)
        b = []
        xmin = []
        xmax = []
        ymin = []
        ymax = []
        for ii in range(0, len(a)):
            if a[ii].layer == layername:
                # if hasattr(a[ii],'data_type'):
                if len(a[ii].xy) > 1:
                    aa = np.array(a[ii].xy) / 1000 * pixels_per_um
                    b.append(aa)
                    xmin.append(min([k for k, v in aa]))
                    xmax.append(max([k for k, v in aa]))
                    ymin.append(min([v for k, v in aa]))
                    ymax.append(max([v for k, v in aa]))
        self.polylist = b

        xmin = min(xmin)
        xmax = max(xmax)
        ymin = min(ymin)
        ymax = max(ymax)
        self.xmin = xmin - boundary * (xmax - xmin)
        self.xmax = xmax + boundary * (xmax - xmin)
        self.ymin = ymin - boundary * (ymax - ymin)
        self.ymax = ymax + boundary * (ymax - ymin)
        self.x_range = [self.xmin, self.xmax]
        self.y_range = [self.ymin, self.ymax]

        self.x_gridnum = int((self.xmax - self.xmin) / self.x_gridsize)
        self.y_gridnum = int((self.ymax - self.ymin) / self.y_gridsize)
        img = Image.new("L", (self.x_gridnum, self.y_gridnum), 0)

        self.perimeter = 0.0
        for ii in self.polylist:
            pp = np.array(ii)  # polygon
            polygonlen = len(pp)
            self.perimeter += np.sum(np.abs(pp[0:-1] - pp[1:polygonlen]))

            pp[:, 0] = (pp[:, 0] - self.xmin) / self.x_gridsize
            pp[:, 1] = (pp[:, 1] - self.ymin) / self.y_gridsize
            vetex_list = list(pp)
            polygon = [tuple(y) for y in vetex_list]
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)

            self.perimeter += np.sum(np.abs(pp[0:-1] - pp[1:polygonlen]))

        self.data = torch.from_numpy(np.array(img))

        # Fourier transform pair
        self.spat_part = torch.zeros((self.y_gridnum, self.x_gridnum), dtype=torch.complex128)
        self.freq_part = torch.zeros((self.y_gridnum, self.x_gridnum), dtype=torch.complex128)


    # use the fftw packages
    def maskfft(self):
        self.spat_part[:] = torch.fft.ifftshift(self.data)
        self.freq_part = torch.fft.fftn(self.spat_part)
        self.fdata = torch.fft.fftshift(self.freq_part)


    def maskfftold(self):
        self.fdata = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(self.data)))

    def smooth(self):
        xx = torch.linspace(-1, 1, 21)
        X, Y = torch.meshgrid(xx, xx, indexing="xy")
        R = X ** 2 + Y ** 2
        # change G to 4-D to use F.conv2d
        G = torch.exp(-10 * R)
        G = G.view(1, 1, *G.shape)
        # print(G.shape)
        # print(self.data.shape)
        D = F.conv2d(0.9 * self.data.unsqueeze(0) + 0.05, G, padding="same") / torch.sum(G)
        self.sdata = D.squeeze().to(torch.float64)
        # print(self.sdata.shape)


if __name__ == "__main__":
    """polygon 2 mask"""
    # mp = [ [[-1,6],[-1, 2],[1, 2],[1, 1],[6, 1],[6, 0],[0, 0],[0, 1],[-2, 1],[-2, 6],[-1, 6]], \
    #   [[6, -1],[6, -2],[1, -2],[1, -3],[4, -3],[4, -6],[3, -6],[3, -4],[0, -4],[0, -1],[6, -1]] ]
    # m = Mask()
    # m.x_range = [-300.0,300.0]
    # m.y_range = [-300.0,300.0]
    # m.x_gridsize = 1.5
    # m.y_gridsize = 1.5
    # m.CD = 45
    # m.polygons = mp
    # m.poly2mask()

    """from GDS"""
    gds_dir = '/home/gjchen21/phd/projects/smo/SMO-ICCAD23/data/NanGateLibGDS'
    gds_dir = Path(gds_dir)

    m = Mask(gds_dir / "AND2_X4.gds", 10)
    m.x_range = [-300.0, 300.0]
    m.y_range = [-300.0, 300.0]
    m.x_gridsize = 10
    m.y_gridsize = 10

    m.openGDS()
    m.maskfft()
    m.smooth()
    torch_arr_bound(m.data, "m.data")
    torch_arr_bound(m.sdata, "m.sdata")
    torch_arr_bound(m.fdata, "m.fdata")
    # plt.imshow(
    #     m.data,
    #     extent=(m.x_range[0], m.x_range[1], m.y_range[0], m.y_range[1]),
    #     cmap="hot",
    #     interpolation="none",
    # )
    # plt.figure()
    # plt.imshow(
    #     m.sdata,
    #     extent=(m.x_range[0], m.x_range[1], m.y_range[0], m.y_range[1]),
    #     cmap="hot",
    #     interpolation="none",
    # )
    # plt.show()
