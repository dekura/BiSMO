import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from gdsii.library import Library



class Mask:
    """Binary Mask.

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
        layout_path: str,
        layername: int = 11,
        pixels_per_um: int = 1000,
        xmax=1024,
        ymax=1024,
        maskxpitch = 2048,
        maskypitch = 2048,
        CD=45,
    ):
        self.x_range = [-xmax, xmax]  # nm
        self.y_range = [-ymax, ymax]
        self.maskxpitch = maskxpitch
        self.maskypitch = maskypitch
        self.x_gridsize = 2 * xmax // maskxpitch # nm
        self.y_gridsize = 2 * ymax // maskypitch
        print(f"mask x_gridsize: {self.x_gridsize}")
        print(f"mask y_gridsize: {self.y_gridsize}")
        self.mask_groups = []
        self.CD = CD
        self.layout_path = layout_path
        self.layername = layername
        self.pixels_per_um = pixels_per_um

        """
        Process calculation
        """
        # self.openGDS()
        # self.maskfft()

    def poly2mask(self):
        """Get Pixel-based Mask Image from Polygon Data The Poylgon Data Form are sensitive Similar
        to poly2mask in Matlab."""
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

    def openGDS(self):
        gdsdir = self.layout_path
        layername = self.layername
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
                    aa = np.array(a[ii].xy) / 10000 * pixels_per_um
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

        center_x = (xmax - xmin) // 2
        center_y = (ymax - ymin) // 2

        cpoints = []

        coords_width = self.maskxpitch * self.x_gridsize
        cx_r = np.arange(center_x, xmax,   coords_width // 2)
        cx_l = -np.arange(-center_x, -xmin, coords_width // 2)
        cxs = np.hstack((cx_l, cx_r))

        coords_height = self.maskypitch * self.y_gridsize
        cy_u = np.arange(center_y, ymax,  coords_height // 2)
        cy_d = -np.arange(-center_y, -ymin, coords_height // 2)
        cys = np.hstack((cy_d, cy_u))

        for x in cxs:
            for y in cys:
                cpoints.append((x, y))

        cpoints = list(set(cpoints))

        # spoints.append((xmin, ymin))

        # print(spoints)

        for cc in cpoints:
            self.xmin = cc[0] - (coords_width // 2)
            self.xmax = self.xmin + coords_width
            self.ymin = cc[1] - (coords_height // 2)
            self.ymax = self.ymin + coords_height
            self.x_range = [self.xmin, self.xmax]
            self.y_range = [self.ymin, self.ymax]

            self.x_gridnum = int(self.maskxpitch)
            self.y_gridnum = int(self.maskypitch)
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

            self.mask_groups.append(torch.from_numpy(np.array(img)))

        self.data = self.mask_groups[0]
        # Fourier transform pair, pyfftw syntax
        self.spat_part = torch.zeros((self.y_gridnum, self.x_gridnum), dtype=torch.complex128)
        self.freq_part = torch.zeros((self.y_gridnum, self.x_gridnum), dtype=torch.complex128)

    def open_img(self):
        img = Image.open(self.layout_path)
        img = img.convert("L")
        self.mask_groups.append(torch.from_numpy(np.array(img)))
        self.data = self.mask_groups[0]
        # Fourier transform pair, pyfftw syntax
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
        R = X**2 + Y**2
        # change G to 4-D to use F.conv2d
        G = torch.exp(-10 * R)
        G = G.view(1, 1, *G.shape)
        # print(G.shape)
        # print(self.data.shape)
        D = F.conv2d(0.9 * self.data.unsqueeze(0) + 0.05, G, padding="same") / torch.sum(G)
        self.sdata = D.squeeze().to(torch.float64)
        # print(self.sdata.shape)
