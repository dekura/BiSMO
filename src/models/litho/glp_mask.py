"""
Author: Hongquan
Contact: fallenhhq@gmail.com
Description: load mask from glp.
m.data, m.fdata, m.sdata are used.
"""
import torch
from torchvision import transforms as T
import numpy as np 
import cv2
import torch.nn.functional as F

class Mask:
    """Binary Mask.

    Read from glp.
    """

    def __init__(
        self,
        layout_path: str,
        down = 1,
    ):

        self._layout_path = layout_path
        self._polygons = [] # PGON list, list of points
        self._down = down
        """
        Process calculation
        """
        # self.openGLP()
    
    @property
    def polygons(self): 
        return self._polygons
    
    def openGLP(self):
        if self._layout_path is None:
            return
        # read by lines
        with open(self._layout_path, "r") as fin: 
            lines = fin.readlines()
        for line in lines: 
            splited = line.strip().split()
            if len(splited) < 7: 
                continue
            # change RECT to PGON
            if splited[0] == "RECT":
                info = splited[3:7]
                # left-bottom (x, y)
                frX = int(info[0])
                frY = int(info[1])
                # width and height
                toX = frX + int(info[2])
                toY = frY + int(info[3])
                # 4 points of RECT
                coords = [[frX//self._down, frY//self._down], [frX//self._down, toY//self._down], 
                          [toX//self._down, toY//self._down], [toX//self._down, frY//self._down]]
                self._polygons.append(coords)
            elif splited[0] == "PGON": 
                info = splited[3:]
                coords = []
                for idx in range(0, len(info), 2): 
                    coordX = int(info[idx])
                    coordY = int(info[idx+1])
                    coords.append([coordX//self._down, coordY//self._down])
                self._polygons.append(coords)
    
    # get boundary range of mask, (minX, maxX, minY, maxY)
    def range(self):
        # infined
        minX = 1e12
        minY = 1e12
        maxX = -1e12
        maxY = -1e12
        for polygon in self._polygons: 
            for point in polygon: 
                # a point on the mask range
                if point[0] < minX: 
                    minX = point[0]
                if point[1] < minY: 
                    minY = point[1]
                if point[0] > maxX: 
                    maxX = point[0]
                if point[1] > maxY: 
                    maxY = point[1]
        return minX, minY, maxX, maxY

    # move distance as point (x, y) -> (x+deltaX, y+deltaY)
    def move(self, deltaX, deltaY):
        for polygon in self._polygons: 
            for point in polygon: 
                point[0] += deltaX
                point[1] += deltaY


    def center(self, sizeX=2048, sizeY=2048, offsetX=512, offsetY=512): 
        # canvas <- minX, minY, maxX, maxY
        # input is poly-ed
        canvas = self.range()
        # width and height
        canvasX = canvas[2] - canvas[0]
        canvasY = canvas[3] - canvas[1]
        halfX = (sizeX - canvasX) // 2
        halfY = (sizeY - canvasY) // 2
        # the distance between centers of whole map and mask
        deltaX = halfX - canvas[0]
        deltaY = halfY - canvas[1]
        # in the center and then x - offsetX, y - offsetY
        self.move(deltaX - offsetX, deltaY - offsetY)
    
    # print image, img: ndarray
    def image(self, sizeX=2048, sizeY=2048, offsetX=512, offsetY=512): 
        polygons = list(map(lambda x: np.array(x, np.int64) + np.array([[offsetX, offsetY]]), self._polygons))
        img = np.zeros([sizeX, sizeY], dtype=np.float32)
        for idx in range(len(polygons)): 
            img = cv2.fillPoly(img, [polygons[idx]], color=255)
        return img
    # norm to 0-1
    def mat(self, sizeX=2048, sizeY=2048, offsetX=512, offsetY=512): 
        return self.image(sizeX, sizeY, offsetX, offsetY) / 255.0

    # TileSizeX, TileSizeY, OffsetX, OffsetY are in the config
    def poly2tensor(self, TileSizeX=2048, TileSizeY=2048, OffsetX=512, OffsetY=512):
        self.openGLP()
        # poly -> ndarray
        self.center(TileSizeX, TileSizeY, OffsetX, OffsetY)
        mask_np = self.mat(TileSizeX, TileSizeY, OffsetX, OffsetY)
        # 先写死，后面从config中调的话再改
        self.data = torch.tensor(mask_np, dtype=torch.float32, device=torch.device("cuda"))
        self.params = self.data * 2.0 - 1.0

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

    # use the fftw packages
    def maskfft(self):
        self.fdata = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(self.data)))


if __name__ == "__main__":
    png_path = (
        "./data/ICCAD2013/M1_test1__cross.glp"
    )
    m = Mask(layout_path=png_path)
    # m.open_layout()
    m.poly2tensor()
    from utils import torch_arr_bound

    torch_arr_bound(m.data, "m.data")
