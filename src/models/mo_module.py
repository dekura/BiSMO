import sys
sys.path.append('.')

from src.utils.settings import *
# from src.utils.glp import Design # poly, center
from src.models.litho.glp_mask import Mask
from src.models.litho.eval import Evaluate
import src.models.litho.litho as lithosim
# from src.utils.utils import parseConfig
import time
# import src.initializer as initializer
# import src.evaluation as evaluation

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as func
import cv2

"""
Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
"""

'''
class SimpleCfg: 
    def __init__(self, config): 
        # Read the config from file or a given dict
        if isinstance(config, dict): 
            self._config = config
        elif isinstance(config, str): 
            self._config = parseConfig(config)
        required = ["Iterations", "TargetDensity", "SigmoidSteepness", "WeightEPE", "WeightPVBand", "WeightPVBL2", "StepSize", 
                    "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in required: 
            assert key in self._config, f"[SimpleILT]: Cannot find the config {key}."
        intfields = ["Iterations", "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in intfields: 
            self._config[key] = int(self._config[key])
        floatfields = ["TargetDensity", "SigmoidSteepness", "WeightEPE", "WeightPVBand", "WeightPVBL2", "StepSize"]
        for key in floatfields: 
            self._config[key] = float(self._config[key])

    def __getitem__(self, key): 
        return self._config[key]

input config lithosim
Iterations 20
TargetDensity 0.225
SigmoidSteepness 4.0
WeightEPE 0.5
WeightPVBL2 1.0
WeightPVBand 0.0
StepSize 0.5

TileSizeX 2048
TileSizeY 2048
OffsetX   512
OffsetY   512
ILTSizeX  1024
ILTSizeY  1024
'''

class SimpleILT: 
    def __init__(self, 
                 Iterations: int = 20, 
                 TargetDensity: float = 0.225, 
                 SigmoidSteepness: float = 4.0, 
                 WeightEPE: float = 0.5, 
                 WeightPVBL2: float = 1.0, 
                 WeightPVBand: float = 0.0, 
                 StepSize: float = 0.5, 
                 TileSizeX: int = 2048, 
                 TileSizeY: int = 2048, 
                 OffsetX: int = 512,
                 OffsetY: int = 512, 
                 ILTSizeX: int = 1024, 
                 ILTSizeY: int = 1024,
                #  config=SimpleCfg("./config/simpleilt2048.txt"), 
                #  lithosim=lithosim.LithoSim(), 
                 device=DEVICE): 
        super(SimpleILT, self).__init__()
        # self._config = config
        self._Iterations = Iterations
        self._TargetDensity = TargetDensity
        self._SigmoidSteepness = SigmoidSteepness
        self._WeightEPE = WeightEPE
        self._WeightPVBL2 = WeightPVBL2
        self._WeightPVBand = WeightPVBand
        self._StepSize = StepSize
        self._TileSizeX = TileSizeX
        self._TileSizeY = TileSizeY
        self._OffsetX = OffsetX
        self._OffsetY = OffsetY
        self._ILTSizeX = ILTSizeX
        self._ILTSizeY = ILTSizeY
        self._device = device
        # Lithosim
        self._lithosim = lithosim.LithoSim().to(DEVICE)

        # Filter
        self._filter = torch.zeros([self._TileSizeX, self._TileSizeY], dtype=REALTYPE, device=self._device)
        self._filter[self._OffsetX:self._OffsetX + self._ILTSizeX, \
                     self._OffsetY:self._OffsetY + self._ILTSizeY] = 1
    
    def solve(self, target, params, curv=None, verbose=0): 
        # Initialize
        # if not isinstance(target, torch.Tensor): 
        #     target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        # if not isinstance(params, torch.Tensor): 
        #     params = torch.tensor(params, dtype=REALTYPE, device=self._device)
        backup = params
        params = params.clone().detach().requires_grad_(True)

        # Optimizer 
        opt = optim.SGD([params], lr=self._StepSize)
        # opt = optim.Adam([params], lr=self._config["StepSize"])

        # Optimization process
        lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
        bestParams = None
        bestMask = None
        for idx in range(self._Iterations): 
            # transformation func
            mask = torch.sigmoid(self._SigmoidSteepness * params) * self._filter
            mask += torch.sigmoid(self._SigmoidSteepness * backup) * (1.0 - self._filter)

            printedNom, printedMax, printedMin = self._lithosim(mask)
            l2loss = func.mse_loss(printedNom, target, reduction="sum")
            pvbl2 = func.mse_loss(printedMax, target, reduction="sum") + func.mse_loss(printedMin, target, reduction="sum")
            pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
            pvband = torch.sum((printedMax >= self._TargetDensity) != (printedMin >= self._TargetDensity))
            loss = l2loss + self._WeightPVBL2 * pvbl2 + self._WeightPVBand * pvbloss
            if not curv is None: 
                kernelCurv = torch.tensor([[-1.0/16, 5.0/16, -1.0/16], [5.0/16, -1.0, 5.0/16], [-1.0/16, 5.0/16, -1.0/16]], dtype=REALTYPE, device=DEVICE)
                curvature = func.conv2d(mask[None, None, :, :], kernelCurv[None, None, :, :])[0, 0]
                losscurv = func.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")
                loss += curv * losscurv
            if verbose == 1: 
                print(f"[Iteration {idx}]: L2 = {l2loss.item():.0f}; PVBand: {pvband.item():.0f}")

            if bestParams is None or bestMask is None or loss.item() < lossMin: 
                lossMin, l2Min, pvbMin = loss.item(), l2loss.item(), pvband.item()
                bestParams = params.detach().clone()
                bestMask = mask.detach().clone()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        return l2Min, pvbMin, bestParams, bestMask


    def serial(self, 
               SCALE: int = 1, 
            # TileSizeX, 
            # TileSizeY, 
            # OffsetX, 
            # OffsetY, 
            ): 
        # SCALE = 1
        # performance iterm
        l2s, pvbs, epes, shots, runtimes = [], [], [], [], []
        # litho = lithosim.LithoSim()
        # cfg   = SimpleCfg("./config/simpleilt2048.txt")
        # litho = lithosim.LithoSim("./config/lithosimple.txt")
        # solver = SimpleILT()
        for idx in range(1, 11): 
            design = Mask(f"./data/ICCAD2013/M1_test{idx}.glp", down = SCALE)
            design.poly2tensor()
            target, params = design.data, design.params
            # design.center(self._TileSizeX, self._TileSizeY, self._OffsetX, self._OffsetY)
            # target, params = initializer.PixelInit().run(design, self._TileSizeX, self._TileSizeY, self._OffsetX, self._OffsetY)
            
            begin = time.time()
            l2, pvb, bestParams, bestMask = self.solve(target, params, curv=None)
            runtime = time.time() - begin

            ref = Mask(f"./data/ICCAD2013/M1_test{idx}.glp", down = 1, TileSizeX = self._TileSizeX * SCALE, TileSizeY = self._TileSizeY * SCALE, OffsetX = self._OffsetX * SCALE, OffsetY = self._OffsetY * SCALE)
            ref.poly2tensor()
            target, params = ref.data, ref.params
            # ref.center(self._TileSizeX * SCALE, self._TileSizeY * SCALE, self._OffsetX * SCALE, self._OffsetY * SCALE)
            # target, params = initializer.PixelInit().run(ref, self._TileSizeX * SCALE, self._TileSizeY * SCALE, self._OffsetX * SCALE, self._OffsetY * SCALE)
            eval = Evaluate()
            l2, pvb, epe, shot = eval.eval(mask = bestMask, target = target, scale = SCALE, shots = True)
            # l2, pvb, epe, shot = evaluation.evaluate(bestMask, target, scale=SCALE, shots=True)
            cv2.imwrite(f"./data/result_img/MOSAIC_test{idx}.png", (bestMask * 255).detach().cpu().numpy())

            print(f"[Testcase {idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}; SolveTime: {runtime:.2f}s")

            l2s.append(l2)
            pvbs.append(pvb)
            epes.append(epe)
            shots.append(shot)
            runtimes.append(runtime)

        print(f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.1f}; Shot {np.mean(shots):.1f}; SolveTime {np.mean(runtimes):.2f}s")


if __name__ == '__main__':
    mo = SimpleILT()
    mo.serial()
