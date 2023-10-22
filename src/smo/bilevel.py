'''
Decription: Bilevel Optimization.
How one problem's forward input to another?
As defined in 'forward', 2 params can be considered as 2 inputs.

MO Abbe Module forward: input params and forward.
 and training_step: define model step and training step, 
 now you can input params defined in the Module.

In SO, source_params is defined and input to MO.
In MO, mask_params is defined and input to SO.
'''

import sys
sys.path.append('.')

import torch
from torch import optim

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem

from src.models.litho.source import Source
from src.models.litho.img_mask import Mask

from models import MO, SO

# origin source and mask
source = Source()
mask_path = 'data/img2013/M1_test1.png'
mask = Mask(layout_path=mask_path,target_path=mask_path)

# mo problem params
mo_config = Config(retain_graph=True)

# need to add params.
mo_module = MO(source, mask)

mo_opt = optim.Adam(
    mo_module.parameters(), 
    lr = 0.01, 
    # betas = (0.5, 0.999), 
    weight_decay = 0.0
)

mo_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    mo_opt, 
    mode = 'min', 
    factor = 0.1, 
    patience = 20, 
    min_lr = 0
)

# so problem params.
so_config = Config(type="darts", unroll_steps=1)

# need to add params.
so_module = SO(source, mask)

so_opt = optim.Adam(
    so_module.parameters(), 
    lr = 0.01, 
    # betas = (0.5, 0.999), 
    weight_decay = 0.0
)

so_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    so_opt, 
    mode = 'min', 
    factor = 0.1, 
    patience = 20, 
    min_lr = 0
)

# source_value and mask_value
class MOProblem(ImplicitProblem):
    def __init__(self, 
                 name: str = 'mo', 
                 config: Config = mo_config, 
                 module: MO = mo_module, 
                 optimizer = mo_opt, 
                 scheduler = mo_scheduler, 
                 train_data_loader = None, 
                 extra_config = None, 
                 weight_l2: float = 1000, 
                 weight_pvb: float = 8000,):
        super().__init__(name, config, module, optimizer, scheduler, train_data_loader, extra_config)

        self.criterion = torch.nn.MSELoss()
        self.weight_l2 = weight_l2
        self.weight_pvb = weight_pvb

    def training_step(self,):
        # get source_params from SO
        _, _, source_params = self.so()

        _, RIlist, _ = self.forward(source_params)
        l2 = self.criterion(RIlist[1], self.mask.target_data.float())
        pvb = self.criterion(RIlist[1], RIlist[0]) + self.criterion(RIlist[1], RIlist[2])
        loss = l2 * self.weight_l2 + pvb * self.weight_pvb
        return loss


class SOProblem(ImplicitProblem):
    def __init__(self, 
                 name: str = 'so', 
                 config: Config = so_config, 
                 module: SO = so_module, 
                 optimizer = so_opt, 
                 scheduler = so_scheduler, 
                 train_data_loader = None, 
                 extra_config = None, 
                 weight_l2: float = 1000, 
                 weight_pvb: float = 8000,):
        super().__init__(name, config, module, optimizer, scheduler, train_data_loader, extra_config)

        self.criterion = torch.nn.MSELoss()
        self.weight_l2 = weight_l2
        self.weight_pvb = weight_pvb

    def training_step(self):
        # get mask_params from MO
        _, _, mask_params = self.mo()
        _, RIlist, _ = self.forward(mask_params)
        l2 = self.criterion(RIlist[1], self.mask.target_data.float())
        pvb = self.criterion(RIlist[1], RIlist[0]) + self.criterion(RIlist[1], RIlist[2])
        loss = l2 * self.weight_l2 + pvb * self.weight_pvb
        return loss

mo_problem = MOProblem()
so_problem = SOProblem()

# define SMO, where outter is mo and inner is so.
smo_problem = [mo_problem, so_problem]
l2u = {so_problem: [mo_problem]}
u2l = {mo_problem: [so_problem]}
dependencies = {"l2u": l2u, "u2l": u2l}

# class NASEngine(Engine):
#     @torch.no_grad()
#     def validation(self):
#         # _, _, source_params = self.so()
#         _, _, mask_params = self.mo()

#         _, RIlist, _  = self.so
#         RI_min = torch.where(RIlist[0] > 0.5, 1.0, 0.0).float()
#         RI_norm = torch.where(RIlist[1] > 0.5, 1.0, 0.0).float()
#         RI_max = torch.where(RIlist[2] > 0.5, 1.0, 0.0).float()

#         for x, target in test_queue:
#             x, target = x.to(device), target.to(device, non_blocking=True)
#             alphas = self.arch()
#             _, correct = self.classifier.module.loss(x, alphas, target, acc=True)
#             corrects += correct
#             total += x.size(0)
#         acc = corrects / total

#         alphas = self.arch()
#         torch.save({"genotype": self.classifier.module.genotype(alphas)}, "genotype.t7")
#         return {"loss": loss}

# engine = NASEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine_config = EngineConfig(train_iters=10, logger_type="none")
engine = Engine(config=engine_config, problems=smo_problem, dependencies=dependencies)
engine.run()
