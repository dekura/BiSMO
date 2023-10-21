'''
Decription: Bilevel Optimization.
How one problem's forward input to another?
As defined in 'forward', 2 params can be considered as 2 inputs.

'''

import sys
sys.path.append('.')

import torch
import torch.nn.functional as F
from torch import optim

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem

from src.models.litho.source import Source
from src.models.litho.img_mask import Mask

from models import MO, SO

# mo problem params
mo_config = Config(retain_graph=True)

# need to add params.
mo_module = MO()

mo_opt = optim.Adam(
    mo_module.parameters(), 
    lr = 0.01, 
    betas = (0.5, 0.999), 
)

mo_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    mo_opt, 
    mode = 'min', 
    factor = 0.1, 
    patience = 20, 
)

# so problem params.
so_config = Config(type="darts", unroll_steps=1)

# need to add params.
so_module = SO()

so_opt = optim.Adam(
    so_module.parameters(), 
    lr = 0.01, 
    betas = (0.5, 0.999), 
)

so_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    so_opt, 
    mode = 'min', 
    factor = 0.1, 
    patience = 20, 
)

# source_value and mask_value
class MOProblem(ImplicitProblem):
    def __init__(self, 
                 name: str = 'mo', 
                 config: Config = so_config, 
                 module: MO = so_module, 
                 optimizer = so_opt, 
                 scheduler = so_scheduler, 
                 train_data_loader = None, 
                 extra_config = None):
        super().__init__(name, config, module, optimizer, scheduler, train_data_loader, extra_config)

    def training_step(self,):
        alphas = self.forward()
        loss = self.classifier.module.loss(x, alphas, target)

        return loss

class SOProblem(ImplicitProblem):
    def __init__(self, 
                 name: str = 'so', 
                 config: Config = so_config, 
                 module: MO = so_module, 
                 optimizer = so_opt, 
                 scheduler = so_scheduler, 
                 train_data_loader = None, 
                 extra_config = None):
        super().__init__(name, config, module, optimizer, scheduler, train_data_loader, extra_config)

    def training_step(self):

        # mo forward
        alphas = self.mo()
        loss = self.module.loss(x, alphas, target)

        return loss
