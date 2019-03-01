'''
Author(s): Christian Roncal
Leiserson Research Group 03/01/2019 
Based on  https://github.com/LaceyChen17/neural-collaborative-filtering
'''
import torch
from torch.autograd import Variable
from utils import *


class Engine(object):
    
    def __init__(self, config):
        self.config = config
        self.opt = use_optimizer(self.model, config)
        self.crit = torch.nn.MSELoss()

    def train_a_batch(self, x, y, idxs):
        # make sure have model attr with assert