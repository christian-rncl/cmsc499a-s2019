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
        self.interactions = config['interactions']
        self.opt = use_optimizer(self.model, config)
        self.crit = torch.nn.MSELoss()

    def train_a_batch(self, x, y, idxs):
        # make sure have model attr with assert
        # tensor of 1's for comparison
        gt = torch.from_numpy(np.ones(len(idxs)))

        if self.config['cuda']:
            x, y, idx = x.cuda(), y.cuda(), idxs.cuda()
        
        self.opt.zero_grad()
        pred = self.model(user)
        loss = self.crit(pred, gt)
        loss.backward()
        self.opt.step()
        
        if self.config['use_cuda'] is True:
            loss = loss.data.cpu().numpy()[0]
        else:
            loss = loss.data.numpy()[0]
        return loss

        