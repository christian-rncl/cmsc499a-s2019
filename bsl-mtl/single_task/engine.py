'''
Author(s): Christian Roncal
Leiserson Research Group 03/01/2019 
Based on  https://github.com/LaceyChen17/neural-collaborative-filtering
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
from utils import *


class Engine(object):
    
    def __init__(self, config):
        self.config = config
        self.interactions = config['interactions']
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        self.crit = nn.MSELoss()

    def train_a_batch(self, x, y, idx):
        # make sure have model attr with assert
        # tensor of 1's for comparison
        gt = torch.from_numpy(np.ones(len(idx)))
        x_i = self.interactions['node1'][idx]
        y_i = self.interactions['node2'][idx]
        if self.config['cuda']:
            x, y, x_i, y_i =  x.cuda(), y.cuda(), x_i.cuda(), y_i.cuda()
        
        self.opt.zero_grad()
        pred = self.model(x, y, idx)
        loss = self.crit(pred, gt)
        loss.backward()
        self.opt.step()
        
        if self.config['cuda'] is True:
            loss = loss.data.cpu().numpy()[0]
        else:
            loss = loss.data.numpy()[0]
        return loss

        
    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0

        for batch_id, batch in enumerate(train_loader):
            # assert 
            x, y, Mij = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])

            if self.config['cuda'] is True:
                x = x.cuda()
                y = y.cuda()
                Mij = Mij.cuda()

            loss = self.train_a_batch(x, y, Mij)
            print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)
