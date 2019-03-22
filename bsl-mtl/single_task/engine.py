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

    def train_a_batch(self, x, y, x_nodes, y_nodes):

        if self.config['cuda']:
            x, y, x_nodes, y_nodes =  x.cuda(), y.cuda(), x_nodes.cuda(), y_nodes.cuda()
        
        self.opt.zero_grad()
        pred = self.model(x, y, x_nodes, y_nodes)
        loss = self.crit(gt, pred)
        loss.backward()
        self.opt.step()
        
        if self.config['cuda'] is True:
            loss = loss.data.cpu().numpy()
        else:
            loss = loss.data.numpy()[0]
        return loss

        
    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0

        for batch_id, batch in enumerate(train_loader):
            # assert 
            # unpack
            n_vars = len(batch)
            x, y, x_nodes, y_nodes = [Variable(batch[i]) for i in range(n_vars)]

            if self.config['cuda']:
                x = x.cuda()
                y = y.cuda()
                x_nodes = x_nodes.cuda()
                y_nodes = y_nodes.cuda()

            loss = self.train_a_batch(x, y, x_nodes, y_nodes)
            print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss

        self._writer.add_scalar('model/loss', total_loss, epoch_id)
