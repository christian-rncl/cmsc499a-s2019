'''
Author(s): Christian Roncal
Leiserson Research Group March 1
based on: https://github.com/LaceyChen17/neural-collaborative-filtering/blob/master/src/gmf.py
'''

import torch
import torch.nn as nn
from engine import Engine
from utils import use_cuda

class GMF(nn.Module):

    def create_embeddings(self, n_embeddings, dim, sparsity):
        if(self.sparse):
            embedding = nn.Embedding(
                num_embeddings = n_embeddings, 
                embedding_dim = dim, 
                sparse = self.sparse)

            nn.init.sparse_(embedding.weight, sparsity=self.sparsity)

            return embedding

        else: 
            embedding = nn.Embedding(
                num_embeddings = n_embeddings, 
                embedding_dim = dim)
            nn.init.xavier_normal_(embedding.weight)

            return embedding

    def __init__(self, config):
        super(GMF, self).__init__()

        self.num_virus = config['num_virus']
        self.num_human = config['num_human']
        self.latent_dim = config['latent_dim']
        self.sparse = config['sparse']
        # % of elements / col to be set to 0
        if(self.sparse):
            self.sparsity = config['sparsity']

        self.virus, self.human = [self.create_embeddings(*dims, self.sparse) 
        for dims in [(self.num_virus, self.latent_dim), (self.num_human, self.latent_dim)]
        ]
        ## TODO: experiment with bias

    def forward(self, x, y, x_nodes, y_nodes):
        assert(len(x_nodes) == len(y_nodes))
        U_xi = self.human(x_nodes)
        V_yj = self.virus(y_nodes)
        xUVy = (U_xi.double() * x.double()).sum(1) 
        xUVy = (xUVy.double() * V_yj.t().double()).sum(1)
        xUVy = (xUVy.double() * y).sum(1)

        return xUVy

class GMFEngine(Engine):
    def __init__(self, config):
        self.model = GMF(config)
        if config['cuda']:
            use_cuda(True)
            self.model.cuda()

        super(GMFEngine, self).__init__(config)