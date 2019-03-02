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
        embedding = nn.Embedding(
            num_embeddings = n_embeddings, 
            embedding_dim = dim, 
            sparse = self.sparse)

        nn.init.sparse_(embedding.weight, sparsity=self.sparsity)

        return embedding

    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_virus = config['num_virus']
        self.num_human = config['num_human']
        self.latent_dim = config['latent_dim']
        self.sparse = config['sparse']
        # % of elements / col to be set to 0
        self.sparsity = config['sparsity']

        self.virus, self.human = [self.create_embeddings(*dims, self.sparse) 
        for dims in [(self.num_virus, self.latent_dim), (self.num_human, self.latent_dim)]
        ]
        ## TODO: experiment with bias

    def forward(self, x, y, idxs):
        U, Y = self.get_embeddings(interaction_idxs)

        Ux_idx = self.human(x_i)
        Vy_idx = self.virus(y_i)
        xU = (x * Ux_idx).sum(1)
        xUV = (xU * Vy_idx).sum(1)
        xUVy = (xUV * y).sum(1)
        return xUVy.view(-1, 1)

    def get_embeddings(self, idxs):
        x_indices = [self.config['interactions']['node1'][i] for i in
        idxs]
        y_indices = [self.config['interactions']['node2'][i] for i in
        idxs]
        assert(len(x_indices) == len(y_indices))
        n = len(x_indices)
        x_embeddings = []
        y_embeddings = []
        
        for i in range(n):
            x_embeddings[i] = self.human.get(i)
            y_embeddings[i] = self.virus.get(i)

       return x_embeddings, y_embeddings 

class GMFEngine(Engine):
    def __init__(self, config):
        self.model = GMF(config)
        if config['cuda']:
            use_cuda(True)
            self.model.cuda()

        super(GMFEngine, self).__init__(config)