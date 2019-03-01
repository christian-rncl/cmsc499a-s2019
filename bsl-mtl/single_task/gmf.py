'''
Author(s): Christian Roncal
Leiserson Research Group March 1
based on: https://github.com/LaceyChen17/neural-collaborative-filtering/blob/master/src/gmf.py
'''

import torch
from engine import Engine
from utils import use_cuda

class GMF(torch.nn.Module):

    def create_embeddings(self, n_embeddings, dim, sparsity):
        embedding = torch.nn.Embedding(
            num_embeddings = n_embeddings, 
            embedding_dim = dim, 
            sparse = self.sparse)

        nn.init.sparse_(embedding, sparsity=self.sparsity)

        return embedding

    def __init__(self, config):
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

    def forward(self, x_idx, y_idx, x, y):
        Ux_idx = self.human(x_idx)
        Vy_idx = self.virus(y_idx)
        xU = (x * Ux_idx).sum(1)
        xUV = (xU * Vy_idx).sum(1)
        xUVy = (xUV * y).sum(1)
        return xUVy.view(-1, 1)

class GMFEngine(Engine):
    def __init__(self, config):
        self.model = GMF(config)
        if config['use_cuda']:
            use_cuda(True)
            self.model.cuda()

        super(GMFEngine, self).__init__(config)