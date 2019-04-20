'''
Author(s): Christian Roncal
Leiserson Research Group March 1
'''

import torch
import torch.nn as nn
import torch.functional 
# from engine import Engine

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
            # nn.init.xavier_normal_(embedding.weight)
            nn.init.normal_(embedding.weight)

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

        # self.virus, self.human, self.virus_b, self.human_b = [self.create_embeddings(*dims, self.sparse) 
        #     for dims in [(self.num_virus, self.latent_dim), (self.num_human, self.latent_dim),
        #         (self.num_virus, 1), (self.num_human, 1)]
        # ]
        self.virus, self.human = [self.create_embeddings(*dims, self.sparse) 
            for dims in [(self.num_virus, self.latent_dim), (self.num_human, self.latent_dim)]
        ]
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        nn.init.nor
        self.logistic = torch.nn.Sigmoid()


    def forward(self, v_idxs, h_idxs):
        U = self.virus(v_idxs)
        V = self.human(h_idxs)
        UV = torch.mul(U, V)
        logits = self.affine_output(UV)

        return self.logistic(logits)

        # for bilinear
        # xUVy = (U_xi.double() * h_feats.double()).sum(1)  # xU
        # xUVy = (xUVy.double() * V_yj.t().double()).sum(1) # xUV
        # xUVy = (xUVy.double() * v_feats.double()).sum(1) # xUVy

        # return torch.sigmoid(xUVy)