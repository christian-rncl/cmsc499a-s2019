'''
Author(s): Christian Roncal
Leiserson Research Group March 1
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# from engine import Engine

class GMF(nn.Module):

    def create_embeddings(self, n_embeddings, dim, sparsity):
        if(self.sparse):
            embedding = nn.Embedding(
                num_embeddings = n_embeddings, 
                embedding_dim = 10, 
                sparse = self.sparse)

            nn.init.sparse_(embedding.weight.data, sparsity=self.sparse)

            return embedding

        else: 
            embedding = nn.Embedding(
                num_embeddings = n_embeddings, 
                embedding_dim = dim)
            embedding.weight.data.normal_(-.01, .01)

            return embedding

    def __init__(self, config):
        super(GMF, self).__init__()

        self.num_virus = config['num_virus']
        self.num_human = config['num_human']
        self.latent_dim = config['latent_dim']
        self.sparse = config['sparse']
        self.reg = 1.0
        self.reg_bias = 1.0

        self.virus, self.human, self.vb, self.hb = [self.create_embeddings(*dims, self.sparse) 
            for dims in [(self.num_virus, self.latent_dim), (self.num_human, self.latent_dim), (self.num_virus, 1), (self.num_human, 1)]
        ]
        # self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim * 2)
        # self.affine_output.weight.data.uniform_(-.01, .01)
        self.bias = nn.Parameter(torch.ones(1))
        # self.logistic = nn.Sigmoid()


    def forward(self, v_idxs, h_idxs):
        U = self.virus(v_idxs)
        V = self.human(h_idxs)
        bu = self.vb(v_idxs).squeeze()
        bv = self.hb(h_idxs).squeeze()
        biases = (self.bias + bu + bv)

        UV = torch.sum(U*V, dim=1)
        pred = UV + biases
        print(pred.shape)
        return pred
        # return UV


    def l2_regularize(self, array):
        loss = torch.sum(array ** 2.0)
        return loss

    def loss(self, prediction, target):
        loss_mse = F.mse_loss(prediction, target)
        prior_bias_virus =  self.l2_regularize(self.virus.weight) * self.reg_bias
        prior_bias_human = self.l2_regularize(self.human.weight) * self.reg_bias

        prior_virus = self.l2_regularize(self.virus.weight) * self.reg
        prior_human = self.l2_regularize(self.human.weight) * self.reg
        total = loss_mse + prior_virus + prior_human + prior_bias_virus + prior_bias_human
        
        return total
