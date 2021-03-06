'''
Author(s): Christian Roncal
Leiserson Research Group March 1
'''

import torch
import torch.nn as nn
import torch.functional 
# from engine import Engine

class BMF(nn.Module):

    def create_embeddings(self, n_embeddings, dim, sparsity):
        if(self.sparse):
            embedding = nn.Embedding(
                num_embeddings = n_embeddings, 
                embedding_dim = dim, 
                sparse = self.sparse)

            nn.init.sparse_(embedding.weight.data, sparsity=self.sparse)

            return embedding

        else: 
            embedding = nn.Embedding(
                num_embeddings = n_embeddings, 
                embedding_dim = dim)
            # nn.init.xavier_normal_(embedding.weight)
            nn.init.normal_(embedding.weight.data)

            return embedding

    def loadAsEmbeddings(self, vfeats, hfeats):
        self.vfeats = nn.Embedding.from_pretrained(torch.from_numpy(vfeats).float())
        self.hfeats = nn.Embedding.from_pretrained(torch.from_numpy(hfeats).float())
        self.vfeats.weight.requires_grad=False
        self.hfeats.weight.requires_grad=False

    def __init__(self, config):
        super(BMF, self).__init__()

        self.num_virus = config['num_virus']
        self.num_human = config['num_human']
        self.latent_dim = config['latent_dim']
        self.sparse = config['sparse']

        if config['hfeats'] is not None and config['vfeats'] is not None:
            self.loadAsEmbeddings(config['vfeats'], config['hfeats'])

        self.virus, self.human = [self.create_embeddings(*dims, self.sparse) 
            for dims in [(self.num_virus, self.latent_dim), (self.num_human, self.latent_dim)]
        ]
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        nn.init.xavier_normal_(self.affine_output.weight.data)
        self.logistic = torch.nn.Sigmoid()


    def forward(self, v_idxs, h_idxs):
        U = self.virus(v_idxs)
        x = self.vfeats(v_idxs)
        V = self.human(h_idxs)
        y = self.hfeats(h_idxs)

        xUVy = torch.mul(x, U)
        xUVy = torch.mul(xUVy, V)
        xUVy = torch.mul(xUVy, y)

        logits = self.affine_output(xUVy)
        return self.logistic(logits)