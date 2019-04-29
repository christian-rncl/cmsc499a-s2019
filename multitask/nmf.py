
'''
Author(s): Christian Roncal
Leiserson Research Group March 1
'''

import torch
import torch.nn as nn
# import torch.functional 
# from engine import Engine

class NMF(nn.Module):
    def __init__(self, M, N, K):
        super(NMF, self).__init__()
        self.A = nn.Parameter(torch.rand(M, K, requires_grad=True))
        self.S = nn.Parameter(torch.rand(K, N, requires_grad=True))

    def forward(self):
        return torch.matmul(self.A, self.S)
