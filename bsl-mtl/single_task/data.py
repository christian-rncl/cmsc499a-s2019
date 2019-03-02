'''
Author(s): Christian Roncal
Leiserson Research Group 2/28/2019
Based on: https://github.com/LaceyChen17/neural-collaborative-filtering
'''
import torch
import pandas as pd
import numpy as np
import numpy.random as rand
from torch.utils.data import DataLoader, Dataset


class SingleTaskDataset(Dataset):
    
    ## must be tensors
    def __init__(self, virus, human, virus_nodes, human_nodes):
        self.human = human
        self.virus = virus
        self.virus_nodes = virus_nodes
        self.human_nodes = human_nodes


    def __getitem__(self, idx):
        return (self.human[idx], self.virus[idx], 
        self.virus_nodes[idx], self.human_nodes[idx])

    def __len__(self):
        return len(self.human)

'''
    Single task experiments data generator,
    Given params interaction matrix, human/virus features for a single task, 
    process data and split to train/cv/test based on settings 
'''
class SingleTaskGenerator(object):
    '''
    interactions: df representing interactions
    human: np array 
    '''
    def __init__(self, interactions, human, virus, pct_train):
        
        self.interactions = interactions
        self.human = human
        self.virus = virus
        #get indices of nonzero in interactions matrix
        self.split_data(interactions, pct_train)

    def split_data(self, interactions, pct_train):
        pool = interactions.loc[interactions['edge'] == 1].index.values
        rand.shuffle(pool)
        cutoff = int(pct_train * len(pool))

        # save indices
        self.train = pool[:cutoff]
        self.N = len(self.train)
        self.test = pool[cutoff:]
    
        
    def create_loader(self, bs):
        human = []
        virus = []
        human_nodes = []
        virus_nodes = []

        for i in range(self.N):
            idx = int(self.train[i])
            # print(type(idx))
            v_idx = int(self.interactions['node1'][idx])
            h_idx = int(self.interactions['node2'][idx])

            human.append(torch.from_numpy(self.human[h_idx, :]))
            virus.append(torch.from_numpy(self.virus[v_idx, :]))
            virus_nodes.append(v_idx)
            human_nodes.append(h_idx)

        dset = SingleTaskDataset(virus, human, human_nodes, virus_nodes)

        return DataLoader(dset, batch_size=bs, shuffle=True)