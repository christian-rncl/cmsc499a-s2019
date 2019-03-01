'''
Leiserson Research Group 2/28/2019
Author(s): Christian Roncal
'''
import torch
import pandas as pd
import numpy as np
import numpy.random as rand
from torch.utils.data import DataLoader, Dataset


class SingleTaskDataset(Dataset):
    
    ## must be tensors
    def __init__(self, interactions, human, virus):
        self.human = human, self.virus = virus

    def __getitem__(self, idx):
        return self.human[idx], self.virus[idx]

    def __len__(self):
        return self.interactions.size(0)

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
    def __init__(self, interactions, human, virus, split_settings):
        
        self.interactions = interactions.values
        self.human = human.values
        self.virus = virus.values
        #get indices of nonzero in interactions matrix
        self.split_data(interactions)

    def split_data(self, interactions):
        pool = interactions.loc[interactions['edge'] == 1].index.values
        rand.shuffle(self.pool)
        cutoff = int(.80 * len(self.pool))

        # save indices
        self.train = pool[:cutoff]
        self.N = len(self.train)
        self.test = pool[cutoff:]
    
        
    def create_data_loader(self, bs):
        human = []
        virus = []
        interaction_idx = []

        for i in range(self.N):
            idx =  self.test[i]
            h_idx = int(self.interactions['node1'][idx])
            v_idx = int(self.interactions['node2'][idx])

            human.append(torch.from_numpy(self.human[h_idx, :]))
            virus.append(torch.from_numpy(self.virus[v_idx, :]))
            interaction_idx.append(torch.long(idx))
        print(human, virus, interaction_idx)