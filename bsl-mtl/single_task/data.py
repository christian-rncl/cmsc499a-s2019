'''
Author(s): Christian Roncal
Leiserson Research Group 2/28/2019
Based on: https://github.com/LaceyChen17/neural-collaborative-filtering
'''
import torch
import pandas as pd
import numpy as np
import numpy.random as rand
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split


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
    def __init__(self, interactions, human_feats, virus_feats, pct_test):
        self.interactions = interactions
        self.human_feats = human_feats
        self.virus_feats = virus_feats
        self.split_data(interactions, pct_test)

    def split_data(self, interactions, pct_test):
        X = self.interactions.drop(['edge'], axis=1).values
        y = self.interactions['edge'].values
        self.Xtrain, self.Xtest, self.yTrain, self.yTest = train_test_split(X,y, test_size=pct_test, random_state=42)
        self.Xtrain, self.Xval, self.yTrain, self.yVal = train_test_split(self.Xtrain, self.yTrain, test_size=.10, random_state=42)
    

    def create_train_loader(self, bs):
        return self.create_loader(self.Xtrain, self.yTrain, bs)
        
    def create_loader(self, dsetX, dsetY, bs):
        human_feats = []
        virus_feats = []
        Xpairs = [] # virus node id x human node id pair
        ys = []  # 1/0 interactions
        n_train = len(dsetX)

        for i in range(n_train):
            # original dset was 1 indexed, subtract 1 to offset
            v_nodeid, h_nodeid = dsetX[i][0] - 1, dsetX[i][1] - 1

            human_feats.append(torch.from_numpy(self.human_feats[h_nodeid, :]))
            virus_feats.append(torch.from_numpy(self.virus_feats[v_nodeid, :]))
            Xpairs.append( (v_nodeid, h_nodeid) )
            ys.append(dsetY[i])

        dset = TensorDataset(torch.from_numpy(np.asarray(Xpairs)),torch.stack(human_feats), torch.stack(virus_feats), 
            torch.from_numpy(np.asarray(ys)))

        return DataLoader(dset, batch_size=bs, shuffle=True)