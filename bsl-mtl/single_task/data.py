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
        # index the dataset such that node1 and node2 are continuous
        self.index_interactions(interactions)
        self.interactions = interactions
        self.human_feats = human_feats
        self.virus_feats = virus_feats
        self.split_data(self.indexed_interactions, pct_test)

    def split_data(self, interactions, pct_test):
        X = self.interactions.drop(['edge'], axis=1).values
        y = self.interactions['edge'].values
        self.Xtrain, self.Xtest, self.yTrain, self.yTest = train_test_split(X,y, test_size=pct_test, random_state=42)
        self.Xtrain, self.Xval, self.yTrain, self.yVal = train_test_split(self.Xtrain, self.yTrain, test_size=.10, random_state=42)
    
    def index_interactions(self, interactions):
        virus_idxs = sorted(list(interactions['node1'].unique()))
        human_idxs = sorted(list(interactions['node2'].unique()))

        self.vtoi = {v : i for i, v in enumerate(virus_idxs)}
        self.itov = {i : v for i, v in enumerate(virus_idxs)}
        self.htoi = {h : i for i, h in enumerate(human_idxs)}
        self.itoh = {i : h for i, h in enumerate(human_idxs)}

        self.indexed_interactions = interactions
        self.indexed_interactions['node1'] = self.indexed_interactions['node1'].apply(lambda x: self.vtoi[x])
        self.indexed_interactions['node2'] = self.indexed_interactions['node2'].apply(lambda x: self.htoi[x])


    def create_train_loader(self, bs):
        return self.create_loader(self.Xtrain, self.yTrain, bs)

    def create_val_loader(self, bs):
        return self.create_loader(self.Xval, self.yVal, bs)

    def create_test_loader(self, bs):
        return self.create_loader(self.Xtest, self.yTest, bs)

    def create_debug_loader(self, bs):
        return self.create_loader(self.Xtest[:10], self.yVal[:10], bs)
        
    def create_loader(self, dsetX, dsetY, bs):
        human_feats = []
        virus_feats = []
        Xpairs = [] # virus node id x human node id pair
        ys = []  # 1/0 interactions
        n_train = len(dsetX)

        for i in range(n_train):
            v_nodeid, h_nodeid = dsetX[i][0], dsetX[i][1]

            human_feats.append(torch.from_numpy(self.human_feats[h_nodeid, :]))
            virus_feats.append(torch.from_numpy(self.virus_feats[v_nodeid, :]))
            Xpairs.append( (v_nodeid, h_nodeid) )
            ys.append(dsetY[i])

        dset = TensorDataset(torch.from_numpy(np.asarray(Xpairs)),torch.stack(human_feats), torch.stack(virus_feats), 
            torch.from_numpy(np.asarray(ys)))

        return DataLoader(dset, batch_size=bs, shuffle=True)