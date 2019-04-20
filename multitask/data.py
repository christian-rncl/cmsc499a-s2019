'''
Author(s): Christian Roncal
Leiserson Research Group 2/28/2019
'''
import torch
import pandas as pd
import numpy as np
import numpy.random as rand
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split

'''
    Single task experiments data generator,
    Given params interaction matrix, human/virus features for a single task, 
    process data and split to train/cv/test based on settings 
'''
class ProteinInteractionGenerator(object):
    '''
    interactions: df representing interactions
    human: np array 
    '''
    # def __init__(self, interactions, human_feats, virus_feats, htoi, vtoi, pct_test, device):
    def __init__(self, config):
        self.htoi = config['htoi']
        self.vtoi = config['vtoi']
        # index interactions based on htoi and vtoi
        self.index_interactions(config['interactions'])
        # # self.device = torch.device('cuda') if cuda else torch.device('cpu')
        self.device = config['device']
        print('using device: ', self.device)

        self.split_data(config['pct_test'])

    def split_data(self, pct_test):
        # X = self.interactions.drop(['virusUprot', 'humanUprot'], axis=1).values
        X = self.interactions[['virusUprot', 'humanUprot']]
        y = self.interactions['edge']
        n_pos = len(y[y > 0])
        n_neg = len(y) - n_pos
        print(f'Found {n_pos} positives, and {n_neg} negatives!', n_pos/len(y))

        self.Xtrain, self.Xtest, self.yTrain, self.yTest = train_test_split(X,y, test_size=pct_test, random_state=42)
        self.Xtrain, self.Xval, self.yTrain, self.yVal = train_test_split(self.Xtrain, self.yTrain, test_size=.10, random_state=42)

        train_pos = len(self.yTrain[self.yTrain > 0])
        val_pos = len(self.yVal[self.yVal > 0])
        test_pos = len(self.yTest[self.yTest > 0])

        print(8 * '-')
        print(f'{train_pos} in training set, {train_pos/len(self.yTrain)}')
        print(f'{val_pos} in val set, {train_pos/len(self.yVal)}')
        print(f'{test_pos} in test set, {test_pos/len(self.yTest)}')
    
    def index_interactions(self, interactions):
        self.interactions = interactions
        self.interactions['virusUprot'] = self.interactions['virusUprot'].apply(lambda x: self.vtoi[x])
        self.interactions['humanUprot'] = self.interactions['humanUprot'].apply(lambda x: self.htoi[x])


    def create_train_loader(self, bs):
        return self.create_loader(self.Xtrain, self.yTrain, bs)

    def create_val_loader(self, bs):
        return self.create_loader(self.Xval, self.yVal, bs)

    def create_test_loader(self, bs):
        return self.create_loader(self.Xtest, self.yTest, bs)

    def create_debug_loader(self, bs):
        return self.create_loader(self.Xtest[:10], self.yVal[:10], bs)
        
    def create_loader(self, dsetX, dsetY, bs):
        virus_idxs = dsetX['virusUprot'].values
        human_idxs = dsetX['humanUprot'].values
        ys = torch.from_numpy(dsetY.values).to(self.device, dtype=torch.float32)


        dset = TensorDataset(
            torch.from_numpy(virus_idxs).to(self.device, dtype=torch.long),
            torch.from_numpy(human_idxs).to(self.device, dtype=torch.long),
            ys.view(ys.shape[0], 1)
            )
        
        return DataLoader(dset, batch_size=bs, shuffle=True)


