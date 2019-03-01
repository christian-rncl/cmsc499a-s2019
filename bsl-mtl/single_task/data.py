'''
Leiserson Research Group 2/28/2019
Author(s): Christian Roncal
'''
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

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
        self.pool = interactions.loc[interactions['edge'] == 1].index.values
        split_data()

    def split_data(self):
        data = []

        for i in self.pool():
            