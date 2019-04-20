import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

from data import ProteinInteractionGenerator
from utils import loadjson
from gmf import GMF

class GMFConfig:
    def __init__(self, path, debug, device):
        self.device = device
        self.create_generator(path, debug)
        self.create_model()

    def create_model(self):

        print('-' * 15, "Creating model", '-' * 15)

        latent_dim = 2799
        config = {
            'num_virus': self.n_v,
            'num_human': self.n_h,
            'latent_dim': latent_dim,
            'sparse': False # set false for now because some optimizers dont work with sparse
        }

        self.model = GMF(config)
        self.model.to(self.device)

        print(self.model)
        print('-' * 15, "Done with model", '-' * 15)
        print()

    def create_generator(self, path, debug):
        ############################
        ##   paths 
        ########################### 
        train_csv =f'{path}full_train.csv'

        ############################
        ##   Load data
        ############################
        print('-' * 15, "Loading data", '-' * 15)

        print("loading traning matrix at: ", train_csv)
        M = pd.read_csv(train_csv)

        if debug:
            print("Making debug dataset.....")
            pos = M.loc[M['edge'] > 0].sample(frac=1)
            negs = M.loc[M['edge'] == 0].sample(frac=1)
            M = pd.concat([pos, negs[:len(pos)]], ignore_index=True).sample(frac=1)

        htoi = {v:k for k,v in enumerate(M['humanUprot'].unique())}
        vtoi = {v:k for k,v in enumerate(M['virusUprot'].unique())}

        ############################
        ##   Prepare data (dataloader)
        ############################
        print('-' * 15, "Creating data loaders", '-' * 15)

        data_config = {
            'interactions':M,
            'htoi':htoi,
            'vtoi':vtoi,
            'pct_test':.10,
            'device': self.device
        }

        self.n_v = len(vtoi)
        self.n_h = len(htoi)
        self.gen = ProteinInteractionGenerator(data_config)

        print('-' * 15, "Generator done", '-' * 15)
        print()

    def get_generator(self):
        return self.gen

    def get_model(self):
        return self.model

