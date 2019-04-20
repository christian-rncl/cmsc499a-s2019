import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

from data import ProteinInteractionGenerator
from utils import loadjson
from bilinearmf import BMF

class BMFConfig:
    def __init__(self, path, debug, device):
        self.device = device
        self.create_generator(path, debug)
        self.create_model()

    def create_model(self):
        print('-' * 15, "Creating model", '-' * 15)
        config = {
            'num_virus': self.n_v,
            'num_human': self.n_h,
            'vfeats': self.vfeats,
            'hfeats': self.hfeats,
            'latent_dim': self.latent_dim,
            'sparse': False # set false for now because some optimizers dont work with sparse
        }

        self.model = BMF(config)
        self.model.to(self.device)

        print(self.model)
        print('-' * 15, "Done with model", '-' * 15)
        print()

    def create_generator(self, path, debug):
        ############################
        ##   paths 
        ########################### 
        train_csv =f'{path}full_train.csv'
        vfeats_txt = f'{path}vfeats.txt'
        hfeats_txt = f'{path}hfeats.txt'
        htoi_json = f'{path}htoi.json'
        vtoi_json = f'{path}vtoi.json'

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

        print("loading features at: ", vfeats_txt, hfeats_txt)
        vfeats = np.loadtxt(vfeats_txt)
        hfeats = np.loadtxt(hfeats_txt)

        print("loading indices at: ", vtoi_json, htoi_json)
        htoi = loadjson(htoi_json)
        vtoi = loadjson(vtoi_json)
        print('-' * 15, "Finished loading data", '-' * 15)
        print()

        ############################
        ##   Prepare data (dataloader)
        ############################
        print('-' * 15, "Creating Generator", '-' * 15)

        data_config = {
            'interactions':M,
            'htoi':htoi,
            'vtoi':vtoi,
            'pct_test':.10,
            'device': self.device
        }

        self.vfeats = vfeats
        self.hfeats = hfeats
        self.n_v = len(vtoi)
        self.n_h = len(htoi)
        self.latent_dim = vfeats.shape[1]
        self.gen = ProteinInteractionGenerator(data_config)

        print('-' * 15, "Generator done", '-' * 15)
        print()

    def get_generator(self):
        return self.gen

    def get_model(self):
        return self.model