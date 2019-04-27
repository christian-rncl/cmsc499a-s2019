import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

from data import ProteinInteractionGenerator
from utils import loadjson
from gmf import GMF

class GMFConfig_dbg:
    def __init__(self, device, n = 3, m = 3, prob = .6):
        self.device = device
        self.create_generator(n,m,prob)
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
        # print("params-------")
        # print(list(self.model.parameters()))
        # print("end-------")
        # print('grad: ', list(self.model.parameters())[0].grad)
        # print('grad: ', list(self.model.parameters())[3].grad)
        # print('grad: ', list(self.model.parameters())[4].grad)
        print('-' * 15, "Done with model", '-' * 15)
        print()

    def create_generator(self, n, m, prob):
        ############################
        ##  generate bipartite
        ########################### 
        print('-' * 15, "Generating graph", '-' * 15)
        G = nx.bipartite.random_graph(n, m, .30)
        observed = list(G.edges())
        nodes = list(G.nodes())
        virusUprot = []
        humanUprot = []
        edges = [] 


        for i in tqdm(range(len(nodes))):
            for j in tqdm(range(len(nodes))):
                virusUprot.append(i) 
                humanUprot.append(j)
                if (i, j) in observed:
                    edges.append(1.0)
                else:
                    edges.append(0.0)

        M = pd.DataFrame({'virusUprot': virusUprot, 
                            'humanUprot': humanUprot,
                            'edge': edges})


        htoi = {v:k for k,v in enumerate(M['humanUprot'].unique())}
        vtoi = {v:k for k,v in enumerate(M['virusUprot'].unique())}
        print('-' * 15, "Dataframe created", '-' * 15)
        print()

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