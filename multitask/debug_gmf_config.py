import pandas as pd
import numpy as np
import networkx as nx

from data import ProteinInteractionGenerator
from utils import loadjson
from bilinearmf import BMF

class GMFConfig_dbg:
    def __init__(self, device, n = 500, m = 100, prob = .50):
        self.device = device
        self.create_generator(n,m,prob)
        # self.create_model()

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

    def create_generator(self, n, m, prob):
        ############################
        ##  generate bipartite
        ########################### 
        print('-' * 15, "Generating graph", '-' * 15)
        G = nx.bipartite.random_graph(n, m, .5)
        observed = list(G.edges())
        nodes = list(G.nodes())
        virusUprot = []
        humanUprot = []
        edges = [] 

        for i in range(len(nodes)):
            for j in range(len(nodes)):
                virusUprot.append(i) 
                humanUprot.append(j)
                if (i, j) in observed:
                    edges.append(1.0)
                else:
                    edges.append(0.0)

        M = pd.DataFrame({'viusUprot': virusUprot, 
                            'humanUprot': humanUprot,
                            'edges': edges})

        print('-' * 15, "Dataframe created", '-' * 15)

        htoi = loadjson(htoi_json)
        vtoi = loadjson(vtoi_json)
        print()

        # ############################
        # ##   Prepare data (dataloader)
        # ############################
        # print('-' * 15, "Creating Generator", '-' * 15)

        # data_config = {
        #     'interactions':M,
        #     'htoi':htoi,
        #     'vtoi':vtoi,
        #     'pct_test':.10,
        #     'device': self.device
        # }

        # self.vfeats = vfeats
        # self.hfeats = hfeats
        # self.n_v = len(vtoi)
        # self.n_h = len(htoi)
        # self.latent_dim = vfeats.shape[1]
        # self.gen = ProteinInteractionGenerator(data_config)

        # print('-' * 15, "Generator done", '-' * 15)
        # print()

    def get_generator(self):
        return self.gen

    def get_model(self):
        return self.model