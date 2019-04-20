import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

from data import ProteinInteractionGenerator
from utils import loadjson
# from bilinearmf import BMF
from gmf import GMF

############################
##   paths & settings
############################
train_csv =f'{path}full_train.csv'

############################
##   Load data
############################
print('-' * 15, "Loading data", '-' * 15)

print("loading traning matrix at: ", train_csv)
M = pd.read_csv(train_csv)

if DEBUG:
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
    'device': device
}

gen = ProteinInteractionGenerator(data_config)

train_loader = gen.create_train_loader(BS)
val_loader = gen.create_val_loader(BS)
test_loader = gen.create_test_loader(BS)

print('-' * 15, "Data loaders done", '-' * 15)
print()

############################
##  BMF Model
############################
print('-' * 15, "Creating model", '-' * 15)

n_v, n_h = len(vtoi), len(htoi)
latent_dim = 2799 

config = {
    'num_virus': n_v,
    'num_human': n_h,
    'latent_dim': latent_dim,
    'sparse': False # set false for now because some optimizers dont work with sparse
}

model = GMF(config)
model.to(device)
print(model)

print('-' * 15, "Done with model", '-' * 15)
print()