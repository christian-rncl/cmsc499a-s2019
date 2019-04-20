import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

from data import ProteinInteractionGenerator
from utils import loadjson
from bilinearmf import BMF

############################
##   paths & settings
############################
path = './data/'
train_csv =f'{path}full_train.csv'
vfeats_txt = f'{path}vfeats.txt'
hfeats_txt = f'{path}hfeats.txt'
htoi_json = f'{path}htoi.json'
vtoi_json = f'{path}vtoi.json'

### General settings
BS = 64
# device = 'cpu'
device = 'cuda'
DEBUG = True
epochs = 3

### Tensorboard settings 
log_dir = './logs'
log_interval = 10

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
latent_dim = vfeats.shape[1]

config = {
    'num_virus': n_v,
    'num_human': n_h,
    'vfeats':vfeats,
    'hfeats':hfeats,
    'latent_dim': latent_dim,
    'sparse': False # set false for now because some optimizers dont work with sparse
}

model = BMF(config)
model.to(device)
print(model)

print('-' * 15, "Done with model", '-' * 15)
print()