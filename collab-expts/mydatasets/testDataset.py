import torch
import pandas as pd
from torch.utils import data

class TestData(data.Dataset):

    def __init__(self, df):
        print(df.head())
        nonzero_entries = df.loc[df['m_IJ'] > 0]
        print(nonzero_entries)