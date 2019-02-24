import torch
import pandas as pd
from torch.utils import data

class TestData(data.Dataset):

    def __init__(self, df):
        rows, cols = df.as_matrix(columns=df['row']), df.as_matrix(columns=df['col'])

        print(rows, cols)