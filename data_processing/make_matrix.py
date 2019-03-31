# Christian Roncal cmsc499a Dr.Leiserson 03/31/19

import multiprocessing as mp
import os 
import pandas as pd
import numpy as np
import argparse


def read_interaction_csvs(interaction_csvs):
    '''
        creates dataframes rom interaction csvs, if multiple merges.
    '''
    if len(interaction_csvs > 1):
        print('Merging files....')
        dfs = [pd.read_csv(fname) for fname in interaction_csvs]
        return pd.concat(dfs).reset_index()
    else: 
        return pd.read_csv(interaction_files[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nworkers', default=4, help="number of workers in map pool")
    parser.add_argument('-i', '--interactions', action='append', help='<Required> Set flag', required=True)
    
    args = parser.parse_args()
    interaction_csvs = args.interactions
    nworkers = args.nworkers

    df = read_interaction_csvs(interaction_csvs)




