# Christian Roncal cmsc499a Dr.Leiserson 03/31/19
import multiprocessing as mp
import os 
import pandas as pd
import numpy as np
import argparse
from utils import mp_processpairs


'''
    creates dataframes rom interaction csvs, if multiple merges.
'''
def read_interaction_csvs(interaction_csvs):
    if len(interaction_csvs) > 1:
        print('Merging files....')
        dfs = [pd.read_csv(fname) for fname in interaction_csvs]
        return pd.concat(dfs).reset_index()
    else: 
        return pd.read_csv(interaction_csvs[0])

'''
    worker for mp_processpairs
    given a dict with single virus index (v), and a list of UNIQUE human indices (hidxs) and 
    observed pairs (pairs) return a dataframe wit cols: vidx, hidx, edge(0/1)
'''
def mp_pairmatcher(v, human_idxs, pairs):

    d = {'virusUprot':[], 'humanUprot':[], 'edge':[]}
    
    for h in human_idxs:
        d['virusUprot'].append(v)
        d['humanUprot'].append(h)
        
        if (v, h) in pairs:
            d['edge'].append(1.0)
        else:
            d['edge'].append(0.0)
    
    return pd.DataFrame(d)

def retfn(results):
    # try:
    return pd.concat(results, ignore_index=True)
    # except:
    #     return pd.concat(results).reset_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nworkers', default=4, help="number of workers in map pool")
    parser.add_argument('-i', '--interactions', action='append', help='<Required> Set flag', required=True)
    parser.add_argument('-o', '--outputname', help="output of filename", required=True)
    
    args = parser.parse_args()
    interaction_csvs = args.interactions
    nworkers = args.nworkers
    outputname = args.outputname

    df = read_interaction_csvs(interaction_csvs)
    df = mp_processpairs(df, mp_pairmatcher, retfn, nworkers)
    df.to_csv(outputname)



