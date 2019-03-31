import pandas as pd
import numpy as np
import chardet
import re
import multiprocessing as mp


# https://stackoverflow.com/questions/33819557/unicodedecodeerror-utf-8-codec-while-reading-a-csv-file
# for some reason phisto csvs can't easily be loaded, this loads it with the right encoding 
# and overwrites the original as per the stack overflow post
def fix_and_save_csv(fname):
    assert(fname[-4:] == '.csv')

    with open(fname, 'rb') as f:
        result = chardet.detect(f.read())  # or readline if the file is large

    df = pd.read_csv(fname, encoding=result['encoding'])

    df.to_csv(fname)


'''
given a dataframe of interactions,
with columns v_idx (virus indices) and h_idx (human indices), 
for all possible unique (virus,human)  pairs apply 
@param worker_fn to all possible (v, h) pairs to get an iterable of dataframes
then apply @ret_fn to iterable of dataframes to get a single dataframe.

primarily used in ranking dataframes by observed, and creating the matrix representation
for interactions.
'''
def mp_processpairs(df, worker_fn, ret_fn):
    virus_idxs = df['v_idx'].values
    human_idxs = df['h_idx'].values
    
    # get all observed pairs
    pairs = [(v, h) for v, h in zip(virus_idxs, human_idxs)]

    virus_idxs_uniq = df['v_idx'].unique()
    human_idxs_uniq = df['h_idx'].unique()

    pool = mp.Pool(20)
    results = pool.starmap(worker_fn, 
                           [(v, human_idxs_uniq, pairs) for v in virus_idxs_uniq], 15)
    
    return ret_fn(results)


def regex_select(df, column, regex):

    def regex_selector(name):
        if type(name) != str: return False
        return bool(re.search(regex, name))

    print(column, regex)
    return df[df[column].apply(regex_selector)]

def rank_