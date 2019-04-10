import pandas as pd
import numpy as np
import chardet
import re
import multiprocessing as mp

#### Data loading

'''
https://stackoverflow.com/questions/33819557/unicodedecodeerror-utf-8-codec-while-reading-a-csv-file
# for some reason phisto csvs can't easily be loaded, this loads it with the right encoding 
# and overwrites the original as per the stack overflow post
'''
def fix_and_save_csv(fname):
    assert(fname[-4:] == '.csv')

    with open(fname, 'rb') as f:
        result = chardet.detect(f.readline())  # or readline if the file is large

    # print(result['encoding'])
    df = pd.read_csv(fname, encoding='cp1252') # for hepc task only

    df.to_csv(fname)


#### PROCESSING GENERAL


'''
given a dataframe of interactions,
with columns v_idx (virus indices) and h_idx (human indices), 
for all possible unique (virus,human)  pairs apply 
@param worker_fn to all possible (v, h) pairs to get an iterable of dataframes
then apply @ret_fn to iterable of dataframes to get a single dataframe.

primarily used in ranking dataframes by observed, and creating the matrix representation
for interactions.
'''
def mp_processpairs(df, worker_fn, ret_fn, nworkers):
    virus_idxs = df['virusUprot'].values
    human_idxs = df['humanUprot'].values
    
    # get all observed pairs
    pairs = [(v, h) for v, h in zip(virus_idxs, human_idxs)]

    virus_idxs_uniq = df['virusUprot'].unique()
    human_idxs_uniq = df['humanUprot'].unique()

    pool = mp.Pool(nworkers)

    results = pool.starmap(worker_fn, 
                           [(v, human_idxs_uniq, pairs) for v in virus_idxs_uniq], 15)
    
    pool.close()
    pool.join()
    return ret_fn(results)

'''

'''
# the worker, counts interactions
def mp_interaction_counter(v, human_idxs, pairs):
    n_pos = 0
    n_neg = 0

    for h in human_idxs:
        if (v, h) in pairs:
            n_pos += 1
        else:
            n_neg += 1
    
    counts = {'virus':[v], 'n_pos':[n_pos], 'n_neg':[n_neg], 'ratio':[n_pos / (n_pos + n_neg)]}
    return pd.DataFrame(counts)

# ret_fn
def retfn(results):
    # drop(columns=['index'])
    # # return pd.concat(d).drop(columns=['index']).reset_index()
    try:
        return pd.concat(results).sort_values(by='n_pos', ascending=False).reset_index().drop(columns=['index'])
    except:
        print('no index column for some reason... ')
        return pd.concat(results).sort_values(by='n_pos', ascending=False).reset_index()


def rank_by_interactions(df, nworkers):
    return mp_processpairs(df, mp_interaction_counter, retfn, nworkers)

###### DF SELECTION
'''
select rows from @param df matching the @regex on @column
'''
def regex_select(df, column, regex):

    def regex_selector(name):
        if type(name) != str: return False
        return bool(re.search(regex, name))

    print(column, regex)
    return df[df[column].apply(regex_selector)]

def removeObsoletes(df):
    obsoletes = ['P08107']
    
    return df[df.humanUprot != 'P08107']