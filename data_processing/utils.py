import pandas as pd
import numpy as np
import chardet

# https://stackoverflow.com/questions/33819557/unicodedecodeerror-utf-8-codec-while-reading-a-csv-file
# for some reason phisto csvs can't easily be loaded, this loads it with the right encoding 
# and overwrites the original as per the stack overflow post
def fix_and_save_csv(fname):
    assert(fname[-4:] == '.csv')

    with open(fname, 'rb') as f:
        result = chardet.detect(f.read())  # or readline if the file is large

    df = pd.read_csv(fname, encoding=result['encoding'])

    df.to_csv(fname)
