import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

import os
PATH = '/Users/dhanley2/Documents/Personal/recursion/data'
path_data = PATH
traindf = pd.read_csv( os.path.join( path_data, 'train.csv'))#.iloc[:3000]
testdf  = pd.read_csv( os.path.join( path_data, 'test.csv'))
train_ctrl = pd.read_csv( os.path.join( path_data, 'train_controls.csv'))#.iloc[:3000]
test_ctrl = pd.read_csv( os.path.join( path_data, 'test_controls.csv'))#.iloc[:3000]

lb1 = pd.read_csv( os.path.join( path_data, '../sub/lb/probs_v31_512_allfolds_leak_doodle.csv'))
lb2 = pd.read_csv( os.path.join( path_data, '../sub/lb/submit_32_TTActrls_leak_postproc.csv'))
lb3 = pd.read_csv( os.path.join( path_data, '../sub/lb/submit_35_leak_postproc.csv'))

sirna_matches= lb1[(lb1.sirna==lb2.sirna)].sirna.value_counts()
full_sirna_matches = sirna_matches[(sirna_matches>=18)].index.values
testdf['experiment'].unique().shape
idx = (lb1.sirna==lb2.sirna) & (lb1.sirna.isin(full_sirna_matches))
sublearn = testdf[idx].reset_index(drop=True)
sublearn['sirna'] = lb1.sirna[idx].values
sublearn['sirna'].hist(bins = 200, figsize=(10,5))

sublearn.to_csv( os.path.join( path_data, 'sublearn.csv'), index = False)