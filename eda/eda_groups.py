import pandas as pd
import numpy as np
import os
PATH = '/Users/dhanley2/Documents/Personal/recursion/data'

trndf = pd.read_csv(os.path.join(PATH, 'train.csv'))
tstdf = pd.read_csv(os.path.join(PATH, 'test.csv'))
trnctrldf = pd.read_csv(os.path.join(PATH, 'train_controls.csv'))
tstctrldf = pd.read_csv(os.path.join(PATH, 'test_controls.csv'))
statsdf = pd.read_csv(os.path.join(PATH, 'stats.csv'))




trndf[['experiment', 'plate']].groupby('sirna')
