import pandas as pd
import pickle
import numpy as np
import os
from tqdm import tqdm
from scipy.stats.mstats import hmean
import seaborn as sns
from scipy.special import softmax
import torch

def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

PATH = '/Users/dhanley2/Documents/Personal/recursion/data'
os.chdir(os.path.join(PATH, '..'))
      
from logs import get_logger
from utils import dumpobj, loadobj, GradualWarmupScheduler, single_pred


traindf = pd.read_csv( os.path.join( PATH, 'train.csv'))#.iloc[:3000]
testdf  = pd.read_csv( os.path.join( PATH, 'test.csv'))
train_ctrl = pd.read_csv( os.path.join( PATH, 'train_controls.csv'))#.iloc[:3000]
test_ctrl = pd.read_csv( os.path.join( PATH, 'test_controls.csv'))#.iloc[:3000]




samp = traindf[traindf['experiment']=='HEPG2-01']
samp['mode'] = 'train'

def _get_np_path(mydir, index, records, site = 1):
    experiment, well, plate, mode = records[index].experiment, \
                                    records[index].well, \
                                    records[index].plate, \
                                    records[index].mode
    return '/'.join([mydir,mode,experiment,f'Plate{plate}',f'{well}_s{site}_w.pk'])

loaddir = os.path.join(PATH, '256X256X6')
recs = samp.to_records(index=False)
dfls = []
for t in range(samp.shape[0]):
    rec = _get_np_path(loaddir, t, recs)
    arr = loadobj(rec)
    dfls.append( pd.DataFrame(arr.reshape(-1, arr.shape[-1])))
df = pd.concat(dfls, 0)
df.hist(figsize = (10,6), bins = 50)
pd.DataFrame(np.log(df)).hist(figsize = (10,6), bins = 50)
