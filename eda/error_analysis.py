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

path_data = PATH
traindf = pd.read_csv( os.path.join( path_data, 'train.csv'))#.iloc[:3000]
testdf  = pd.read_csv( os.path.join( path_data, 'test.csv'))
traincdf = pd.read_csv( os.path.join( path_data, 'train_controls.csv'))#.iloc[:3000]
testcdf = pd.read_csv( os.path.join( path_data, 'test_controls.csv'))#.iloc[:3000]

foldsdf = pd.read_csv( os.path.join( path_data, 'folds.csv'))
folds = dict((i,foldsdf[foldsdf['fold']==i]['experiment'].tolist()) for i in range(5))


probsvalls = []
for f in range(5):
    valdf = traindf[traindf['experiment'].isin(folds[f])]
    y_val = valdf['sirna'].values
    # Load predictions
    predval = loadobj(os.path.join( path_data, '../sub/weights/v31/val_probs_512_fold{}.pk'.format(f)))
    probsval =  hmean([i.clip(1e-40, 1) for i in predval])[:,:1108] 
    valdf = traindf[traindf['experiment'].isin(folds[f])]
    probsval = pd.DataFrame(probsval)
    probsval['id_code'] = valdf['id_code'].values
    probsval['sirna'] = valdf['sirna'].values
    probsvalls.append(probsval)
probsval = pd.concat(probsvalls, 0)
probsval = probsval.set_index('id_code').loc[traindf.id_code.values].reset_index()

probsval['sirna_pred'] = single_pred(traindf, probsval[range(1108)].values).astype(int)
traindf['sirna_pred'] = probsval['sirna_pred'] 



traindf['acc'] = (traindf['sirna_pred'] == traindf['sirna']).astype(np.int8)
traindf.head()
traindf['expt'] = traindf['experiment'].apply(lambda x: x.split('-')[0])
traindf['well1'] = traindf['well'].apply(lambda x: x[0])
traindf['well2'] = traindf['well'].apply(lambda x: int(x[1:]))
traindf.groupby(['expt'])['acc'].agg(['mean', 'count']).reset_index()
traindf.groupby(['experiment'])['acc'].mean().reset_index()

traindf.groupby(['sirna', 'expt'])['acc'].mean().unstack().to_csv('eda/charts/expt_sirna.csv')

traindf.groupby(['experiment', 'plate'])['acc'].mean().unstack().to_csv('eda/charts/err_plate.csv')
traindf.groupby(['sirna', 'expt'])['acc'].agg(['mean', 'count']).unstack().to_csv('eda/charts/err_sirna.csv')
traindf.groupby(['well2', 'well1'])['acc'].mean().unstack().to_csv('eda/charts/err_well.csv')


