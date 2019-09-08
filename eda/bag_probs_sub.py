import pandas as pd
import pickle
import numpy as np
import os
from tqdm import tqdm
from scipy.stats.mstats import hmean
import seaborn as sns



def single_pred(dffold, probs):
    pred_df = dffold[['id_code','experiment']].copy()
    exps = pred_df['experiment'].unique()
    pred_df['sirna'] = 0
    for exp in tqdm(exps):
        preds1 = probs[pred_df['experiment'] == exp]
        done = []
        sirna_r = np.zeros((1108),dtype=int)
        for a in np.argsort(np.reshape(preds1,-1))[::-1]:
            ind = np.unravel_index(a, (preds1.shape[0], 1108), order='C')
            if not ind[1] in done:
                if not ind[0] in sirna_r:
                    sirna_r[ind[1]]+=ind[0]
                    #print([ind[1]]​)
                    done+=[ind[1]]
        preds2 = np.zeros((preds1.shape[0]),dtype=int)
        for i in range(len(sirna_r)):
            preds2[sirna_r[i]] = i
        pred_df.loc[pred_df['experiment'] == exp,'sirna'] = preds2
    return pred_df.sirna.values

def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

PATH = '/Users/dhanley2/Documents/Personal/recursion/data'
path_data = PATH
traindf = pd.read_csv( os.path.join( path_data, 'train.csv'))#.iloc[:3000]
testdf  = pd.read_csv( os.path.join( path_data, 'test.csv'))
traincdf = pd.read_csv( os.path.join( path_data, 'train_controls.csv'))#.iloc[:3000]
testcdf = pd.read_csv( os.path.join( path_data, 'test_controls.csv'))#.iloc[:3000]
huvec18df = pd.read_csv( os.path.join( path_data, 'huvec18.csv'))#.iloc[:3000]

valdf = huvec18df
y_val = valdf['sirna'].values

# Load predictions
predval = loadobj(os.path.join( path_data, '../sub/tts/val_probs_256_fold5.pk'))
predtst = loadobj(os.path.join( path_data, '../sub/tts/tst_probs_256_fold5.pk'))
print(len(predval))

for i in range(10):
    predmax = np.argmax(hmean(predval[-5:])[:,:1108], 1)
    matchesbagmax = (predmax.flatten().astype(np.int32) == y_val.flatten().astype(np.int32)).sum() 
    print('Last {} Accuracy Bag Max: {:.4f}'.format(i, matchesbagmax/predmax.shape[0]))

probsdf = pd.DataFrame(hmean(predtst[-5:])[:,:1108])
probsdf['id_code'] = testdf['id_code']
probsdf.to_csv(os.path.join( path_data, '../sub/weights/v31/probs_v31_256_fold5.csv'), index=False)

'''
subproc = pd.read_csv('~/Downloads/sub_v36_fold5.csv')
(subproc.sirna==huvec18df.sirna).mean()
'''