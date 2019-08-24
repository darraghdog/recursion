import pandas as pd
import pickle
import numpy as np
import os

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

folds = pd.read_csv( os.path.join( path_data, 'folds.csv'))
fold0 = folds[folds['fold']==0]['experiment'].tolist()
onecycdf = pd.read_csv( os.path.join( path_data, '../one_cycle.csv'))#.iloc[:3000]
onecycdf = onecycdf.iloc[:,1:]
onecycdf = onecycdf.set_index('lr_log10')
onecycdf.head()

onecycdf.reset_index(inplace=True)
onecycdf = onecycdf.pivot_table('losses', ['lr_log10'], 'weight_decay')
onecycdf.plot.line()
onecycdf.plot.line(ylim = (6,7), xlim = (-5, -1), figsize = (10,10))
onecycdf.rolling(20).mean().plot.line(ylim = (6,7), figsize = (10,10))

onecycdf.head()


onecycdf['weight_decay'] = 0.01
onecycdf1 = onecycdf.copy()
onecycdf1['weight_decay'] = 0.001
onecycdf2 = onecycdf.copy()
onecycdf2['weight_decay'] = 0.0001
ocdf = pd.concat([onecycdf, onecycdf1, onecycdf2])
ocdf['weight_decay'] = 'wd_'+ocdf['weight_decay'].astype(str)
ocdf.reset_index(inplace=True)
ocdf.pivot_table('losses', ['lr_log10'], 'weight_decay')
#onecycdf['min_loss'] = onecycdf['losses'].min()