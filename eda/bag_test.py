import pandas as pd
import pickle
import numpy as np
import os
from tqdm import tqdm
from scipy.stats.mstats import hmean


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

valdf = traindf[traindf['experiment'].isin(fold0)]
y_val = valdf['sirna'].values

# Load predictions
predval = loadobj(os.path.join( path_data, '../sub/weights/v31/val_probs_512_fold0.pk'))
predtst = loadobj(os.path.join( path_data, '../sub/weights/v31/tst_probs_512_fold0.pk'))
print(len(predval))

for i in range(20):
    predmax = np.argmax((sum(predval[-i:])/len(predval[-i]))[:,:1108], 1)
    matchesbagmax = (predmax.flatten().astype(np.int32) == y_val.flatten().astype(np.int32)).sum() 
    print('Last {} Accuracy Bag Max: {:.4f}'.format(i, matchesbagmax/predmax.shape[0]))

for i in range( 20):
    predmax = np.argmax(hmean(predval[-i:])[:,:1108], 1)
    matchesbagmax = (predmax.flatten().astype(np.int32) == y_val.flatten().astype(np.int32)).sum() 
    print('Last {} Accuracy Bag Max: {:.4f}'.format(i, matchesbagmax/predmax.shape[0]))

# predsbag = np.argmax(probsbag, 1)
probsbag = hmean(predtst[-20:])[:,:1108] # (sum(predtst[-5:])/len(predtst[-5:]))[:,:1108]
#probsbag = sum(predtst)/len(predtst)
#probsbag = probsbag[:,:1108]
submission = pd.read_csv( os.path.join( path_data, 'test.csv'))
submission['sirna'] = single_pred(submission, probsbag).astype(int)
# submission['sirna'] = predsbag.astype(int)
submission.to_csv(os.path.join( path_data, 
                               '../sub/weights/v31/mixme_mean_v31_fold{}.csv.gz'.format(0)), 
                                index=False, columns=['id_code','sirna']
                                , compression = 'gzip')
probsdf = pd.DataFrame(probsbag)
probsdf['id_code'] = submission['id_code']
probsdf.to_csv(os.path.join( path_data, 
                               '../sub/weights/v31/probs_v31_fold{}.csv'.format(0)), 
                                index=False)
    
# check the sub
subold = pd.read_csv( os.path.join( path_data, '../sub/weights/v28/mixme_fold0.csv'))
subold1 = pd.read_csv( os.path.join( path_data, '../sub/mixme_fold0_v25_512.csv'))
sum(subold.sirna== subold1.sirna)/subold1.shape[0]

sum(subold.sirna== submission.sirna)/submission.shape[0]

