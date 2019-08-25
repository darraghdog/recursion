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

folds = pd.read_csv( os.path.join( path_data, 'folds.csv'))
fold0 = folds[folds['fold']==0]['experiment'].tolist()

valdf = traindf[traindf['experiment'].isin(fold0)]
y_val = valdf['sirna'].values

# Load predictions
predval = loadobj(os.path.join( path_data, '../sub/weights/v28/val_probs_512__fold0.pk'))
predtst = loadobj(os.path.join( path_data, '../sub/weights/v28/tst_probs_512__fold0.pk'))
print(len(predval))

for i in range(20):
    predmax = np.argmax((sum(predval[-i:])/len(predval[-i]))[:,:1108], 1)
    matchesbagmax = (predmax.flatten().astype(np.int32) == y_val.flatten().astype(np.int32)).sum() 
    print('Last {} Accuracy Bag Max: {:.4f}'.format(i, matchesbagmax/predmax.shape[0]))

for i in range( 20):
    predmax = np.argmax(hmean(predval[-i:])[:,:1108], 1)
    matchesbagmax = (predmax.flatten().astype(np.int32) == y_val.flatten().astype(np.int32)).sum() 
    print('Last {} Accuracy Bag Max: {:.4f}'.format(i, matchesbagmax/predmax.shape[0]))

i = 20
predmax = np.median(np.array([np.argmax(predval[-ii][:,:1108], 1) for ii in range(i)]), 0)
matchesbagmax = (predmax.flatten().astype(np.int32) == y_val.flatten().astype(np.int32)).sum() 
print('Last {} Accuracy Bag Max: {:.4f}'.format(i, matchesbagmax/predmax.shape[0]))



for i in range(1, 10):
    predmax = np.argmax((sum(predval[-(20+i):(-i)])/len(predval[-(20+i):(-i)]))[:,:1108], 1)
    matchesbagmax = (predmax.flatten().astype(np.int32) == y_val.flatten().astype(np.int32)).sum() 
    print('Last {} Accuracy Bag Max: {:.4f}'.format(i, matchesbagmax/predmax.shape[0]))


'''
Check leak
'''
# predsbag = np.argmax(probsbag, 1)
probsbag = hmean(predtst[-20:])[:,:1108] # (sum(predtst[-5:])/len(predtst[-5:]))[:,:1108]
#probsbag = sum(predtst)/len(predtst)
#probsbag = probsbag[:,:1108]
sub = pd.read_csv( os.path.join( path_data, 'test.csv'))
sub['sirna'] = single_pred(sub, probsbag).astype(int)


# Look at train groups
plate_groups = np.zeros((1108,4), int)
for sirna in range(1108):
    grp = traindf.loc[traindf.sirna==sirna,:].plate.value_counts().index.values
    assert len(grp) == 3
    plate_groups[sirna,0:3] = grp
    plate_groups[sirna,3] = 10 - grp.sum()
dfhmtrn = plate_groups
dfhmtrn[:2]


# Look at test groups
all_test_exp = testdf.experiment.unique()
group_plate_probs = np.zeros((len(all_test_exp),4))
for idx in range(len(all_test_exp)):
    preds = sub.loc[testdf.experiment == all_test_exp[idx],'sirna'].values
    pp_mult = np.zeros((len(preds),1108))
    pp_mult[range(len(preds)),preds] = 1
    sub_test = testdf.loc[testdf.experiment == all_test_exp[idx],:]
    assert len(pp_mult) == len(sub_test)
    for j in range(4):
        mask = np.repeat(plate_groups[np.newaxis, :, j], len(pp_mult), axis=0) == \
               np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
        group_plate_probs[idx,j] = np.array(pp_mult)[mask].sum()/len(pp_mult)
dfhmtst = pd.DataFrame(group_plate_probs, index = all_test_exp)
sns.heatmap(dfhmtst, annot=True)


# Get test assignments
exp_to_group = group_plate_probs.argmax(1)

tstgrpdf = pd.DataFrame(exp_to_group, index = all_test_exp, columns = ['group'])
tstgrpdf.head()


# Create a mask to zero out values
def select_plate_group(pp_mult, idx):
    sub_test = testdf.loc[testdf.experiment == all_test_exp[idx],:]
    assert len(pp_mult) == len(sub_test)
    mask = np.repeat(plate_groups[np.newaxis, :, exp_to_group[idx]], len(pp_mult), axis=0) != \
           np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
    pp_mult[mask] = 0
    return pp_mult

# Mask out the values
predicted = probsbag.copy()
probsleak = probsbag.copy()

for t, idx in enumerate(range(len(all_test_exp))):
    print(all_test_exp[idx])
    indices = (testdf.experiment == all_test_exp[idx])    
    preds = predicted[indices,:].copy()    
    probsleak[indices] = select_plate_group(preds, idx)

# predsbag = np.argmax(probsbag, 1)
submission = pd.read_csv( os.path.join( path_data, 'test.csv'))
submission['sirna'] = single_pred(submission, probsleak).astype(int)
# submission['sirna'] = predsbag.astype(int)
submission.to_csv(os.path.join( path_data, 
                               '../sub/weights/v28/mixme_hmean_v28_fold{}_leak.csv.gz'.format(0)), 
                                index=False, columns=['id_code','sirna']
                                , compression = 'gzip')

# check the sub
subold = pd.read_csv( os.path.join( path_data, \
                        '../sub/weights/v28/mixme_hmean_v28_fold{}_leak.csv.gz'.format(0)))
subold1 = pd.read_csv( os.path.join( path_data, '../sub/weights/v28/mixme_hmean_v28_fold0.csv.gz'))
sum(subold.sirna== subold1.sirna)/subold1.shape[0]

sum(subold.sirna== submission.sirna)/submission.shape[0]

