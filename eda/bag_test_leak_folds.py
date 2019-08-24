import pandas as pd
import pickle
import numpy as np
import os
from tqdm import tqdm
from scipy.stats.mstats import hmean
import seaborn as sns
from scipy.special import softmax

def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

PATH = '/Users/dhanley2/Documents/Personal/recursion/data'
path_data = PATH
traindf = pd.read_csv( os.path.join( path_data, 'train.csv'))#.iloc[:3000]
testdf  = pd.read_csv( os.path.join( path_data, 'test.csv'))
traincdf = pd.read_csv( os.path.join( path_data, 'train_controls.csv'))#.iloc[:3000]
testcdf = pd.read_csv( os.path.join( path_data, 'test_controls.csv'))#.iloc[:3000]

foldsdf = pd.read_csv( os.path.join( path_data, 'folds.csv'))
folds = dict((i,foldsdf[foldsdf['fold']==i]['experiment'].tolist()) for i in range(5))


for f in range(5):
    valdf = traindf[traindf['experiment'].isin(folds[f])]
    y_val = valdf['sirna'].values
    
    # Load predictions
    predval = loadobj(os.path.join( path_data, '../sub/weights/v31/val_probs_512_fold{}.pk'.format(f)))
    predtst = loadobj(os.path.join( path_data, '../sub/weights/v31/tst_probs_512_fold{}.pk'.format(f)))
    print(y_val.shape)
    print(predval[0].shape)
    for i in range( 21):
        predmax = np.argmax(hmean([i.clip(1e-40, 1) for i in predval[-i:]])[:,:1108], 1)
        matchesbagmax = (predmax.flatten().astype(np.int32) == y_val.flatten().astype(np.int32)).sum() 
        print('Last {} Accuracy Bag Max: {:.4f}'.format(i, matchesbagmax/predmax.shape[0]))

probststls = []
for f in range(5):
    valdf = traindf[traindf['experiment'].isin(folds[f])]
    y_val = valdf['sirna'].values
    # Load predictions
    predval = loadobj(os.path.join( path_data, '../sub/weights/v31/val_probs_512_fold{}.pk'.format(f)))
    predtst = loadobj(os.path.join( path_data, '../sub/weights/v31/tst_probs_512_fold{}.pk'.format(f)))
    probsval =  hmean([i.clip(1e-40, 1) for i in predval])[:,:1108] 
    probststls.append( hmean([i.clip(1e-40, 1) for i in predtst])[:,:1108] )
    valdf = traindf[traindf['experiment'].isin(folds[f])]
    probsval = pd.DataFrame(probsval)
    probsval['id_code'] = valdf['id_code'].values
    probsval.to_csv(os.path.join(path_data, 
                                 '../sub/weights/v31/probs_v31_512_fold{}.csv.gzip'.format(f)), 
                                compression = 'gzip',
                                index=False)
    
predsub = loadobj(os.path.join( path_data, '../sub/weights/v31/tst_probs_512_fold5.pk'))
probssub =  hmean([i.clip(1e-40, 1) for i in predsub])[:,:1108] 
sub = pd.read_csv( os.path.join( path_data, 'test.csv'))
probssub  = pd.DataFrame(probssub )
probssub ['id_code'] = sub['id_code']
probssub.to_csv(os.path.join(path_data, 
                             '../sub/weights/v31/probs_v31_512_allfolds.csv.gz'), 
                                compression = 'gzip', 
                                index=False)
#python post_processing_leak.py ../sub/weights/v31/probs_v31_512_allfolds.csv ../sub/weights/v31/probs_v31_512_allfolds_leak_doodle.csv
    
'''
# Output test probs  
probstst = hmean([i.clip(1e-40, 1) for i in probststls])
sub = pd.read_csv( os.path.join( path_data, 'test.csv'))
probstst = pd.DataFrame(probstst)
probstst['id_code'] = sub['id_code']
probstst.to_csv(os.path.join(path_data, 
                             '../sub/weights/v31/probs_v31_512_5folds_bagged.csv'),  
                                compression = 'gzip',
                                index=False)
'''
    '''
Options below 
# try with single sirna and leak
'''
probstst = hmean([i.clip(1e-40, 1) for i in probststls])
sub = pd.read_csv( os.path.join( path_data, 'test.csv'))
sub['sirna'] = probstst.argmax(1)

# Look at train groups
plate_groups = np.zeros((1108,4), int)
for sirna in range(1108):
    grp = traindf.loc[traindf.sirna==sirna,:].plate.value_counts().index.values
    assert len(grp) == 3
    plate_groups[sirna,0:3] = grp
    plate_groups[sirna,3] = 10 - grp.sum()
dfhmtrn = plate_groups

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
predicted = probstst.copy()
probsleak = probstst.copy()

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
                               '../sub/weights/v31/sub_v31_bagfolds_snglsirna_leak.csv.gz'), 
                                index=False, columns=['id_code','sirna']
                                , compression = 'gzip')
# check the sub
subold = pd.read_csv( os.path.join( path_data, \
                        '../sub/weights/v31/sub_v31_fold0_leak.csv'))
sum(submission.sirna== subold.sirna)/subold.shape[0]
