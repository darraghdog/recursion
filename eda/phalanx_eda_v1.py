import pandas as pd
import pickle
import numpy as np
import os
from tqdm import tqdm
from scipy.stats.mstats import hmean
import collections
from sklearn.metrics.pairwise import cosine_similarity


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
    

def oneshot(embtst, embtrn, trn_sirna_series):
    train_test_similarity = cosine_similarity(embtst, embtrn)
    train_test_similarity.shape #(19897, , 36515)
    tts = train_test_similarity.transpose()
    tts = pd.DataFrame(tts, index = trn_sirna_series)
    tts = tts.reset_index().groupby('sirna').max()
    tts = tts.transpose().values
    tts.shape #(19897, 1108)
    return tts

def expshot(embtst, embtrn, sirnas, exp):
    embtrn1 = pd.DataFrame(embtrn, index = sirnas)
    embtrn1['experiment'] = exp.apply(lambda x: x.split('-')[0]).values
    embtrn1 = embtrn1.reset_index().groupby(['sirna', 'experiment']).mean()
    sirnas1 = embtrn1.reset_index()['sirna'].values
    train_test_similarity = cosine_similarity(embtst, embtrn1)
    train_test_similarity.shape #(19897, , 36515)
    tts = train_test_similarity.transpose()
    tts = pd.DataFrame(tts, index = sirnas1)
    tts = tts.reset_index().groupby('index').max()
    tts = tts.transpose().values
    tts.shape #(19897, 1108)
    return tts

PATH = '/Users/dhanley2/Documents/Personal/recursion/data'
path_data = PATH
traindf = pd.read_csv( os.path.join( path_data, 'train.csv'))#.iloc[:3000]
testdf  = pd.read_csv( os.path.join( path_data, 'test.csv'))
traincdf = pd.read_csv( os.path.join( path_data, 'train_controls.csv'))#.iloc[:3000]
testcdf = pd.read_csv( os.path.join( path_data, 'test_controls.csv'))#.iloc[:3000]
huvecdf = pd.read_csv( os.path.join( path_data, 'huvec18.csv'))#.iloc[:3000]
folds = pd.read_csv( os.path.join( path_data, 'folds.csv'))
traindf = traindf.merge(folds, on = 'experiment')

'''
Correlation
'''

dfls = []
embls = []
traindf['tts'] = traindf['ttsexp'] = traindf['tts1'] = traindf['ttsexp1'] = -1
traindf = traindf.set_index('fold')
for i in tqdm(range(5)):
    validx = traindf['experiment'].isin(folds[folds['fold']==i]['experiment'].tolist()).values
    embvalls = loadobj(os.path.join( path_data, '../sub/tts/_emb_val_probs_512_fold{}.pk'.format(i)))
    embls.append((sum(embvalls)/len(embvalls)))
    dfval = loadobj(os.path.join( path_data, '../sub/tts/_df_val_probs_512_fold{}.pk'.format(i)))
    dfls .append(dfval)

df = pd.concat(dfls, 0)
emb = np.concatenate(embls, 0)

fold = 5
embtrnls = loadobj(os.path.join( path_data, '../sub/tts/_emb_trn_probs_512_fold{}.pk'.format(i)))
dftrn = loadobj(os.path.join( path_data, '../sub/tts/_df_trn_probs_512_fold{}.pk'.format(i)))
embtstls = loadobj(os.path.join( path_data, '../sub/tts/_emb_tst_probs_512_fold{}.pk'.format(i)))
dftst = loadobj(os.path.join( path_data, '../sub/tts/_df_tst_probs_512_fold{}.pk'.format(i)))
df = pd.concat([dftrn, dftst], 0)
emb = np.concatenate((sum(embtrnls)/len(embtrnls), sum(embtstls)/len(embtstls)), 0)
emb = pd.DataFrame(emb)
emb['sirna'] = df.sirna.values
emb['experiment']  = df.experiment.values

dftst.experiment.unique()

grpdf = emb.groupby('experiment')[emb.columns[:1023].tolist()].mean()

grpdf = grpdf.transpose()
corr = grpdf.corr()
corr.to_csv(os.path.join( path_data, '../../emb_exp_correlation.csv'))

'''
Full validation
'''

traindf['tts'] = traindf['ttsexp'] = traindf['tts1'] = traindf['ttsexp1'] = -1
traindf = traindf.set_index('fold')

origtrn = dftrn.experiment
newtrn = dftrn.experiment.str.replace('HUVEC-05', 'A_HUVEC-05')

for i in tqdm(range(5)):
    validx = ~traindf['experiment'].isin(folds[folds['fold']==i]['experiment'].tolist()).values
    embtrnls = loadobj(os.path.join( path_data, '../sub/tts/_emb_trn_probs_512_fold{}.pk'.format(i)))
    embvalls = loadobj(os.path.join( path_data, '../sub/tts/_emb_val_probs_512_fold{}.pk'.format(i)))
    embtrn = (sum(embtrnls)/len(embtrnls))[validx]
    embval = sum(embvalls)/len(embvalls)
    dftrn = loadobj(os.path.join( path_data, '../sub/tts/_df_trn_probs_512_fold{}.pk'.format(i)))
    dftrn = dftrn[validx]
    traindf['ttsexp'].loc[i] = expshot(embval, embtrn, dftrn.sirna, dftrn.experiment).argmax(1)
    traindf['ttsexpA'].loc[i] = expshot(embval, embtrn, dftrn.sirna, dftrn.experiment).argmax(1)


traindf['eqttsexp'] = (traindf['ttsexp']==traindf['sirna']).astype(np.int8)
scoresdf = traindf.reset_index() \
            .groupby(['fold', 'experiment'])['eqttsexp'] \
            .mean().sort_values('eqttsexp')
scoresdf.to_csv(os.path.join( path_data, '../../tts_scores2.csv'))

'''
Full validation
'''

traindf['tts'] = traindf['ttsexp'] = traindf['tts1'] = traindf['ttsexp1'] = -1
traindf = traindf.set_index('fold')

for i in tqdm(range(5)):
    validx = ~traindf['experiment'].isin(folds[folds['fold']==i]['experiment'].tolist()).values
    embtrnls = loadobj(os.path.join( path_data, '../sub/tts/_emb_trn_probs_512_fold{}.pk'.format(i)))
    embvalls = loadobj(os.path.join( path_data, '../sub/tts/_emb_val_probs_512_fold{}.pk'.format(i)))
    embtrn = (sum(embtrnls)/len(embtrnls))[validx]
    embval = sum(embvalls)/len(embvalls)
    dftrn = loadobj(os.path.join( path_data, '../sub/tts/_df_trn_probs_512_fold{}.pk'.format(i)))
    dftrn = dftrn[validx]
    traindf['tts1'].loc[i] = (sum([oneshot(ev, et[validx], dftrn.sirna) \
           for (ev, et) in zip(embvalls,embtrnls)])/len(embtrnls))  .argmax(1)
    traindf['ttsexp1'].loc[i] = (sum([expshot(ev, et[validx], dftrn.sirna, dftrn.experiment) \
           for (ev, et) in zip(embvalls,embtrnls)])/len(embtrnls))  .argmax(1)
    traindf['tts'].loc[i] = oneshot(embval, embtrn, dftrn.sirna).argmax(1)
    traindf['ttsexp'].loc[i] = expshot(embval, embtrn, dftrn.sirna, dftrn.experiment).argmax(1)

traindf['eqtts'] = (traindf['tts']==traindf['sirna']).astype(np.int8)
traindf['eqttsexp'] = (traindf['ttsexp']==traindf['sirna']).astype(np.int8)
traindf['eqtts1'] = (traindf['tts1']==traindf['sirna']).astype(np.int8)
traindf['eqttsexp1'] = (traindf['ttsexp1']==traindf['sirna']).astype(np.int8)
scoresdf = traindf.reset_index() \
            .groupby(['fold', 'experiment'])['eqtts', 'eqttsexp', 'eqtts1', 'eqttsexp1'] \ 
            .mean().sort_values('eqttsexp')
scoresdf.to_csv(os.path.join( path_data, '../../tts_scores.csv'))

'''
Validation set -Fold 0
'''

dftrn = loadobj(os.path.join( path_data, '../sub/tts/_df_trn_probs_512_fold0.pk'))
dfval = loadobj(os.path.join( path_data, '../sub/tts/_df_val_probs_512_fold0.pk'))

for i in range(10):
    embtrn = loadobj(os.path.join( path_data, '../sub/tts/_emb_trn_probs_512_fold0.pk'))[-i]
    embval = loadobj(os.path.join( path_data, '../sub/tts/_emb_val_probs_512_fold0.pk'))[-i]
    tts = oneshot(embval, embtrn, dftrn.sirna)
    ttsexp = expshot(embval, embtrn, dftrn.sirna, dftrn.experiment)
    print('---{}---'.format(i))
    print((tts.argmax(1) == dfval.sirna.values).mean())
    print((ttsexp.argmax(1) == dfval.sirna.values).mean())

embtrn = loadobj(os.path.join( path_data, '../sub/tts/_emb_trn_probs_512_fold0.pk'))
embval = loadobj(os.path.join( path_data, '../sub/tts/_emb_val_probs_512_fold0.pk'))
embtrn = sum(embtrn)/len(embtrn)
embval = sum(embval)/len(embval)
tts = oneshot(embval, embtrn, dftrn.sirna)
ttsexp = expshot(embval, embtrn, dftrn.sirna, dftrn.experiment)
print('---{}---'.format(i))
print((tts.argmax(1) == dfval.sirna.values).mean())
print((ttsexp.argmax(1) == dfval.sirna.values).mean())

'''
Val on Fold 5 using all folds and HUVEC18 as val 
'''

dftrn = loadobj(os.path.join( path_data, '../sub/tts/_df_trn_probs_512_fold5.pk'))
dfval = loadobj(os.path.join( path_data, '../sub/tts/_df_val_probs_512_fold5.pk'))

for i in range(10):
    embtrn = loadobj(os.path.join( path_data, '../sub/tts/_emb_trn_probs_512_fold5.pk'))[-i]
    embval = loadobj(os.path.join( path_data, '../sub/tts/_emb_val_probs_512_fold5.pk'))[-i]
    tts = oneshot(embval, embtrn, dftrn.sirna)
    ttsexp = expshot(embval, embtrn, dftrn.sirna, dftrn.experiment)
    print('---{}---'.format(i))
    print((tts.argmax(1) == dfval.sirna.values).mean())
    print((ttsexp.argmax(1) == dfval.sirna.values).mean())

embtrn = loadobj(os.path.join( path_data, '../sub/tts/_emb_trn_probs_512_fold5.pk'))
embval = loadobj(os.path.join( path_data, '../sub/tts/_emb_val_probs_512_fold5.pk'))
embtrn = sum(embtrn)/len(embtrn)
embval = sum(embval)/len(embval)
tts = oneshot(embval, embtrn, dftrn.sirna)
ttsexp = expshot(embval, embtrn, dftrn.sirna, dftrn.experiment)
print('---{}---'.format(i))
print((tts.argmax(1) == dfval.sirna.values).mean())
print((ttsexp.argmax(1) == dfval.sirna.values).mean())

'''
Val on Fold 5 using all folds and HUVEC18 as val 
'''

dftrn = loadobj(os.path.join( path_data, '../sub/tts/_df_trn_probs_512_fold5.pk'))
dfval = loadobj(os.path.join( path_data, '../sub/tts/_df_val_probs_512_fold5.pk'))
dftst = loadobj(os.path.join( path_data, '../sub/tts/_df_tst_probs_512_fold5.pk'))

embtrn = loadobj(os.path.join( path_data, '../sub/tts/_emb_trn_probs_512_fold5.pk'))
embval = loadobj(os.path.join( path_data, '../sub/tts/_emb_val_probs_512_fold5.pk'))
embtst = loadobj(os.path.join( path_data, '../sub/tts/_emb_tst_probs_512_fold5.pk'))

ix = dftst.reset_index().set_index('id_code').loc[huvecdf.id_code]['index'].values
ttsexpls = [expshot(embs, embt, dftrn.sirna, dftrn.experiment) for (embs, embt) in tqdm(zip(embtst, embtrn))]
print(((sum(ttsexpls)/10).argmax(1)[ix] == huvecdf.sirna.values).mean())
ttsexpavg = expshot(sum(embtst)/10, sum(embtrn)/10, dftrn.sirna, dftrn.experiment) 
print((ttsexpavg.argmax(1)[ix] == huvecdf.sirna.values).mean())


'''
Test set -Fold 5
Closest at experiment level
'''

traindf = pd.read_csv( os.path.join( path_data, 'train.csv'))#.iloc[:3000]
testdf  = pd.read_csv( os.path.join( path_data, 'test.csv'))

def oneshot(embtst, embtrn, trn_sirna_series):
    train_test_similarity = cosine_similarity(embtst, embtrn)
    train_test_similarity.shape #(19897, , 36515)
    tts = train_test_similarity.transpose()
    tts = pd.DataFrame(tts, index = trn_sirna_series)
    tts = tts.reset_index().groupby('sirna').max()
    tts = tts.transpose().values
    tts.shape #(19897, 1108)
    return tts

def expshot(embtst, embtrn, sirnas, exp):
    embtrn1 = pd.DataFrame(embtrn, index = sirnas)
    embtrn1['experiment'] = exp.apply(lambda x: x.split('-')[0]).values
    embtrn1 = embtrn1.reset_index().groupby(['sirna', 'experiment']).mean()
    sirnas1 = embtrn1.reset_index()['sirna'].values
    train_test_similarity = cosine_similarity(embtst, embtrn1)
    train_test_similarity.shape #(19897, , 36515)
    tts = train_test_similarity.transpose()
    tts = pd.DataFrame(tts, index = sirnas1)
    tts = tts.reset_index().groupby('index').max()
    tts = tts.transpose().values
    tts.shape #(19897, 1108)
    return tts

dftrn = loadobj(os.path.join( path_data, '../sub/tts/_df_trn_probs_512_fold5.pk'))
dfval = loadobj(os.path.join( path_data, '../sub/tts/_df_val_probs_512_fold5.pk'))

embtrn = loadobj(os.path.join( path_data, '../sub/tts/_emb_trn_probs_512_fold5.pk'))
embtst = loadobj(os.path.join( path_data, '../sub/tts/_emb_tst_probs_512_fold5.pk'))
embtrn = sum(embtrn)/len(embtrn)
embtst = sum(embtst)/len(embtst)
ttsexp = expshot(embtst, embtrn, dftrn.sirna, dftrn.experiment)
ttsexp = pd.DataFrame(ttsexp)
ttsexp['id_code'] = testdf['id_code'].values
ttsexp.to_csv(os.path.join( path_data, '../sub/tts/ttsexp_v31_fold5.csv'),index = False)
# Run : python ../../eda/post_processing_leak.py ttsexp_v31_fold5.csv ttsexp_v31_fold5_ddlleak.csv 
ttsleak = pd.read_csv(os.path.join( path_data, '../sub/tts/ttsexp_v31_fold5_ddlleak.csv'))

'''
# check the sub
'''
subold1 = pd.read_csv( os.path.join( path_data, '../sub/lb/submit_41_leak_postproc__cosinehack_ddlleak.csv'))
print('Argmax matches {}'.format((ttsexp.values[:,:-1].argmax(1) == subold1.sirna.values).mean()))
print('Doodleleak matches {}'.format((ttsleak.sirna.values == subold1.sirna.values).mean()))

subold1 = pd.read_csv( os.path.join( path_data, '../sub/lb/submit_41_leak_postproc__cosinehack_ddlleak.csv'))
subold2 = pd.read_csv( os.path.join( path_data, '../sub/lb/submit_41_leak_postproc.csv'))
subold3 = pd.read_csv( os.path.join( path_data, '../sub/lb/probs_v31_512_allfolds_leak_doodle.csv'))


print('Argmax HUVEC {}'.format((ttsexp.set_index('id_code').loc[dfval['id_code']].values.argmax(1) == dfval.sirna.values).mean()))
print('Doodleleak HUVEC {}'.format((ttsleak.set_index('id_code').loc[dfval['id_code']].sirna.values == dfval.sirna.values).mean()))
print('Doodleleak HUVEC {}'.format((subold1.set_index('id_code').loc[dfval['id_code']].sirna.values == dfval.sirna.values).mean()))
print('Doodleleak HUVEC {}'.format((subold2.set_index('id_code').loc[dfval['id_code']].sirna.values == dfval.sirna.values).mean()))


subhack = subold2.copy()
idx = subold3.sirna != subold2.sirna
ix = ttsleak.sirna.values[idx]==subold3.sirna[idx]
ixloc = subhack[idx][ix].index
subhack.loc[ixloc, 'sirna']  = ttsleak.sirna.values[idx][ix] 
subhack.to_csv(PATH+'/../sub/submit_41_leak_postproc__cosinehack_ddlleak.csv', index = False)