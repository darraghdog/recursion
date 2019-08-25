import pandas as pd
import pickle
import numpy as np
import os
from tqdm import tqdm
from scipy.stats.mstats import hmean
import collections
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

def expshotdelta_v2(embtst, embtrn, sirnas, exp, testexp):
    # v3 - mask and norm
    '''
    LB : 0.946
    CV : 
    HEPG2    0.753161
    HUVEC    0.903890
    RPE      0.885335
    U2OS     0.735259
    '''

    # Get mean embedding per sirna experiment, and subtract the experiment level embedding
    embtrn = pd.DataFrame(embtrn, index = sirnas)
    embtrn['experiment'] = exp.apply(lambda x: x.split('-')[0]).values
    embtrnavg = embtrn.reset_index().groupby(['experiment']).mean()
    embtrn = embtrn.reset_index().groupby(['sirna', 'experiment']).mean()
    trnexp = embtrn.reset_index()['experiment'].values
    sirnas = embtrn.reset_index()['sirna'].values
    embtrn = embtrn.values - embtrnavg.loc[embtrn.reset_index().experiment][embtrn.columns].values    
    
    # Subtract mean experiment embedding from each sample
    embtst = pd.DataFrame(embtst, index = testexp)
    embtstavg = embtst.reset_index().groupby(['experiment']).mean()
    tstexp = embtst.reset_index()['experiment'].values
    tstexp = pd.Series(tstexp).apply(lambda x: x.split('-')[0]).values
    embtst = embtst.values - embtstavg.loc[embtst.reset_index().experiment][embtst.columns].values    

    # Get cosine similarity train test
    train_test_similarity = cosine_similarity(embtst, embtrn)
    tts = train_test_similarity.transpose() #(4432, 6642)
    
    # Create mask where experiments are not the same, to remove incidental similarity
    mask = pd.DataFrame(np.zeros(tts.shape))
    for e in np.unique(tstexp):
        mask.iloc[np.where(trnexp==e)[0], np.where(tstexp==e)[0]] = 1
    tts[mask==0] = tts.min()
    
    tts = pd.DataFrame(tts, index = sirnas)
    tts = tts.reset_index().groupby('index').max()
    tts = tts.transpose().values

    return tts


def expshotdelta_v3(embtst, embtrn, sirnas, exp, testexp):
    # v3 - mask and norm
    '''
    LB : ?
    CV : 
        HEPG2    0.755484
        HUVEC    0.904625
        RPE      0.887011
        U2OS     0.733755
    '''
    # Get l2 norm
    embnorm = np.concatenate((embtrn, embtst), 0)
    embnorm = normalize(embnorm, axis=1, norm='l2')
    embtrn = embnorm[:sirnas.shape[0]]
    embtst = embnorm[sirnas.shape[0]:]

    # Get mean embedding per sirna experiment, and subtract the experiment level embedding
    embtrn = pd.DataFrame(embtrn, index = sirnas)
    embtrn['experiment'] = exp.apply(lambda x: x.split('-')[0]).values
    embtrnavg = embtrn.reset_index().groupby(['experiment']).mean()
    embtrn = embtrn.reset_index().groupby(['sirna', 'experiment']).mean()
    trnexp = embtrn.reset_index()['experiment'].values
    sirnas = embtrn.reset_index()['sirna'].values
    embtrn = embtrn.values - embtrnavg.loc[embtrn.reset_index().experiment][embtrn.columns].values    
    
    # Subtract mean experiment embedding from each sample
    embtst = pd.DataFrame(embtst, index = testexp)
    embtstavg = embtst.reset_index().groupby(['experiment']).mean()
    tstexp = embtst.reset_index()['experiment'].values
    tstexp = pd.Series(tstexp).apply(lambda x: x.split('-')[0]).values
    embtst = embtst.values - embtstavg.loc[embtst.reset_index().experiment][embtst.columns].values    

    # Get cosine similarity train test
    train_test_similarity = cosine_similarity(embtst, embtrn)
    tts = train_test_similarity.transpose() #(4432, 6642)
    
    tts = pd.DataFrame(tts, index = sirnas)
    tts = tts.reset_index().groupby('index').max()
    tts = tts.transpose().values

    return tts

def expshotdelta_v4(embtst, embtrn, sirnas, exp, testexp):
    # v3 - mask and norm
    '''
    LB : 0.952
    CV : 
        HEPG2    0.760645
        HUVEC    0.906038
        RPE      0.887140
        U2OS     0.742479
    '''
    # Get l2 norm
    embnorm = np.concatenate((embtrn, embtst), 0)
    embnorm = normalize(embnorm, axis=1, norm='l2')
    embtrn = embnorm[:sirnas.shape[0]]
    embtst = embnorm[sirnas.shape[0]:]

    # Get mean embedding per sirna experiment, and subtract the experiment level embedding
    embtrn = pd.DataFrame(embtrn, index = sirnas)
    embtrn['experiment'] = exp.apply(lambda x: x.split('-')[0]).values
    embtrnavg = embtrn.reset_index().groupby(['experiment']).mean()
    embtrn = embtrn.reset_index().groupby(['sirna', 'experiment']).mean()
    trnexp = embtrn.reset_index()['experiment'].values
    sirnas = embtrn.reset_index()['sirna'].values
    embtrn = embtrn.values - embtrnavg.loc[embtrn.reset_index().experiment][embtrn.columns].values    
    
    # Subtract mean experiment embedding from each sample
    embtst = pd.DataFrame(embtst, index = testexp)
    embtstavg = embtst.reset_index().groupby(['experiment']).mean()
    tstexp = embtst.reset_index()['experiment'].values
    tstexp = pd.Series(tstexp).apply(lambda x: x.split('-')[0]).values
    embtst = embtst.values - embtstavg.loc[embtst.reset_index().experiment][embtst.columns].values    

    # Get cosine similarity train test
    train_test_similarity = cosine_similarity(embtst, embtrn)
    tts = train_test_similarity.transpose() #(4432, 6642)
    
    # Create mask where experiments are not the same, to remove incidental similarity
    mask = pd.DataFrame(np.zeros(tts.shape))
    for e in np.unique(tstexp):
        mask.iloc[np.where(trnexp==e)[0], np.where(tstexp==e)[0]] = 1
    tts[mask==0] = tts.min()
    
    tts = pd.DataFrame(tts, index = sirnas)
    tts = tts.reset_index().groupby('index').max()
    tts = tts.transpose().values

    return tts


path_data = '/Users/dhanley2/Documents/Personal/recursion/sub/tts/'
traindf = pd.read_csv( os.path.join( path_data, 'train.csv'))#.iloc[:3000]
testdf  = pd.read_csv( os.path.join( path_data, 'test.csv'))
folds = pd.read_csv( os.path.join( path_data, 'folds.csv'))
traindf = traindf.merge(folds, on = 'experiment')
traindf = traindf.set_index('fold')

'''
Full validation
'''
traindf['pred_cossim'] = -1

for i in tqdm(range(5)):
    validx = ~traindf['experiment'].isin(folds[folds['fold']==i]['experiment'].tolist()).values
    embtrnls = loadobj(os.path.join( path_data, '_emb_trn_probs_512_fold{}.pk'.format(i)))
    embvalls = loadobj(os.path.join( path_data, '_emb_val_probs_512_fold{}.pk'.format(i)))
    embtrn = (sum(embtrnls)/len(embtrnls))[validx]
    embval = sum(embvalls)/len(embvalls)
    dftrn = loadobj(os.path.join( path_data, '_df_trn_probs_512_fold{}.pk'.format(i)))
    dfval = loadobj(os.path.join( path_data, '_df_val_probs_512_fold{}.pk'.format(i)))
    dftrn = dftrn[validx]
    traindf['pred_cossim'].loc[i] = expshotdelta_v4(embval, embtrn, dftrn.sirna, dftrn.experiment, dfval.experiment).argmax(1)
    
traindf['eqpred_cossim'] = (traindf['pred_cossim']==traindf['sirna']).astype(np.int8)
traindf['exptype'] = traindf['experiment'].apply(lambda x: x.split('-')[0])
print(traindf['eqpred_cossim'].mean())
print(traindf.groupby('exptype')['eqpred_cossim'].mean())
print(traindf.groupby('experiment')['eqpred_cossim'].mean())

'''
Make Sub
'''

dftrn = loadobj(os.path.join( path_data, '_df_trn_probs_512_fold5.pk'))
dftst = loadobj(os.path.join( path_data, '_df_tst_probs_512_fold5.pk'))
embtrn = loadobj(os.path.join( path_data, '_emb_trn_probs_512_fold5.pk'))
embtst = loadobj(os.path.join( path_data, '_emb_tst_probs_512_fold5.pk'))
embtrn = sum(embtrn)/len(embtrn)
embtst = sum(embtst)/len(embtst)
ttsexp = expshotdelta_v4(embtst, embtrn, dftrn.sirna, dftrn.experiment, dftst.experiment)
ttsexp = pd.DataFrame(ttsexp)
ttsexp['id_code'] = testdf['id_code'].values
ttsexp.to_csv(os.path.join( path_data, 'ttsexp_mask_v31_fold5.csv'),index = False)
# Run : python post_processing_leak.py ttsexp_mask_v31_fold5.csv ttsexp_simmask_v31_fold5_ddlleak.csv 
#ttsleak = pd.read_csv(os.path.join( path_data, 'ttsexp_mask_v31_fold5_ddlleak.csv'))
