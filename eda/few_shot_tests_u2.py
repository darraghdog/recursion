import pandas as pd
import pickle
import numpy as np
import os
from tqdm import tqdm
from scipy.stats.mstats import hmean
import collections
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances


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

def expshotdelta_v5(embtst, embtrn, sirnas, trnexp, tstexp, dist_types=['cosine', 'l1']):
    '''
    LB : 0.954
    CV : 
        HEPG2    0.764903
        HUVEC    0.907338
        RPE      0.892171
        U2OS     0.751805
    '''
    # Get l2 norm
    embnorm = np.concatenate((embtrn, embtst), 0)
    embnorm = normalize(embnorm, axis=1, norm='l2')
    embtrn = embnorm[:sirnas.shape[0]]
    embtst = embnorm[sirnas.shape[0]:]
    
    # Create indexed df from each embedding matrix
    embtst = pd.DataFrame(embtst, index = tstexp)
    embtrn = pd.DataFrame(embtrn, index = sirnas)
    embtrn['experiment_grp'] = trnexp.apply(lambda x: x.split('-')[0]).values
    
    # Mean embedding per experiment and per experiment group for train
    embgrptrnavg = embtrn.reset_index().groupby(['experiment_grp']).mean()
    embtrn = embtrn.reset_index().groupby(['sirna', 'experiment_grp']).mean()
    # Mean embedding per experiment for test
    embgrptstavg = embtst.reset_index().groupby(['experiment']).mean()
    
    # Get train and test experiment groups and sirnas
    trnexp_grp = embtrn.reset_index()['experiment_grp'].values
    tstexp_grp = embtst.reset_index()['experiment'].apply(lambda x: x.split('-')[0]).values
    sirnas = embtrn.reset_index()['sirna'].values    
    
    # Subtract mean experiment embedding 
    embtrn = embtrn.values - embgrptrnavg.loc[trnexp_grp][embtrn.columns].values 
    embtst = embtst.values - embgrptstavg.loc[tstexp][embtst.columns].values  
    
    # Get calculate distance and -ve to get similarity
    ttsls = [(pairwise_distances(embtst, embtrn, metric = d)) for d in dist_types]
    # Standardise both distance metrics
    tts = -1 * sum([(m-np.mean(m))/np.std(m) for m in ttsls])
    tts = tts.transpose() 
    
    # Create mask where experiments are not the same, to remove incidental similarity
    mask = pd.DataFrame(np.zeros(tts.shape))
    for e in np.unique(tstexp_grp):
        mask.iloc[np.where(trnexp_grp==e)[0], np.where(tstexp_grp==e)[0]] = 1
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
u2_run = True

for i in tqdm(range(5)):
    validx = ~traindf['experiment'].isin(folds[folds['fold']==i]['experiment'].tolist()).values
    val_u2 = u2_run & (traindf['experiment'][~validx].str.contains('U2').sum()>0)
    embtrnls = loadobj(os.path.join( path_data, '_emb_trn_probs_512_fold{}.pk'.format(i)))
    embvalls = loadobj(os.path.join( path_data, '_emb_val_probs_512_fold{}.pk'.format(i)))
    embtrn = (sum(embtrnls)/len(embtrnls))[validx]
    embval = sum(embvalls)/len(embvalls)
    dftrn = loadobj(os.path.join( path_data, '_df_trn_probs_512_fold{}.pk'.format(i)))
    dfval = loadobj(os.path.join( path_data, '_df_val_probs_512_fold{}.pk'.format(i)))
    dftrn = dftrn[validx]
    traindf['pred_cossim'].loc[i] = expshotdelta_v5(embval, embtrn, dftrn.sirna, dftrn.experiment, dfval.experiment).argmax(1)
    if val_u2 :
        embtrnls = loadobj(os.path.join( path_data, 'u2/_emb_u2_trn_probs_512_fold{}.pk'.format(i)))
        embvalls = loadobj(os.path.join( path_data, 'u2/_emb_u2_val_probs_512_fold{}.pk'.format(i)))
        dftrn = loadobj(os.path.join( path_data, 'u2/_df_u2_trn_probs_512_fold{}.pk'.format(i)))
        dfval = loadobj(os.path.join( path_data, 'u2/_df_u2_val_probs_512_fold{}.pk'.format(i)))
        validx = ~dftrn.experiment.isin(dfval.experiment)
        dftrn = dftrn[validx]
        embtrn = (sum(embtrnls)/len(embtrnls))[validx]
        embval = sum(embvalls)/len(embvalls)
        predu2 = expshotdelta_v5(embval, embtrn, dftrn.sirna, dftrn.experiment, dfval.experiment).argmax(1)
        traindf = traindf.reset_index().set_index('id_code')
        traindf['pred_cossim'].loc[dfval.id_code] = predu2
        traindf = traindf.reset_index().set_index('fold')
        
traindf['eqpred_cossim'] = (traindf['pred_cossim']==traindf['sirna']).astype(np.int8)
traindf['exptype'] = traindf['experiment'].apply(lambda x: x.split('-')[0])
print(traindf['eqpred_cossim'].mean())
print(traindf.groupby('exptype')['eqpred_cossim'].mean())
print(traindf.groupby('experiment')['eqpred_cossim'].mean())

'''Without u2 finetune
HEPG2    0.764903
HUVEC    0.907338
RPE      0.892171
U2OS     0.751805

U2OS-01     0.771661
U2OS-02     0.664260
U2OS-03     0.819495
'''

'''With u2 finetune
HEPG2    0.764903
HUVEC    0.907338
RPE      0.892171
U2OS     0.783394
'''

'''
Make Sub
'''

dftrn = loadobj(os.path.join( path_data, '_df_trn_probs_512_fold5.pk'))
dftst = loadobj(os.path.join( path_data, '_df_tst_probs_512_fold5.pk'))
embtrn = loadobj(os.path.join( path_data, '_emb_trn_probs_512_fold5.pk'))
embtst = loadobj(os.path.join( path_data, '_emb_tst_probs_512_fold5.pk'))
embtrn = sum(embtrn)/len(embtrn)
embtst = sum(embtst)/len(embtst)
ttsexp = expshotdelta_v5(embtst, embtrn, dftrn.sirna, dftrn.experiment, dftst.experiment)
ttsexp = pd.DataFrame(ttsexp)
ttsexp['id_code'] = testdf['id_code'].values
ttsexp = ttsexp.set_index('id_code')

# LB0.968 was u2_v1
dftrn = loadobj(os.path.join( path_data, 'u2_v2/_df_u2_trn_probs_512_fold5.pk'))
dftst = loadobj(os.path.join( path_data, 'u2_v2/_df_u2_tst_probs_512_fold5.pk'))
embtrn = loadobj(os.path.join( path_data, 'u2_v2/_emb_u2_trn_probs_512_fold5.pk'))
embtst = loadobj(os.path.join( path_data, 'u2_v2/_emb_u2_tst_probs_512_fold5.pk'))
embtrn = sum(embtrn)/len(embtrn)
embtst = sum(embtst)/len(embtst)
ttsexpu2 = expshotdelta_v5(embtst, embtrn, dftrn.sirna, dftrn.experiment, dftst.experiment)
ttsexp.loc[dftst.id_code] = ttsexpu2

ttsexp = ttsexp.reset_index()[list(range(1108))+['id_code']]

ttsexp.to_csv(os.path.join( path_data, 'ttsexp_mask_v31_finetune_u2_500ep_fold5.csv'), index = False)
# Run : python post_processing_leak.py ttsexp_mask_v31_finetune_u2_500ep_fold5.csv ttsexp_mask_v31_finetune_u2_500ep_fold5_ddlleak.csv 
