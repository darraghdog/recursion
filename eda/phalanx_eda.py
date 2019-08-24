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
huvecdf = pd.read_csv( os.path.join( path_data, 'huvec18.csv'))#.iloc[:3000]


# Load predictions
embtrn = loadobj(os.path.join( path_data, '../scripts/densenetv31/emb_trn_probs_512_fold5.pk'))
embtst = loadobj(os.path.join( path_data, '../scripts/densenetv31/emb_tst_probs_512_fold5.pk'))
dftrn = loadobj(os.path.join( path_data, '../scripts/densenetv31/df_trn_probs_512_fold5.pk'))
dftst = loadobj(os.path.join( path_data, '../scripts/densenetv31/df_tst_probs_512_fold5.pk'))

embtrn = pd.DataFrame(embtrn, index = dftrn.sirna)
#embtrn = embtrn.reset_index().groupby('sirna').mean()
#embtrn = embtrn.values

def oneshot(embtst, embtrn, trn_sirna_series):
    train_test_similarity = cosine_similarity(embtst, embtrn)
    train_test_similarity.shape #(19897, , 36515)
    tts = train_test_similarity.transpose()
    tts = pd.DataFrame(tts, index = trn_sirna_series)
    tts = tts.reset_index().groupby('sirna').max()
    tts = tts.transpose().values
    tts.shape #(19897, 1108)
    return tts

def expshot(embtst, embtrn, trn_sirna_series):
    embtrn = pd.DataFrame(embtrn, index = trn_sirna_series)
    
    train_test_similarity = cosine_similarity(embtst, embtrn)
    train_test_similarity.shape #(19897, , 36515)
    tts = train_test_similarity.transpose()
    tts = pd.DataFrame(tts, index = trn_sirna_series)
    tts = tts.reset_index().groupby('sirna').max()
    tts = tts.transpose().values
    tts.shape #(19897, 1108)
    return tts


from sklearn.metrics.pairwise import cosine_similarity
train_test_similarity = cosine_similarity(embtst, embtrn)
train_test_similarity.shape #(19897, 1108)
tts = train_test_similarity.transpose()
tts = pd.DataFrame(tts, index = dftrn.sirna)
tts = tts.reset_index().groupby('sirna').max()
tts = tts.transpose().values

tts.shape

ttsleak = pd.DataFrame(tts)
ttsleak['id_code'] = testdf['id_code'].values
ttsleak.to_csv(os.path.join( path_data, '../sub/tts/tts_dist.csv'),index = False)
ttsleak = pd.read_csv(os.path.join( path_data, '../sub/tts/tts_dist_ddl_leak.csv'))
subold2 = pd.read_csv( os.path.join( path_data, '../sub/lb/submit_32_TTActrls_leak_postproc.csv'))
idx = testdf.reset_index().set_index('id_code').loc[huvecdf['id_code']]['index'].tolist()
(subold2.iloc[idx].sirna == huvecdf.sirna.values).mean()
(ttsleak.iloc[idx].sirna == huvecdf.sirna.values).mean()



sub1 = dftrn.sirna[train_test_similarity.argmax(1)]

topk = (-train_test_similarity).sort(1)
topk .shape

dftrn.sirna[topk[:,:10]]

# predsbag = np.argmax(probsbag, 1)
submission = pd.read_csv( os.path.join( path_data, 'test.csv'))
submission['sirna'] = single_pred(submission, tts).astype(int)



idx = testdf.reset_index().set_index('id_code').loc[huvecdf['id_code']]['index'].tolist()
(train_test_similarity[idx].argmax(1) == huvecdf.sirna.values).mean()
(subold.set_index('id_code').loc[huvecdf['id_code']].sirna == huvecdf.sirna.values).mean()

# check the sub
subold1 = pd.read_csv( os.path.join( path_data, '../sub/lb/probs_v31_512_allfolds_leak_doodle.csv'))
subold2 = pd.read_csv( os.path.join( path_data, '../sub/lb/submit_32_TTActrls_leak_postproc.csv'))

(dftrn.sirna[topk[:,4]].values == subold1.sirna).mean()
(sub1.values == subold1.sirna).mean()
(sub1.values == subold2.sirna).mean()
(tts.argmax(1) == subold1.sirna).mean()
(submission['sirna'].values == subold1.sirna).mean()
(subold1.sirna == subold2.sirna).mean()

idx = subold1.sirna != subold2.sirna
(submission['sirna'].values[idx]==subold1.sirna[idx]).value_counts()
(submission['sirna'].values[idx]==subold2.sirna[idx]).value_counts()

subhack = subold1.copy()
idx = subold1.sirna != subold2.sirna
ix = sub1.values[idx]==subold2.sirna[idx]
ixloc = subhack[idx][ix].index
subhack.loc[ixloc, 'sirna']  = sub1.values[idx][ix] 
subhack.to_csv(PATH+'/../sub/probs_v31_512_allfolds_cosinehack.csv', index = False)

subhack = subold2.copy()
idx = subold1.sirna != subold2.sirna
ix = sub1.values[idx]==subold1.sirna[idx]
ixloc = subhack[idx][ix].index
subhack.loc[ixloc, 'sirna']  = sub1.values[idx][ix] 
subhack.to_csv(PATH+'/../sub/submit_32_TTActrls_leak_postproc_cosinehack.csv', index = False)


subhack = subold2.copy()
idx = subold1.sirna != subold2.sirna
ix = ttsleak.sirna.values[idx]==subold1.sirna[idx]
ixloc = subhack[idx][ix].index
subhack.loc[ixloc, 'sirna']  = ttsleak.sirna.values[idx][ix] 
subhack.to_csv(PATH+'/../sub/submit_32_TTActrls_leak_postproc__cosinehack_ddlleak.csv', index = False)

# check the sub
subold1 = pd.read_csv( os.path.join( path_data, '../sub/lb/probs_v31_512_allfolds_leak_doodle.csv'))
subold2 = pd.read_csv( os.path.join( path_data, '../sub/lb/submit_41_leak_postproc.csv'))

subhack = subold2.copy()
idx = subold1.sirna != subold2.sirna
ix = ttsleak.sirna.values[idx]==subold1.sirna[idx]
ixloc = subhack[idx][ix].index
subhack.loc[ixloc, 'sirna']  = ttsleak.sirna.values[idx][ix] 
subhack.to_csv(PATH+'/../sub/submit_41_leak_postproc__cosinehack_ddlleak.csv', index = False)



(submission['sirna'] == subold.sirna).mean()
(submission['sirna'] == subold.sirna).mean()

idx = testdf.reset_index().set_index('id_code').loc[huvecdf['id_code']]['index'].tolist()
(subold2.iloc[idx].sirna == huvecdf.sirna.values).mean()
(ttsleak.iloc[idx].sirna == huvecdf.sirna.values).mean()

(submission.iloc[idx].sirna == huvecdf.sirna.values).mean()

(dftrn.sirna[topk[:,0]].values[idx] == huvecdf.sirna.values).mean()

ix = (subold1.iloc[idx].sirna != huvecdf.sirna.values)

(dftrn.sirna[topk[:,0]].values[idx][ix] == huvecdf.sirna.values[ix]).mean()








