from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import optparse
import os, sys
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

import warnings
warnings.filterwarnings('ignore')

# Print info about environments
parser = optparse.OptionParser()
parser.add_option('-r', '--rootpath', action="store", dest="rootpath", help="root directory", default="/share/dhanley2/recursion/")
parser.add_option('-i', '--imgpath', action="store", dest="imgpath", help="root directory", default="data/mount/512X512X6/")
parser.add_option('-w', '--workpath', action="store", dest="workpath", help="Working path", default="densenetv1/weights")
parser.add_option('-v', '--embpath', action="store", dest="embpath", help="Working path", default="mount/densenetv53/raw")

parser.add_option('-c', '--customwt', action="store", dest="customwt", help="Weight of annotator count in loss", default="1.0")
parser.add_option('-n', '--probsname', action="store", dest="probsname", help="probs file name", default="probs_256")
parser.add_option('-a', '--embname', action="store", dest="embname", help="Embeddings file name", default="_cls_site{}_{}_probs_512_fold5.pk")
parser.add_option('-b', '--dfname', action="store", dest="dfname", help="Embeddings file name", default="_df_site{}_{}_probs_512_fold5.pk")
parser.add_option('-g', '--logmsg', action="store", dest="logmsg", help="root directory", default="Recursion-pytorch")

options, args = parser.parse_args()
package_dir = options.rootpath
sys.path.append(package_dir)
from logs import get_logger
from utils import dumpobj, loadobj, GradualWarmupScheduler


# Print info about environments
logger = get_logger(options.logmsg, 'INFO') # noqa


logger.info('Load params : time {}'.format(datetime.datetime.now().time()))
for (k,v) in options.__dict__.items():
    logger.info('{}{}'.format(k.ljust(20), v))


ROOT = options.rootpath
path_data = os.path.join(ROOT, 'data')
path_img = os.path.join(ROOT, options.imgpath)
emb_path = os.path.join(ROOT, options.embpath)
WORK_DIR = os.path.join(ROOT, options.workpath)
EMBNAME = options.embname
PROBSNAME = options.probsname
DFNAME = options.dfname

def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

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


logger.info('Load Dataframes')

traindf = pd.read_csv( os.path.join( path_data, 'train.csv'))#.iloc[:3000]
testdf  = pd.read_csv( os.path.join( path_data, 'test.csv'))
folds = pd.read_csv( os.path.join( path_data, 'folds.csv'))
traindf = traindf.merge(folds, on = 'experiment')
traindf = traindf.set_index('fold')
bestdf = pd.read_csv( os.path.join( path_data, 'tmp.csv'))
logger.info(bestdf.shape)
u2idx = bestdf.id_code.str.contains('U2OS')
logger.info(u2idx.mean())

'''
Make Sub
'''

dftrn = loadobj(os.path.join( emb_path, DFNAME.format('trn')))
dftst = loadobj(os.path.join( emb_path, DFNAME.format('tst')))
logger.info(dftrn.shape)
logger.info(dftst.shape)
embtst = loadobj(os.path.join( emb_path, EMBNAME.format('tst')))
embtstsc = loadobj(os.path.join( emb_path, EMBNAME.format('tst').replace('emb_', 'cls_')))

scores = [(e[u2idx].argmax(1)==bestdf.sirna.values[u2idx]).mean() for e in embtstsc]
logger.info('Val Scores')
logger.info(scores)
logger.info([s>0.86 for s in scores])

embtrn = loadobj(os.path.join( emb_path, EMBNAME.format('trn')))
embtrn = [e[:,:1108] for (e, s) in zip(embtrn, scores) if s> 0.87  ]
embtst = [e[:,:1108] for (e, s) in zip(embtst, scores) if s> 0.87  ]
logger.info('Kept {} epochs'.format(len(embtrn)))
embtrn = sum(embtrn)/len(embtrn)
embtst = sum(embtst)/len(embtst)
logger.info(embtrn.shape)
logger.info(embtst.shape)
ttsexp = expshotdelta_v5(embtst, embtrn, dftrn.sirna, dftrn.experiment, dftst.experiment)
ttsexp = pd.DataFrame(ttsexp.astype(np.float32))
ttsexp['id_code'] = testdf['id_code'].values
OUTFILE = os.path.join( WORK_DIR, PROBSNAME)
logger.info('Write out {}'.format(OUTFILE))
ttsexp.to_csv(OUTFILE,index = False, compression = 'gzip')

nsamp = 10000
eqpred_cossim = (ttsexp.values.argmax(1)[:nsamp]==bestdf['sirna'][:nsamp].values).astype(np.int8)
tstpreddf = pd.DataFrame({'exptype':testdf['experiment'][:nsamp].apply(lambda x: x.split('-')[0]), 
              'eqpred_cossim':eqpred_cossim[:nsamp],
              'experiment':testdf['experiment'][:nsamp]})
logger.info(eqpred_cossim.mean())
logger.info(tstpreddf.groupby('exptype')['eqpred_cossim'].mean())
logger.info(tstpreddf.groupby('experiment')['eqpred_cossim'].mean())


# Run : python post_processing_leak.py ttsexp_mask_v31_cosv5_fold5.csv ttsexp_mask_v31_cosv5_fold5_ddlleak.csv 
#ttsleak = pd.read_csv(os.path.join( path_data, 'ttsexp_mask_v31_cosv5_fold5_ddlleak.csv'))
