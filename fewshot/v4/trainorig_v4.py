import pandas as pd
import pickle
import numpy as np
import os
import sys
from tqdm import tqdm
from scipy.stats.mstats import hmean
import collections
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
from sklearn.model_selection import KFold
sys.path.append('/Users/dhanley2/Documents/Personal/recursion/')
from logs import get_logger
from utils import dumpobj, loadobj, GradualWarmupScheduler


class EmbeddingDS(D.Dataset):
    def __init__(self, df, embeddings, mode='train'):
        
        self.records = df.to_records(index=False)
        self.embedding = embeddings
        self.mode = mode
        self.len = df.shape[0]

    def __getitem__(self, index):

        if self.mode == 'train':
            return self.embedding[index], self.records[index].sirna_exp
        else:
            return self.embedding[index], self.records[index].id_code
    
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return np.array(res)

def embtrnavg(embtst, embtrn, sirnas, exp):
    embtrn = pd.DataFrame(embtrn, index = sirnas)
    embtrn['experiment'] = exp.apply(lambda x: x.split('-')[0]).values
    embtrn = embtrn.reset_index().groupby(['sirna', 'experiment']).mean()
    return embtrn

def expshot(embtst, embavg):
    sirnas = embavg.reset_index()['sirna'].values
    train_test_similarity = cosine_similarity(embtst, embavg)
    tts = train_test_similarity.transpose()
    tts = pd.DataFrame(tts, index = sirnas)
    tts = tts.reset_index().groupby('index').max()
    tts = tts.transpose().values
    return tts

path_data = '/Users/dhanley2/Documents/Personal/recursion/sub/tts/'
traindf = pd.read_csv( os.path.join( path_data, 'train.csv'))#.iloc[:3000]
testdf  = pd.read_csv( os.path.join( path_data, 'test.csv'))

'''
Params
'''
folds = pd.read_csv( os.path.join( path_data, 'folds.csv'))
traindf = traindf.merge(folds, on = 'experiment')
traindf = traindf.set_index('fold')
expdict = {'HUVEC' : 1, 'RPE' : 2, 'HEPG2' : 0, 'U2OS' : 3}
n_classes = (traindf['sirna'].max()+1)
batch_size = 16
lrmult = 20
fold = 0
EPOCHS = 20
WARMUP_EPOCHS = 10
WORK_DIR = path_data
WEIGHTS_NAME = 'cos_classifier_fold{}.pk'
# device=torch.device('cpu')
lr = 0.000025
logmsg = 'Train CosDist Fold {}'.format(fold)

# Print info about environments
logger = get_logger(logmsg, 'INFO') # noqa
logger.info('Cuda set up')


'''
Full validation
'''
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
    embavg = embtrnavg(embval, embtrn, dftrn.sirna, dftrn.experiment)
    traindf['pred_cossim'].loc[i] = expshot(embval, embavg).argmax(1)
    
traindf['eqpred_cossim'] = (traindf['pred_cossim']==traindf['sirna']).astype(np.int8)
traindf['exptype'] = traindf['experiment'].apply(lambda x: x.split('-')[0])
print(50*'-')
print(traindf['eqpred_cossim'].mean())
print(50*'-')
print(traindf.groupby('exptype')['eqpred_cossim'].mean())
print(50*'-')
print(traindf.groupby('fold')['eqpred_cossim'].mean())
'''
#fold
#0    0.886631
#1    0.826008
#2    0.723536
#3    0.895938
#4    0.910008

'''
Load correct fold
'''
for i in [fold]:
    validx = ~traindf['experiment'].isin(folds[folds['fold']==i]['experiment'].tolist()).values
    embtrnls = loadobj(os.path.join( path_data, '_emb_trn_probs_512_fold{}.pk'.format(i)))
    embvalls = loadobj(os.path.join( path_data, '_emb_val_probs_512_fold{}.pk'.format(i)))
    embtrn = (sum(embtrnls)/len(embtrnls))[validx]
    embval = sum(embvalls)/len(embvalls)
    dftrn = loadobj(os.path.join( path_data, '_df_trn_probs_512_fold{}.pk'.format(i)))
    dfval = loadobj(os.path.join( path_data, '_df_val_probs_512_fold{}.pk'.format(i)))
    dftrn = dftrn[validx]
    embavg = embtrnavg(embval, embtrn, dftrn.sirna, dftrn.experiment)


'''
Create Data Loaders
'''

dftrn['expkey'] = dftrn['experiment'].apply(lambda x: expdict[x.split('-')[0]])
dfval['expkey'] = dfval['experiment'].apply(lambda x: expdict[x.split('-')[0]])
dftrn['sirna_exp'] = dftrn['sirna'] + ( dftrn['expkey'] * n_classes )

ds_trn = EmbeddingDS(dftrn, embtrn)
ds_val = EmbeddingDS(dfval, embval, mode='val')

trnloader = D.DataLoader(ds_trn, batch_size=batch_size, shuffle=True, num_workers=32)
valloader = D.DataLoader(ds_val, batch_size=batch_size*8, shuffle=False, num_workers=32)
y_trn = dftrn.sirna
y_val = dfval.sirna

'''
Create Model
'''
class CosClassifier(nn.Module):
    def __init__(self, embavg):
        
        super().__init__()
        self.cosdist = nn.CosineSimilarity(dim=2, eps=1e-4)
        embavg = torch.tensor(embavg)
        self.cosclassifier = torch.nn.Parameter(F.normalize(embavg, p=2, \
                                         dim=embavg.dim()-1, eps=1e-12))

    def forward(self, x):        
        
        # Normalise
        outemb = F.normalize(x, p=2, dim=x.dim()-1, eps=1e-12)
        # Cosine distance output
        outdist = self.cosdist(outemb.unsqueeze(1), self.cosclassifier.unsqueeze(0).repeat(outemb.size(0),1, 1))
        return outdist
    
idxcol = ['experiment', 'sirna']
embavg = embavg.reset_index().sort_values(idxcol).set_index(idxcol)

model = CosClassifier(embavg.values)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)

scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=lrmult, total_epoch=WARMUP_EPOCHS, after_scheduler=scheduler_cosine)


logger.info('Start training')
tlen = len(trnloader)

for epoch in range(EPOCHS):
    scheduler_warmup.step()
    tloss = 0
    model.train()
    acc = np.zeros(1)

    for param_group in optimizer.param_groups:
        logger.info('Epoch: {} lr: {}'.format(epoch+1, param_group['lr']))
    for t, (x, y) in tqdm(enumerate(trnloader), total = len(trnloader)):
        x = x.to(device)
        y = y.cpu()
        optimizer.zero_grad()
        input_var = torch.autograd.Variable(x, requires_grad=True)
        target_var = torch.autograd.Variable(y)
        outdist  = model(input_var)
        loss = criterion(outdist, target_var)
        loss.backward()
        optimizer.step()
        tloss += loss.item()
        acc += accuracy(outdist.cpu(), y.cpu())
        del loss, outdist, y, x# , target
    output_model_file = os.path.join( WORK_DIR, WEIGHTS_NAME.format(epoch))
    cosclsdf = pd.DataFrame(model.cosclassifier.detach().numpy(), index = embavg.index)
    outmsg = 'Epoch {} -> Train Loss: {:.4f}, ACC: {:.2f}%'.format(epoch+1, tloss/tlen, acc[0]/tlen)
    logger.info('{}'.format(outmsg))
    cosclsdf.to_pickle(output_model_file)
    
    logger.info('Validation fold {}'.format(epoch))
    model.eval()
    y_pred_val = []
    y_probs_val = []
    for t, (x, y) in tqdm(enumerate(valloader), total = len(valloader)):
        x = x.to(device) 
        outdist = model(x)
        outprobs = outdist.detach().numpy()
        y_pred_val += outprobs.argmax(1).tolist()
        y_probs_val.append(outprobs)
    y_probs_val = np.concatenate(y_probs_val, 0)
    y_pred_val = np.array(y_pred_val)%1108
    match = y_pred_val == y_val.values
    logger.info('Epoch {} Accuracy {}'.format(epoch, match.sum()/len(match)))
        