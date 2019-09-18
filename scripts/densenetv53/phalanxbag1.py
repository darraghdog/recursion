from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import optparse
import os, sys
import numpy as np 
import pandas as pd
from PIL import Image
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
from sklearn.model_selection import KFold
from scipy.stats.mstats import hmean
from sklearn.metrics.pairwise import cosine_similarity

import cv2
import gc
import random
import logging
import datetime

import torchvision
from torchvision import transforms as T

from albumentations import (Cutout, Compose, Normalize, RandomRotate90, HorizontalFlip,
                           VerticalFlip, ShiftScaleRotate, Transpose, OneOf, IAAAdditiveGaussianNoise,
                           GaussNoise, RandomGamma, RandomContrast, RandomBrightness, HueSaturationValue,
                           RandomCrop, Lambda, NoOp, CenterCrop, Resize
                           )

from tqdm import tqdm
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
from apex.optimizers.fused_adam import FusedAdam


import warnings
warnings.filterwarnings('ignore')

# Print info about environments
parser = optparse.OptionParser()
parser.add_option('-s', '--seed', action="store", dest="seed", help="model seed", default="1234")
parser.add_option('-o', '--fold', action="store", dest="fold", help="Fold for split", default="0")
parser.add_option('-p', '--nbags', action="store", dest="nbags", help="Number of bags for averaging", default="0")
parser.add_option('-e', '--epochs', action="store", dest="epochs", help="epochs", default="5")
parser.add_option('-b', '--batchsize', action="store", dest="batchsize", help="batch size", default="16")
parser.add_option('-r', '--rootpath', action="store", dest="rootpath", help="root directory", default="/share/dhanley2/recursion/")
parser.add_option('-i', '--imgpath', action="store", dest="imgpath", help="root directory", default="data/mount/512X512X6/")
parser.add_option('-w', '--workpath', action="store", dest="workpath", help="Working path", default="densenetv1/weights")
parser.add_option('-f', '--weightsname', action="store", dest="weightsname", help="Weights file name", default="pytorch_model.bin")
parser.add_option('-c', '--customwt', action="store", dest="customwt", help="Weight of annotator count in loss", default="1.0")
parser.add_option('-l', '--lr', action="store", dest="lr", help="learning rate", default="0.00005")
parser.add_option('-t', '--lrmult', action="store", dest="lrmult", help="learning rate multiplier", default="4")
parser.add_option('-u', '--cutmix_prob', action="store", dest="cutmix_prob", help="Cutmix probability", default="0")
parser.add_option('-a', '--beta', action="store", dest="beta", help="Cutmix beta", default="0")
parser.add_option('-n', '--probsname', action="store", dest="probsname", help="probs file name", default="probs_256")
parser.add_option('-g', '--logmsg', action="store", dest="logmsg", help="root directory", default="Recursion-pytorch")
parser.add_option('-j', '--precision', action="store", dest="precision", help="root directory", default="half")
parser.add_option('-z', '--emb', action="store", dest="emb", help="Return Embedding", default="1")
parser.add_option('-y', '--site', action="store", dest="site", help="Site images set", default="0")


options, args = parser.parse_args()
package_dir = options.rootpath
sys.path.append(package_dir)
from logs import get_logger
from utils import dumpobj, loadobj, GradualWarmupScheduler


# Print info about environments
logger = get_logger(options.logmsg, 'INFO') # noqa
logger.info('Cuda set up : time {}'.format(datetime.datetime.now().time()))

device=torch.device('cuda')
logger.info('Device : {}'.format(torch.cuda.get_device_name(0)))
logger.info('Cuda available : {}'.format(torch.cuda.is_available()))
n_gpu = torch.cuda.device_count()
logger.info('Cuda n_gpus : {}'.format(n_gpu ))


logger.info('Load params : time {}'.format(datetime.datetime.now().time()))
for (k,v) in options.__dict__.items():
    logger.info('{}{}'.format(k.ljust(20), v))


cutmix_prob = float(options.cutmix_prob)
beta = float(options.beta)
SEED = int(options.seed)
EMB = int(options.emb)
SITE = int(options.site)

EPOCHS = int(options.epochs)
lr=float(options.lr)
lrmult=int(options.lrmult)
batch_size = int(options.batchsize)
ROOT = options.rootpath
path_data = os.path.join(ROOT, 'data')
path_img = os.path.join(ROOT, options.imgpath)
WORK_DIR = os.path.join(ROOT, options.workpath)
#WORK_DIR = os.path.join('/data/sdsml_prod/projects/data/ldc/recursion', options.workpath)
WEIGHTS_NAME = options.weightsname
PROBS_NAME = options.probsname
PRECISION = options.precision
fold = int(options.fold)
nbags= int(options.nbags)
#classes = 1109
device = 'cuda'
CONTROL = False

'''
# Check directory exists
CHKDIR= WORK_DIR
if os.path.exists(CHKDIR):
    logger.info('Path EXISTS!!!')
    logger.info(CHKDIR)
else:
    logger.info('Path doesnot exist!!!')
    #logger.info(os.listdir(os.path.join( WORK_DIR, '../mount1')))
    logger.info(CHKDIR)
    #break
'''

os.environ["TORCH_HOME"] = os.path.join( path_data, 'mount')
logger.info(os.system('$TORCH_HOME'))


print('Data path : {}'.format(path_data))
print('Image path : {}'.format(path_img))

class ImagesDS(D.Dataset):
    def __init__(self, df, img_dir, mode='train', channels=[1,2,3,4,5,6]):
        
        #df = pd.read_csv(csv_file)
        self.records = df.to_records(index=False)
        self.channels = channels
        #self.site = site
        self.mode = mode
        self.transform = test_aug()
        self.img_dir = img_dir
        self.len = df.shape[0]
        logger.info('ImageDS Shape')
        logger.info(self.len)
    
    @staticmethod
    def _load_img_as_tensor(file_name, mean_, sd_, illum_correction, transform):
        img = loadobj(file_name)
        #img = transform(image = img)['image']
        img = img.astype(np.float32)
        img = img / illum_correction     
        img = transform(image = img)['image']
        img = torch.from_numpy(np.moveaxis(img, -1, 0).astype(np.float32))
        img /= 255.
        img = T.Normalize([*list(mean_)], [*list(sd_)])(img)
        return img  

    def _get_np_path(self, index, site):
        experiment, well, plate, mode = self.records[index].experiment, \
                                        self.records[index].well, \
                                        self.records[index].plate, \
                                        self.records[index].mode
        # ,'mount1/512X512X6'
        return '/'.join([self.img_dir,mode,experiment,f'Plate{plate}',f'{well}_s{site}_w.pk'])
    
    def __getitem__(self, index):
        if SITE != 0:
            pathnp1 = self._get_np_path(index, site = 1)
            pathnp2 = self._get_np_path(index, site = 2)
        else:
            pathnp1 = self._get_np_path(index, site = SITE)
            pathnp2 = self._get_np_path(index, site = SITE)
        pathnp1 = self._get_np_path(index, site = 1)
        pathnp2 = self._get_np_path(index, site = 2)
        experiment, plate, _ = pathnp1.split('/')[-3:]
        stats_key = '{}/{}/{}'.format(experiment, plate[-1], self.records[index].mode )
        rand_filter = random.randint(0,2)
        stats_dict = illumpk[rand_filter][stats_key]

        img1 = self._load_img_as_tensor(pathnp1, 
                                       stats_dict['mean'], 
                                       stats_dict['std'], 
                                       stats_dict['illum_correction_function'],
                                       self.transform)
        img2 = self._load_img_as_tensor(pathnp2, 
                                       stats_dict['mean'], 
                                       stats_dict['std'], 
                                       stats_dict['illum_correction_function'],
                                       self.transform)
        if random.randint(0,1)==1:
            img = np.hstack((img1, img2))
        else:
            img = np.hstack((img2, img1))
        if self.mode in ['train', 'val' ]:
            return img, self.records[index].sirna
        else:
            return img, self.records[index].id_code

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def add_sites(df):
    df1 = df.copy()
    df2 = df.copy()
    df1['site'] = 1
    df2['site'] = 2
    return pd.concat([df1, df2], 0)

def test_aug(p=1.):
    return Compose([
        RandomRotate90(),
        HorizontalFlip(),
        VerticalFlip(),
        Transpose(),
        NoOp(),
    ], p=p)

def train_aug(p=1.):
    return Compose([
        RandomRotate90(),
        HorizontalFlip(),
        VerticalFlip(),
        Transpose(),
        Cutout(
            num_holes=8,
            max_h_size=24,
            max_w_size=24,
            fill_value=0,
            always_apply=False,
            p=0.3,
        ),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                         rotate_limit=45, p=0.5, border_mode = cv2.BORDER_REPLICATE),
    ], p=p)

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

class DensNet(nn.Module):
    def __init__(self, num_classes=1000, num_channels=6):
        super().__init__()
        preloaded = torchvision.models.densenet121(pretrained=True)
        self.features = preloaded.features
        self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)
        self.classifier = nn.Linear(1024, num_classes, bias=True)
        del preloaded
        
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        outcls = self.classifier(out)
        return  out, outcls

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

@torch.no_grad()
def prediction(model, loader):
    embls = []
    clsls = []
    tlen = len(loader)
    for t, (x, _) in enumerate(loader):
        if t%100==0:
            logger.info('Predict step {} of {}'.format(t, tlen))
        x = x.to(device)#.half()
        emb, cls = model(x)#.float()
        embls.append(emb.cpu().numpy())
        clsls.append(cls.cpu().numpy())
    embmat, clsmat = np.concatenate(embls, 0), np.concatenate(clsls, 0)
    return embmat, clsmat

logger.info('Augmentation set up : time {}'.format(datetime.datetime.now().time()))

#transform = train_aug()


logger.info('Load Dataframes')
train_dfall = pd.read_csv( os.path.join( path_data, 'train.csv'))#.iloc[:3000]
test_df  = pd.read_csv( os.path.join( path_data, 'test.csv'))#.iloc[:300]
huvec18_df = pd.read_csv( os.path.join( path_data, 'huvec18.csv'))#.iloc[:300]
train_ctrl = pd.read_csv(os.path.join(path_data, 'train_controls.csv'))
test_ctrl = pd.read_csv(os.path.join(path_data, 'test_controls.csv'))
train_dfall['mode'] = train_ctrl['mode'] = 'train'
test_df['mode'] = test_ctrl['mode'] = 'test'
huvec18_df['mode'] = 'test'
bestdf = pd.read_csv( os.path.join( path_data, 'tmp.csv'))
logger.info(bestdf.shape)
u2idx = bestdf.id_code.str.contains('U2OS')
logger.info(u2idx.mean())



folddf  = pd.read_csv( os.path.join( path_data, 'folds.csv'))
train_dfall = pd.merge(train_dfall, folddf, on = 'experiment' )
train_ctrl = pd.merge(train_ctrl, folddf, on = 'experiment' )
statsdf = pd.read_csv( os.path.join( path_data, 'stats.csv'))

logger.info('Load illumination stats')
illumfiles = dict((i, 'mount/illumsttats_fs{}_{}.pk'.format((2**(i+4)), options.imgpath.split('/')[2])) for i in range(3))
logger.info([os.path.join( path_data, v) for v in illumfiles.values()] )
illumpk = dict((i, loadobj(os.path.join( path_data, illumfiles[i]))) for i in range(3))
logger.info([i for i in illumpk[0].keys()][:5])

logger.info('Calculate stats')
statsdf['experiment'] = statsdf['FileName'].apply(lambda x: x.split('/')[-3])
statsdf['plate'] = statsdf['FileName'].apply(lambda x: x.split('/')[-2])
statsdf['fname'] = statsdf['FileName'].apply(lambda x: x.split('/')[-1])
statsgrpdf = statsdf.groupby(['experiment', 'plate', 'Channel'])['Mean', 'Std'].mean()
experiment_, plate_, _ = ['HEPG2-01', 'Plate1', 'B03_s1_w1.png']
stats_dict = statsgrpdf.loc[(experiment_, plate_)].to_dict()
print(stats_dict)


logger.info('******** Checking Fold Counts **********')
logger.info(train_dfall['fold'].value_counts())

traindf = train_dfall[train_dfall['fold']!=fold]
validdf = train_dfall[train_dfall['fold']==fold]
if validdf.shape[0]==0:
    validdf = huvec18_df
y_val = validdf.sirna.values

# Add the controls
#train_ctrl.sirna = 1108
#test_ctrl.sirna = 1108

trainfull = pd.concat([traindf, 
                       train_ctrl.drop('well_type', 1), 
                       train_ctrl.drop('well_type', 1),
                       test_ctrl.drop('well_type', 1),
                       test_ctrl.drop('well_type', 1)], 0)

classes = trainfull.sirna.max() + 1

# ds = ImagesDS(traindf, path_data)
ds = ImagesDS(trainfull, path_img)
ds_trn = ImagesDS(traindf, path_img)
ds_val = ImagesDS(validdf, path_img, mode='val')
ds_test = ImagesDS(test_df, path_img, mode='test')

dfctrl  =  pd.concat([train_ctrl.drop('well_type', 1),
                       test_ctrl.drop('well_type', 1)], 0)
ds_ctrl = ImagesDS(dfctrl, path_img)
logger.info('Set up model')

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if n_gpu > 0:
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

model = DensNet(num_classes=classes)

loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=16)
trnloader = D.DataLoader(ds_trn, batch_size=batch_size*4, shuffle=False, num_workers=16)
valloader = D.DataLoader(ds_val, batch_size=batch_size*4, shuffle=False, num_workers=16)
tstloader = D.DataLoader(ds_test, batch_size=batch_size*4, shuffle=False, num_workers=16)
ctrlloader = D.DataLoader(ds_ctrl, batch_size=batch_size*4, shuffle=False, num_workers=32)

logger.info('Start training')
trnembls = []
tstembls = []
ctrlembls = []
trnclsls = []
tstclsls = []
ctrlclsls = []




for epoch in range(EPOCHS-nbags, EPOCHS):
    input_model_file = os.path.join( WORK_DIR, WEIGHTS_NAME.replace('.bin', '')+str(epoch)+'.bin'  )
    logger.info(input_model_file)
    model = DensNet(num_classes=classes)
    model.to(device)
    model.load_state_dict(torch.load(input_model_file))
    model.to(device)
    for param in model.parameters():
        param.requires_grad=False
    logger.info('Train file {}'.format(input_model_file))
    # Save raw embeddings

    for bag in range(40):
        logger.info('Infer bag {}'.format(bag))
        embtst, clstst = prediction(model, tstloader)
        logger.info('Epoch {} score'.format(bag))
        logger.info((clstst[u2idx].argmax(1)==bestdf.sirna.values[u2idx]).mean())
        embctrl, clsctrl = prediction(model, ctrlloader)
        embtst, clstst = prediction(model, tstloader) 
        embtrn, clstrn = prediction(model, trnloader)
        tstembls.append(embtst)
        trnembls.append(embtrn)
        ctrlembls.append(embctrl)
        tstclsls.append(clstst)
        trnclsls.append(clstrn)
        ctrlclsls.append(clsctrl)
        clsbag = sum(tstclsls)/len(tstclsls)
        logger.info('Bag {} score'.format(bag))
        logger.info((clstst[u2idx].argmax(1)==bestdf.sirna.values[u2idx]).mean())

if True:
    dumpobj(os.path.join( WORK_DIR, '_emb_site{}_ctrl_{}_fold{}.pk'.format(SITE, PROBS_NAME, fold)), ctrlembls)
    dumpobj(os.path.join( WORK_DIR, '_emb_site{}_trn_{}_fold{}.pk'.format(SITE, PROBS_NAME, fold)), trnembls)
    dumpobj(os.path.join( WORK_DIR, '_emb_site{}_tst_{}_fold{}.pk'.format(SITE,  PROBS_NAME, fold)), tstembls)
    dumpobj(os.path.join( WORK_DIR, '_cls_site{}_ctrl_{}_fold{}.pk'.format(SITE, PROBS_NAME, fold)), ctrlclsls)
    dumpobj(os.path.join( WORK_DIR, '_cls_site{}_trn_{}_fold{}.pk'.format(SITE,  PROBS_NAME, fold)), trnclsls)
    dumpobj(os.path.join( WORK_DIR, '_cls_site{}_tst_{}_fold{}.pk'.format(SITE,  PROBS_NAME, fold)), tstclsls)
    dumpobj(os.path.join( WORK_DIR, '_df_site{}_trn_{}_fold{}.pk'.format(SITE,  PROBS_NAME, fold)), traindf)
    dumpobj(os.path.join( WORK_DIR, '_df_site{}_tst_{}_fold{}.pk'.format(SITE,  PROBS_NAME, fold)), test_df)
    dumpobj(os.path.join( WORK_DIR, '_df_site{}_ctrl_{}_fold{}.pk'.format(SITE, PROBS_NAME, fold)), dfctrl)
