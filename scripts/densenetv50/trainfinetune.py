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
                           RandomCrop, Lambda, NoOp, CenterCrop, Resize)

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
parser.add_option('-x', '--xtrasteps', action="store", dest="xtrasteps", help="Fine tune epochs", default="10")
parser.add_option('-y', '--exp_filter', action="store", dest="exp_filter", help="Fine tune experiment", default="U2")
parser.add_option('-m', '--dump_emb', action="store", dest="dump_emb", help="Number of epochs to dump embeddings", default="0")


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


logger.info('Set up model')
SEED = int(options.seed)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if n_gpu > 0:
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


logger.info('Load params : time {}'.format(datetime.datetime.now().time()))
for (k,v) in options.__dict__.items():
    logger.info('{}{}'.format(k.ljust(20), v))

'''
Load params
'''
cutmix_prob = float(options.cutmix_prob)
beta = float(options.beta)
EPOCHS = int(options.epochs)
XTRASTEPS = int(options.xtrasteps)
EXPERIMENTFILTER = options.exp_filter
dump_emb = int(options.dump_emb)
lr=float(options.lr)
lrmult=int(options.lrmult)
batch_size = int(options.batchsize)
ROOT = options.rootpath
path_data = os.path.join(ROOT, 'data')
path_img = os.path.join(ROOT, options.imgpath)
WORK_DIR = os.path.join(ROOT, options.workpath)
WEIGHTS_NAME = options.weightsname
PROBS_NAME = options.probsname
PRECISION = options.precision
fold = int(options.fold)
nbags= int(options.nbags)
device = 'cuda'

os.environ["TORCH_HOME"] = os.path.join( path_data, 'mount')
logger.info(os.system('$TORCH_HOME'))


class ImagesDS(D.Dataset):
    def __init__(self, df, img_dir, mode='train', channels=[1,2,3,4,5,6]):
        
        self.records = df.to_records(index=False)
        self.channels = channels
        self.mode = mode
        self.transform = train_aug()
        if self.mode != 'train' : self.transform = test_aug()
        self.img_dir = img_dir
        self.len = df.shape[0]
        logger.info('ImageDS Shape')
        logger.info(self.len)
    
    @staticmethod
    def _load_img_as_tensor(file_name, mean_, sd_, illum_correction, transform):
        img = loadobj(file_name)
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
        return '/'.join([self.img_dir,mode,experiment,f'Plate{plate}',f'{well}_s{site}_w.pk'])
    
    def __getitem__(self, index):
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
            p=0.8,
        ),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, 
                         rotate_limit=45, p=0.8, border_mode = cv2.BORDER_REPLICATE),
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
        out = self.classifier(out)
        return out
    
class DensNetEmbedding(nn.Module):
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
        return  out

@torch.no_grad()
def predictionEmbedding(model, loader):
    preds = np.empty(0)
    probs = []
    tlen = len(loader)
    for t, (x, _) in enumerate(loader):
        if t%100==0:
            logger.info('Predict step {} of {}'.format(t, tlen))
        x = x.to(device)#.half()
        output = model(x)#.float()
        probs.append(output.cpu().numpy())
    probs = np.concatenate(probs, 0)
    return probs

def evalmodel(model, epoch):
    model.eval()
    preds, probs = prediction(model, vloader)
    predsmax = np.argmax(probs[:,:1108], 1)
    matchesmax = (predsmax.flatten().astype(np.int32) == y_val.flatten().astype(np.int32)).sum()
    acc = matchesmax/predsmax.shape[0]
    outmsg = 'Epoch {} -> Fold {} -> Accuracy Ep Max: {:.4f}'.format(epoch+1, fold, acc)
    logger.info('{}'.format(outmsg))
    return acc


logger.info('Load Dataframes')
train_dfall = pd.read_csv( os.path.join( path_data, 'train.csv'))#.iloc[:3000]
test_df  = pd.read_csv( os.path.join( path_data, 'test.csv'))#.iloc[:300]
huvec18_df = pd.read_csv( os.path.join( path_data, 'huvec18.csv'))#.iloc[:300]
train_ctrl = pd.read_csv(os.path.join(path_data, 'train_controls.csv'))
test_ctrl = pd.read_csv(os.path.join(path_data, 'test_controls.csv'))
train_dfall['mode'] = train_ctrl['mode'] = 'train'
test_df['mode'] = test_ctrl['mode'] = 'test'
huvec18_df['mode'] = 'test'

logger.info('Add folds to dataframe')
folddf  = pd.read_csv( os.path.join( path_data, 'folds.csv'))
train_dfall = pd.merge(train_dfall, folddf, on = 'experiment' )
train_ctrl = pd.merge(train_ctrl, folddf, on = 'experiment' )

logger.info('Load illumination & img stats')
statsdf = pd.read_csv( os.path.join( path_data, 'stats.csv'))
illumfiles = dict((i, 'mount/illumsttats_fs{}_{}.pk'.format((2**(i+4)), options.imgpath.split('/')[2])) for i in range(3))
logger.info([os.path.join( path_data, v) for v in illumfiles.values()] )
illumpk = dict((i, loadobj(os.path.join( path_data, illumfiles[i]))) for i in range(3))
logger.info([i for i in illumpk[0].keys()][:5])
statsdf['experiment'] = statsdf['FileName'].apply(lambda x: x.split('/')[-3])
statsdf['plate'] = statsdf['FileName'].apply(lambda x: x.split('/')[-2])
statsdf['fname'] = statsdf['FileName'].apply(lambda x: x.split('/')[-1])
statsgrpdf = statsdf.groupby(['experiment', 'plate', 'Channel'])['Mean', 'Std'].mean()
experiment_, plate_, _ = ['HEPG2-01', 'Plate1', 'B03_s1_w1.png']
stats_dict = statsgrpdf.loc[(experiment_, plate_)].to_dict()
print(stats_dict)


logger.info('Create val and train sets for {}, oversample control just because...'.format(EXPERIMENTFILTER))
traindf = train_dfall[train_dfall['fold']!=fold]
validdf = train_dfall[train_dfall['fold']==fold]
trainfull = pd.concat([traindf, 
                       train_ctrl.drop('well_type', 1), 
                       train_ctrl.drop('well_type', 1),
                       test_ctrl.drop('well_type', 1),
                       test_ctrl.drop('well_type', 1)], 0)
classes = trainfull.sirna.max() + 1


logger.info('Limit to {}'.format(EXPERIMENTFILTER))
logger.info(trainfull.shape, validdf.shape, test_df.shape)
trainfull = trainfull[trainfull.experiment.str.contains(EXPERIMENTFILTER)]
validdf = validdf[validdf.experiment.str.contains(EXPERIMENTFILTER)]
test_df = test_df[test_df.experiment.str.contains(EXPERIMENTFILTER)]
logger.info(trainfull.shape, validdf.shape, test_df.shape)

logger.info('Set up torch loaders')
ds = ImagesDS(trainfull, path_img)
ds_val = ImagesDS(validdf, path_img, mode='val')
ds_test = ImagesDS(test_df, path_img, mode='test')
loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=5)
vloader = D.DataLoader(ds_val, batch_size=batch_size*4, shuffle=False, num_workers=5)
tloader = D.DataLoader(ds_test, batch_size=batch_size*4, shuffle=False, num_workers=5)
if fold==5:
    validdf = huvec18_df # Use leak test set for validation
y_val = validdf.sirna.values

logger.info('If no val for this exp in the fold- kill the job ')
if validdf.shape[0]==0:
    logger.info('KILLME! '* 50)

logger.info('Get scores from previous run')
val_acc = []
trn_acc = []
val_loss = []
trn_loss = []
epochls = []
for epoch in range(EPOCHS-10, EPOCHS):
    input_model_file = os.path.join( WORK_DIR, WEIGHTS_NAME.replace('.bin', '')+str(epoch)+'.bin'  )
    logger.info(input_model_file)
    model = DensNet(num_classes=classes)
    model.to(device)
    model.load_state_dict(torch.load(input_model_file))
    if fold != 5:         
        acc, vloss = evalmodel(model, epoch)
        val_acc.append(acc)
        trn_acc.append(0)
        val_loss.append(0)
        trn_loss.append(0)
        epochls.append(epoch)
        scoresdf = pd.DataFrame({'experiment' : [EXPERIMENTFILTER]*len(val_acc), \
                                 'epoch' : epochls, \
                                 'val_acc' : val_acc, \
                                 'trn_acc' : trn_acc, \
                                 'val_loss' : val_loss, \
                                 'trn_loss' : trn_loss})
        scoresdf.to_csv(os.path.join( WORK_DIR, 'scores_{}_fold{}.csv'.format(EXPERIMENTFILTER, fold)), index = False)

criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)

model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False, loss_scale="dynamic")


logger.info('Start training')
tlen = len(loader)
probsls = []
probststls = []
ep_accls = []
for epoch in range(EPOCHS, EPOCHS+XTRASTEPS):
    tloss = 0
    model.train()
    acc = np.zeros(1)
    for param_group in optimizer.param_groups:
        logger.info('Epoch: {} lr: {} Cutmix prob: {}'.format(epoch+1, param_group['lr'], cutmix_prob))  
    for tt, (x, y) in enumerate(loader): 
        x = x.to(device)
        y = y.cuda()
        # cutmix
        optimizer.zero_grad()
        r = np.random.rand(1)
        if beta > 0 and r < cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(x.size()[0]).cuda()
            target_a = y
            target_b = y[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            ## Cutmix
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            # compute output
            input_var = torch.autograd.Variable(x, requires_grad=True)
            target_a_var = torch.autograd.Variable(target_a)
            target_b_var = torch.autograd.Variable(target_b)
            output = model(input_var)
            loss = criterion(output, target_a_var) * lam + criterion(output, target_b_var) * (1. - lam)
        else:
            # compute output
            input_var = torch.autograd.Variable(x, requires_grad=True)
            target_var = torch.autograd.Variable(y)
            output = model(input_var)
            loss = criterion(output, target_var)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        tloss += loss.item()        
        acc += accuracy(output.cpu(), y.cpu())
        del loss, output, y, x
    output_model_file = os.path.join( WORK_DIR, WEIGHTS_NAME.replace('.bin', '')+str(epoch)+'.bin'  )
    if dump_emb:
        torch.save(model.state_dict(), output_model_file) 
    outmsg = 'Epoch {} -> Train Loss: {:.4f}, ACC: {:.2f}%'.format(epoch+1, tloss/tlen, acc[0]/tlen)
    logger.info('{}'.format(outmsg))

    if fold != 5: 
        acc, vloss = evalmodel(model, epoch)
        val_acc.append(acc)
        trn_acc.append(acc[0]/tlen)
        val_loss.append(vloss)
        trn_loss.append(tloss/tlen)
        epochls.append(epoch)
        scoresdf = pd.DataFrame({'experiment' : [EXPERIMENTFILTER]*len(val_acc), \
                                 'epoch' : epochls, \
                                 'val_acc' : val_acc, \
                                 'trn_acc' : trn_acc, \
                                 'val_loss' : val_loss, \
                                 'trn_loss' : trn_loss})
        scoresdf.to_csv(os.path.join( WORK_DIR, 'scores_{}_fold{}.csv'.format(EXPERIMENTFILTER, fold)), index = False)

if dump_emb>0:
    logger.info('Save embeddings in last {} epochs'.format(dump_emb))
    rembls = []
    vembls = []
    tembls = []
    for epoch in range((EPOCHS+XTRASTEPS)-dump_emb, (EPOCHS+XTRASTEPS)):
        input_model_file = os.path.join( WORK_DIR, WEIGHTS_NAME.replace('.bin', '')+str(epoch)+'.bin'  )
        logger.info(input_model_file)
        model = DensNetEmbedding(num_classes=classes)
        model.to(device)
        model.load_state_dict(torch.load(input_model_file))
        model.to(device)
        for param in model.parameters():
            param.requires_grad=False
        logger.info('Train file {}'.format(input_model_file))
        # Save raw embeddings
        rembls.append(predictionEmbedding(model, loader))
        dumpobj(os.path.join( WORK_DIR, '_emb_{}_trn_{}_fold{}.pk'.format(EXPERIMENTFILTER, PROBS_NAME, fold)), rembls)
        dumpobj(os.path.join( WORK_DIR, '_df_{}_trn_{}_fold{}.pk'.format(EXPERIMENTFILTER, PROBS_NAME, fold)), trainfull)
        if fold!=5: 
            vembls.append(predictionEmbedding(model, vloader))
            dumpobj(os.path.join( WORK_DIR, '_emb_u2_val_{}_fold{}.pk'.format(EXPERIMENTFILTER, PROBS_NAME, fold)), vembls)
            dumpobj(os.path.join( WORK_DIR, '_df_{}_val_{}_fold{}.pk'.format(EXPERIMENTFILTER, PROBS_NAME, fold)), validdf)
        else: 
            tembls.append(predictionEmbedding(model, tloader))
            dumpobj(os.path.join( WORK_DIR, '_emb_{}_tst_{}_fold{}.pk'.format(EXPERIMENTFILTER, PROBS_NAME, fold)), tembls)    
            dumpobj(os.path.join( WORK_DIR, '_df_{}_tst_{}_fold{}.pk'.format(EXPERIMENTFILTER, PROBS_NAME, fold)), test_df)


        
