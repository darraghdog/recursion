from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import optparse
import os, sys
import numpy as np 
import pandas as pd
from PIL import Image
import torch
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
                           RandomCrop, Lambda, NoOp, CenterCrop, Resize
                           )

from tqdm import tqdm

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
parser.add_option('-l', '--lr', action="store", dest="lr", help="learning rate", default="0.00003")
#parser.add_option('-d', '--datapath', action="store", dest="datapath", help="root directory", default="/data/mount/512X512X6/")
parser.add_option('-u', '--cutmix_prob', action="store", dest="cutmix_prob", help="Cutmix probability", default="0")
parser.add_option('-a', '--beta', action="store", dest="beta", help="Cutmix beta", default="0")



options, args = parser.parse_args()
package_dir = options.rootpath
sys.path.append(package_dir)
from logs import get_logger
from utils import dumpobj, loadobj

# Print info about environments
logger = get_logger('Recursion-pytorch', 'INFO') # noqa
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
EPOCHS = int(options.epochs)
lr=float(options.lr)
batch_size = int(options.batchsize)
ROOT = options.rootpath
path_data = os.path.join(ROOT, 'data')
path_img = os.path.join(ROOT, options.imgpath)
WORK_DIR = os.path.join(ROOT, options.workpath)
WEIGHTS_NAME = options.weightsname
fold = int(options.fold)
nbags= int(options.nbags)
#classes = 1109
device = 'cuda'
print('Data path : {}'.format(path_data))
print('Image path : {}'.format(path_img))

class ImagesDS(D.Dataset):
    def __init__(self, df, img_dir, mode='train', site=1, channels=[1,2,3,4,5,6]):
        
        #df = pd.read_csv(csv_file)
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.transform = train_aug()
        if self.mode != 'train' : self.transform = test_aug()
        self.img_dir = img_dir
        self.len = df.shape[0]
    
    @staticmethod
    def _load_img_as_tensor(file_name, mean_, sd_, transform):
        img = loadobj(file_name)
        img = transform(image = img)['image']
        img = torch.from_numpy(np.moveaxis(img, -1, 0).astype(np.float32))
        img /= 255.
        img = T.Normalize([*list(mean_.values())], [*list(sd_.values())])(img)
        return img  

    def _get_np_path(self, index):
        #site = random.randint(1, 2)
        experiment, well, plate, mode, site = self.records[index].experiment, \
                                        self.records[index].well, \
                                        self.records[index].plate, \
                                        self.records[index].mode, \
                                        self.records[index].site
        # ,'mount1/512X512X6'
        return '/'.join([self.img_dir,mode,experiment,f'Plate{plate}',f'{well}_s{site}_w.pk'])
    
    def __getitem__(self, index):
        pathnp = self._get_np_path(index)
        experiment, plate, _ = pathnp.split('/')[-3:]
        stats_dict = statsgrpdf.loc[(experiment, plate)].to_dict()
        statsls = [(stats_dict['Mean'][c], stats_dict['Std'][c]) for c in self.channels]
        img = self._load_img_as_tensor(pathnp, stats_dict['Mean'], stats_dict['Std'], self.transform)
        
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

def add_sites(df):
    df1 = df.copy()
    df2 = df.copy()
    df1['site'] = 1
    df2['site'] = 2
    return pd.concat([df1, df2], 0)

def compress_sites(mat):
    return np.maximum( mat[:int(mat.shape[0]/2)] ,  mat[int(mat.shape[0]/2):]  )
    #return 0.5 * ( mat[:int(mat.shape[0]/2)] + mat[int(mat.shape[0]/2):] )

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
    preds = np.empty(0)
    probs = []
    for x, _ in loader:
        x = x.to(device)
        output = model(x)
        idx = output.max(dim=-1)[1].cpu().numpy()
        outmat = torch.sigmoid(output.cpu()).numpy()
        preds = np.append(preds, idx, axis=0)
        probs.append(outmat)
    probs = np.concatenate(probs, 0)
    print(probs.shape)    
    return preds, probs

logger.info('Create image loader : time {}'.format(datetime.datetime.now().time()))
if not os.path.exists(WORK_DIR):
    os.mkdir(WORK_DIR)
    
logger.info('Augmentation set up : time {}'.format(datetime.datetime.now().time()))

#transform = train_aug()


logger.info('Load Dataframes : time {}'.format(datetime.datetime.now().time()))
train_dfall = pd.read_csv( os.path.join( path_data, 'train.csv'))#.iloc[:3000]
test_df  = pd.read_csv( os.path.join( path_data, 'test.csv'))
train_ctrl = pd.read_csv(os.path.join(path_data, 'train_controls.csv'))
test_ctrl = pd.read_csv(os.path.join(path_data, 'test_controls.csv'))

train_dfall['mode'] = train_ctrl['mode'] = 'train'
test_df['mode'] = test_ctrl['mode'] = 'test'

folddf  = pd.read_csv( os.path.join( path_data, 'folds.csv'))
train_dfall = pd.merge(train_dfall, folddf, on = 'experiment' )
statsdf = pd.read_csv( os.path.join( path_data, 'stats.csv'))
if False: # Sample test run
    #samp = random.sample(range(train_dfall.shape[0]), 1500)
    train_dfall = train_dfall.iloc[np.where(train_dfall['sirna']<5)]
    test_df = test_df.iloc[:500]

logger.info('Calculate stats: time {}'.format(datetime.datetime.now().time()))

statsdf['experiment'] = statsdf['FileName'].apply(lambda x: x.split('/')[-3])
statsdf['plate'] = statsdf['FileName'].apply(lambda x: x.split('/')[-2])
statsdf['fname'] = statsdf['FileName'].apply(lambda x: x.split('/')[-1])
statsgrpdf = statsdf.groupby(['experiment', 'plate', 'Channel'])['Mean', 'Std'].mean()
experiment_, plate_, _ = ['HEPG2-01', 'Plate1', 'B03_s1_w1.png']
stats_dict = statsgrpdf.loc[(experiment_, plate_)].to_dict()
print(stats_dict)


traindf = train_dfall[train_dfall['fold']!=fold]
validdf = train_dfall[train_dfall['fold']==fold]
y_val = validdf.sirna.values

# Add the controls
#train_ctrl.sirna = 1108
#test_ctrl.sirna = 1108
trainfull = pd.concat([traindf, 
                       train_ctrl.drop('well_type', 1), 
                        test_ctrl.drop('well_type', 1)], 0)
classes = trainfull.sirna.max() + 1

trainfull = add_sites(trainfull)
validdf = add_sites(validdf)
test_df = add_sites(test_df)

# ds = ImagesDS(traindf, path_data)
ds = ImagesDS(trainfull, path_img)
ds_val = ImagesDS(validdf, path_img, mode='val')
ds_test = ImagesDS(test_df, path_img, mode='test')


logger.info('Set up model : time {}'.format(datetime.datetime.now().time()))

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if n_gpu > 0:
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

model = DensNet(num_classes=classes)
model.to(device)

loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
vloader = D.DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2)
tloader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)

criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

logger.info('Start training : time {}'.format(datetime.datetime.now().time()))
tlen = len(loader)
probsls = []
probststls = []
ep_accls = []
for epoch in range(EPOCHS):
    tloss = 0
    model.train()
    acc = np.zeros(1)
    for x, y in loader: 
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
            # Cutmixup :)
            '''
            x1 = lam * x + (1 - lam) * x[rand_index] 
            x2 = (1 - lam) * x + lam * x[rand_index]
            x1 = x2[:, :, bbx1:bbx2, bby1:bby2]
            '''
            ## Mixup
            #x = lam * x + (1 - lam) * x[rand_index]
            ## Cutmix
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            # compute output
            input_var = torch.autograd.Variable(x, requires_grad=True)
            #input_var = torch.autograd.Variable(x1, requires_grad=True)
            target_a_var = torch.autograd.Variable(target_a)
            target_b_var = torch.autograd.Variable(target_b)
            output = model(input_var)

            loss = criterion(output, target_a_var) * lam + criterion(output, target_b_var) * (1. - lam)
            ## Cutmixup 
            #loss = criterion(output, target_a_var) * 0.5 + criterion(output, target_b_var) * 0.5
        else:
            # compute output
            input_var = torch.autograd.Variable(x, requires_grad=True)
            target_var = torch.autograd.Variable(y)
            output = model(input_var)
            loss = criterion(output, target_var)

        loss.backward()
        optimizer.step()
        tloss += loss.item() 
        acc += accuracy(output.cpu(), y.cpu())
        del loss, output, y, x# , target
    output_model_file = os.path.join( WORK_DIR, WEIGHTS_NAME.replace('.bin', '')+str(epoch)+'.bin'  )
    #if epoch %5==0:
    #    torch.save(model.state_dict(), output_model_file)
    outmsg = 'Epoch {} -> Train Loss: {:.4f}, ACC: {:.2f}%'.format(epoch+1, tloss/tlen, acc[0]/tlen)
    logger.info('{} : time {}'.format(outmsg, datetime.datetime.now().time()))
    if epoch < 0:
        continue
    model.eval()
    print('Fold {} Bag {}'.format(fold, 1+len(probststls)))
    preds, probs = prediction(model, vloader)
    probsls.append(compress_sites(probs))
    preds, probs = prediction(model, tloader)
    probststls.append(compress_sites(probs))
    gc.collect()
    probsbag = sum(probsls)/len(probsls)
    # Only argmax the non controls
    probsbag = probsbag[:,:1108]
    predsmax = np.argmax(probsls[-1][:,:1108], 1)
    predsbagmax = np.argmax(probsbag, 1)
    #predsbag = single_pred(validdf, probsbag).astype(int)
    matchesmax = (predsmax.flatten().astype(np.int32) == y_val.flatten().astype(np.int32)).sum()
    matchesbagmax = (predsbagmax.flatten().astype(np.int32) == y_val.flatten().astype(np.int32)).sum()    
    #matchesbag = (predsbag.flatten().astype(np.int32) == y_val.flatten().astype(np.int32)).sum()    
    #outmsg = 'Epoch {} -> Fold {} -> Accuracy Sngl: {:.4f} -> Accuracy Max: {:.4f} - NPreds {}'.format(\
    #                epoch+1, fold, matchesbag/predsbag.shape[0], matchesbagmax/predsbag.shape[0], len(probsls))
    outmsg = 'Epoch {} -> Fold {} -> Accuracy Ep Max: {:.4f}  -> Accuracy Bag Max: {:.4f} - NPreds {}'.format(\
                    epoch+1, fold, matchesmax/predsmax.shape[0], matchesbagmax/predsbagmax.shape[0], len(probsls))
    logger.info('{} : time {}'.format(outmsg, datetime.datetime.now().time()))

dumpobj(os.path.join( WORK_DIR, 'val_prods_fold{}.pk'.format(fold)), probsls)
dumpobj(os.path.join( WORK_DIR, 'tst_prods_fold{}.pk'.format(fold)), probststls)

logger.info('Submission : time {}'.format(datetime.datetime.now().time()))
probsbag = sum(probststls)/len(probststls)
probsbag = probsbag[:,:1108]

# predsbag = np.argmax(probsbag, 1)
submission = pd.read_csv(path_data + '/test.csv')
submission['sirna'] = single_pred(submission, probsbag).astype(int)
# submission['sirna'] = predsbag.astype(int)
submission.to_csv('mixme_fold{}.csv'.format(fold), index=False, columns=['id_code','sirna'])

