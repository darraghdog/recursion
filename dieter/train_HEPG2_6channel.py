import datetime
import sys
import os
from tqdm import tqdm
import numpy as np
#from rxrx.io import convert_tensor_to_rgb
from skimage.io import imread, imsave
from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
import tensorflow as tf
import optparse
#!pip install albumentations
import albumentations as A
import torch
import random
import pickle

# Print info about environments
parser = optparse.OptionParser()
parser.add_option('-r', '--rootpath', action="store", dest="rootpath", help="root directory", default="/share/dhanley2/recursion")
#parser.add_option('-d', '--targetpath', action="store", dest="targetpath", help="root directory", default="data/128X128X6/cgan/")
parser.add_option('-t', '--datapath', action="store", dest="datapath", help="root directory", default="data/128X128X6")
parser.add_option('-m', '--modelpath', action="store", dest="modelpath", help="root directory", default="dieter/weights/128/2")
parser.add_option('-s', '--seed', action="store", dest="seed", help="model seed", default="1234")
parser.add_option('-f', '--foldsfile', action="store", dest="foldsfile", help="Folds File for split", default="folds.p")
parser.add_option('-n', '--normfile', action="store", dest="normfile", help="Normalisation File", default="experiment_normalizations128x128x6.p")
parser.add_option('-o', '--fold', action="store", dest="fold", help="Fold for split", default="0")
parser.add_option('-e', '--epochs', action="store", dest="epochs", help="Gan training epochs", default="15")
parser.add_option('-b', '--batchsize', action="store", dest="batchsize", help="batch size", default="16")
parser.add_option('-a', '--accum', action="store", dest="accum", help="model accumulation", default="1")
parser.add_option('-y', '--exptype', action="store", dest="exptype", help="experiment type", default="HEPG2")
parser.add_option('-c', '--dimsize', action="store", dest="dimsize", help="root directory", default="128")

options, args = parser.parse_args()
sys.path.append(options.rootpath)
from logs import get_logger
from utils import dumpobj, loadobj

# Print info about environments
logger = get_logger('Recursion-cgan', 'INFO') # noqa
logger.info('Cuda set up : time {}'.format(datetime.datetime.now().time()))

sys.path.append(options.rootpath + '/repos/cyclegan')
import models

device=torch.device('cuda')
logger.info('Device : {}'.format(torch.cuda.get_device_name(0)))
logger.info('Cuda available : {}'.format(torch.cuda.is_available()))
n_gpu = torch.cuda.device_count()
logger.info('Cuda n_gpus : {}'.format(n_gpu ))

logger.info('Load params : time {}'.format(datetime.datetime.now().time()))
for (k,v) in options.__dict__.items():
    logger.info('{}{}'.format(k.ljust(20), v))

ROOT = options.rootpath
root_train = os.path.join(options.datapath, 'train') + '/'
root_test = os.path.join(options.datapath, 'test') + '/'
image_train = os.path.join(options.rootpath, options.datapath, 'train') +'/'
image_test = os.path.join(options.rootpath, options.datapath, 'test') + '/'
foldsfile = os.path.join(options.rootpath, 'data', options.foldsfile)
normfile = os.path.join(options.rootpath, 'data', options.normfile)
MODEL_PATH = os.path.join(options.rootpath, options.modelpath)
SEED = int(options.seed)
EPOCHS  = int(options.epochs)
FOLD = int(options.fold)
ACCUM = int(options.accum)
TYPE = options.exptype
BATCHSIZE = int(options.batchsize)
IMGTYPE = options.datapath.split('/')[-1]
DIMSIZE = int(options.dimsize)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def dumpobj(file, obj):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

class ImageDataLoaderV1:

    def __init__(self, fns, y=None, batch_size = 16, shuffle = True, aug = None):
        self.fns = np.array(fns)
        if not y is None:
            self.y = np.array(y)
            self.test = False
        else:
            self.test = True
            self.y = np.array(self.fns.shape[0] * [y])
        self.len = self.__len__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug = aug
        self.preprocess_x = lambda x: x
        self.preprocess_y = lambda x: x
        self.cast_x = lambda x: x
        self.cast_y = lambda x: x

    def load_aug_img(self,fp, aug, exp):
        img = loadobj(fp)
        print(50*'-')
        print(img.shape)
        print(exp)
        print(img[:2,:2])
        if aug is not None:
            img = self.aug(image=img)['image']
        img = img /255.
        for i in range(6):
            img[:,:,i] = (img[:,:,i] - norms[exp]['mean'][i]) /  norms[exp]['std'][i]
        return img.astype(np.float32)

    def set_gen(self, batch_size, shuffle=True,aug = None, cast_x = None,cast_y = None, preprocess_x=None, preprocess_y=None):
        # sets self parameters needed for gen
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug = aug
        if cast_x is not None:
            self.cast_x = cast_x
        if cast_y is not None:
            self.cast_y = cast_y
        if preprocess_x is not None:
            self.preprocess_x = preprocess_x
        if preprocess_y is not None:
            self.preprocess_y = preprocess_y

    def __iter__(self):
        if self.shuffle:
            ids = torch.randperm(self.len)
        else:
            ids = torch.LongTensor(range(self.len))
        for batch_ids in ids.split(self.batch_size):
            batch = self.get_batch(batch_ids,self.aug)
            yield batch

    def get_batch(self,batch_ids,aug = None):

        fns_batch = [self.fns[i] for i in batch_ids]
        ys_batch = np.array([self.y[i] for i in batch_ids])

        imgs = []
        for i, fn in enumerate(fns_batch):
            EXPERIMENT = fn.split('/')[-3]
            img = self.load_aug_img(fn, aug, EXPERIMENT)
            #print(fn.split('/')[-3], img.mean(), img.std(), img.shape)
            img = self.preprocess_x(img)
            img = np.transpose(img, (2, 0, 1))
            imgs+=[img]

        x = self.cast_x(np.array(imgs, np.float32))
        y = self.cast_y(ys_batch)

        if self.test:
            return torch.from_numpy(x), torch.Tensor()
        else:
            return torch.from_numpy(x),torch.from_numpy(y)

    def __len__(self):
        return len(self.fns)

    def get_stats(self):
        #calc mean, std, max, min, hist
        pass


class CGAN_Dataloader:

    def __init__(self, dl1, dl2, batch_size = 2):

        #
        self.batch_size = batch_size
        self.gen1 = iter(dl1)
        self.gen2 = iter(dl2)
        self.len = min(len(dl1),len(dl2))

    def set_gen(self, batch_size):
        # sets self parameters needed for gen
        self.batch_size = batch_size

    def __iter__(self):

        for i in range(self.len):
            x1,_ = next(self.gen1)
            x2,_ = next(self.gen2)
            input_dict = {'A': x1,
                          'B': x2,
                          'A_paths': ''}
            yield input_dict


    def __len__(self):
        return self.len


class BaseOptions:

    def __init__(self):
        self.dataroot = ''# required=True,help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.name = 'experiment_name'# help='name of the experiment. It decides where to store samples and models')
        self.gpu_ids = ''#, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.checkpoints_dir='./checkpoints'#, help='models are saved here')
        # model parameters
        self.model='cycle_gan'#help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        self.input_nc=6#3# help='# of input image channels: 3 for RGB and 1 for grayscale')
        self.output_nc=6#3#                            help='# of output image channels: 3 for RGB and 1 for grayscale')
        self.ngf=64#, help='# of gen filters in the last conv layer')
        self.ndf=64#, help='# of discrim filters in the first conv layer')
        self.netD='basic'#                            help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        self.netG='unet_{}'.format(DIMSIZE)# 'unet_256'#                         help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        self.n_layers_D=3#, help='only used if netD==n_layers')
        self.norm='instance'#,                            help='instance normalization or batch normalization [instance | batch | none]')
        self.init_type='normal'#,                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        self.init_gain=0.02#,                            help='scaling factor for normal, xavier and orthogonal.')
        self.no_dropout=True#, help='no dropout for the generator')
        # dataset parameters
        self.dataset_mode='unaligned'#                            help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        self.direction='AtoB'#, help='AtoB or BtoA')
        self.serial_batches= True,#                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.num_threads=4#, type=int, help='# threads for loading data')
        self.batch_size= BATCHSIZE# args['gan_batch_size']#, help='input batch size')
        self.load_size=DIMSIZE#, help='scale images to this size')
        self.crop_size=DIMSIZE#, help='then crop to this size')
        self.max_dataset_size=float("inf")#                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.preprocess='none'#                            help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        self.no_flip=True#                            help='if specified, do not flip the images for data augmentation')
        self.display_winsize=DIMSIZE#                            help='display window size for both visdom and HTML')
        # additional parameters
        self.epoch='latest'#                            help='which epoch to load? set to latest to use latest cached model')
        self.load_iter='0'#      help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        self.verbose=True#, help='if specified, print more debugging information')
        self.suffix=''#, type=str,   help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        self.initialized = True



class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def __init__(self):
        super(TrainOptions,self).__init__()
        # visdom and HTML visualization parameters

        self.display_freq=400
        self.display_ncols=4
        self.display_id=1
        self.display_server="http://localhost"
        self.display_env='main'
        self.display_port=8097
        self.update_html_freq=1000
        self.print_freq=100
        self.no_html=True
        self.save_latest_freq=5000
        self.save_epoch_freq=5
        self.save_by_iter=True
        self.continue_train=True
        self.epoch_count=1
        self.phase='train'
        # training parameters
        self.niter=100
        self.niter_decay=100
        self.beta1=0.5
        self.lr=0.0002
        self.gan_mode='lsgan'
        self.pool_size=50
        self.lr_policy='linear'
        self.lr_decay_iters=50
        self.lambda_identity = 10
        self.isTrain = True
        self.gpu_ids = [0]
        self.lambda_A = 10.0
        self.lambda_B = 10.0
        self.lambda_identity = 0.5

def get_fp(image_folder, row):
    exp = row['experiment']
    plate = row['plate']
    well = row['well']
    side = row['side']
    return image_folder + f'/{exp}/Plate{plate}/{well}_s{side}_w.pk'

def get_fold(row):
    exp = row['experiment']
    fold = -1
    for i in range(5):
        if exp in folds[i]:
            fold = i
    return fold


def get_splits(df):
    splits = []
    for fold in range(5):
        tr_ind = df[df['fold'] != fold].index.values
        val_ind = df[df['fold'] == fold].index.values
        splits += [(tr_ind,val_ind)]
    return splits


logger.info('Load Dataframes : time {}'.format(datetime.datetime.now().time()))
train = pd.read_csv( os.path.join( ROOT, 'data', 'train.csv'))#.iloc[:3000]
train2 = pd.read_csv( os.path.join( ROOT, 'data', 'train.csv'))#.iloc[:3000]
train['side'] = 1
train2['side'] = 2
train =pd.concat([train,train2]).reset_index(drop=True)


logger.info('Load Folds: time {}'.format(datetime.datetime.now().time()))
with open(foldsfile,'rb') as f:
    folds = pickle.load(f)

#image_train='/Users/dhanley2/Documents/Personal/recursion/data/128X128X6/train'
train['fp'] = train.apply(lambda x: get_fp(image_train, x),axis = 1)
train['fold'] = train.apply(lambda x: get_fold(x),axis = 1)
train['cell'] = train.apply(lambda x: x['experiment'].split('-')[0], axis = 1)


splits = get_splits(train)

logger.info('Initialise augmentation: time {}'.format(datetime.datetime.now().time()))

# 'RPE-01': {'mean': array([0.38465203, 0.20891064, 0.1868836 ]),
#  'std': array([0.20245762, 0.13382066, 0.13776862])}
# normfile = '/Users/dhanley2/Documents/Personal/recursion/data/experiment_normalizations256X256X6.p'
with open(normfile,'rb') as f:
    norms = pickle.load(f)
    

normal_aug = A.Compose([A.RandomRotate90(p=1),
                        A.HorizontalFlip(p=0.5),
                        A.IAAAffine(translate_percent=10,rotate=45,shear=10, scale=(0.9,1.1)),
])

    
logger.info('Set train params: time {}'.format(datetime.datetime.now().time()))
opt =TrainOptions()
epochs = EPOCHS


logger.info('Import cgan: time {}'.format(datetime.datetime.now().time()))
from models.cycle_gan_model import CycleGANModel


es = ['01','02','03','04','05','06','07']
me = np.mean([norms[f'HEPG2-{i}']['mean'] for i in es],axis = 0)
dists = [np.mean(np.abs(norms[f'HEPG2-{i}']['mean'] - me)) for i in es]

a = es[np.argmin(dists)]
cell = TYPE # args['type']
A_exp = a
B_exps = [e for e in es if not e == a]

#'02',
'''
normal_aug = A.Compose([A.RandomRotate90(p=1),
                        A.HorizontalFlip(p=0.5),
                        A.IAAAffine(translate_percent=10,rotate=45,shear=10, scale=(0.9,1.1)),
])
BATCHSIZE=16
cell = 'HEPG2'
B_exp='01'
TYPE_B = cell +'-'+  B_exp
ROOT='/Users/dhanley2/Documents/Personal/recursion/'
train = pd.read_csv( os.path.join( ROOT, 'data', 'train.csv'))#.iloc[:3000]
train2 = pd.read_csv( os.path.join( ROOT, 'data', 'train.csv'))#.iloc[:3000]
train['side'] = 1
train2['side'] = 2
train =pd.concat([train,train2]).reset_index(drop=True)
image_train='/Users/dhanley2/Documents/Personal/recursion/data/128X128X6/train'
train['fp'] = train.apply(lambda x: get_fp(image_train, x),axis = 1)


B_df = train[(train['experiment'] == cell +'-'+ B_exp) & (train['plate']==1)]
B_dl = ImageDataLoaderV1(B_df['fp'].values,B_df['sirna'].values.astype(int))
B_dl.set_gen(batch_size=BATCHSIZE,shuffle=True, aug=normal_aug)#,cast_x=lambda x: x[None,:])

for i in B_dl:
    print(50*'-')
    print(i[0].shape)
    for j in range(6):
        print(j, i[0][:,j,:,:].min(), i[0][:,j,:,:].max(), i[0][:,j,:,:].mean(), i[0][:,j,:,:].std())
    print(i[0][0,0,:3,:3])
    break
'''

logger.info('start training: time {}'.format(datetime.datetime.now().time()))

for B_exp in B_exps:
    
    TYPE_A = cell +'-'+  A_exp
    TYPE_B = cell +'-'+  B_exp
    
    logger.info('A_Experiment {} B_Experiment {} : time {}'.format( \
            TYPE_A, TYPE_B, datetime.datetime.now().time()))
    
    A_df = train[train['experiment'] == TYPE_A]
    A_dl = ImageDataLoaderV1(A_df['fp'].values,A_df['sirna'].values.astype(int))
    A_dl.set_gen(batch_size=BATCHSIZE,shuffle=True,aug=augdict[TYPE_A])
    
    B_df = train[train['experiment'] == cell +'-'+ TYPE_B]
    B_dl = ImageDataLoaderV1(B_df['fp'].values,B_df['sirna'].values.astype(int))
    B_dl.set_gen(batch_size=BATCHSIZE,shuffle=True, aug=augdict[TYPE_B])
    
    m = CycleGANModel(opt)

    for e in range(epochs):
        logger.info('Epoch {}: time {}'.format(e, datetime.datetime.now().time()))
        dl = CGAN_Dataloader(A_dl, B_dl)
        dl_iter = iter(dl)

        # did 4 epochs
        m.isTrain = True
        tlosses = 0
        from tqdm import tqdm

        for step in tqdm(range(dl.len//BATCHSIZE-1)):
            data = next(dl_iter)
            m.set_input(data)  # unpack data from dataset and apply preprocessing
            m.optimize_parameters()  # calculate loss functions, get gradients, update network weights
            losses = m.get_current_losses()
            tloss = np.sum([losses[item] for item in losses])
            tlosses += tloss / dl.len
        print(tlosses)

    logger.info('Save Network: time {}'.format(datetime.datetime.now().time()))
    SUB_DIR1 = cell + '/'
    SUB_DIR2 = A_exp + '-' + B_exp + '/'
    if not os.path.exists(MODEL_PATH + SUB_DIR1 + SUB_DIR2):
        os.makedirs(MODEL_PATH + SUB_DIR1 + SUB_DIR2)
    m.save_dir = MODEL_PATH + SUB_DIR1 + SUB_DIR2
    m.save_networks(epochs)

    def apply_gan(x):
        x_fake = m.netG_B(torch.from_numpy(x).cuda())
        x_fake = x_fake.cpu().data.numpy()
        return x_fake
    logger.info('Save GANS as image  : time {}'.format(datetime.datetime.now().time()))
    # save normalizations
    B_dl.set_gen(batch_size=1,shuffle=False,cast_x=apply_gan )
    new_fns = [image_train + '/cgan/' + '/'.join(fp.split('/')[-3:]) for fp in B_dl.fns]
    #make folders
    new_folders = list(set(['/'.join(fp.split('/')[:-1]) for fp in new_fns]))
    for f in new_folders:
        if not os.path.exists(f):
            os.makedirs(f)

    i = 0
    for new_img, _ in tqdm(B_dl):
        new_img = np.transpose(new_img[0].cpu().data.numpy(), (1, 2, 0))
        #new_img = np.clip(new_img, 0,1)
        #new_img = np.round(255*new_img).astype(np.uint8)
        if i%10000000:
            logger.info('Out shape : {}'.format(new_img.shape))
            logger.info('Out location : {}'.format(new_fns[i]))
        dumpobj(new_fns[i],new_img)
        i += 1
    break
