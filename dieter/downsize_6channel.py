import datetime
import sys
import os
import pickle
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


# Print info about environments
parser = optparse.OptionParser()
parser.add_option('-r', '--rootpath', action="store", dest="rootpath", help="root directory", default="/share/dhanley2/recursion")
parser.add_option('-d', '--datapath', action="store", dest="datapath", help="root directory", default="data")
parser.add_option('-o', '--targetpath', action="store", dest="targetpath", help="root directory", default="128X128X6")
parser.add_option('-a', '--dimsize', action="store", dest="dimsize", help="root directory", default="128")

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

root_train = os.path.join(options.rootpath, options.datapath, 'train') + '/'
root_test = os.path.join(options.rootpath, options.datapath, 'test') + '/'
target_train = os.path.join(options.rootpath, options.datapath, options.targetpath, 'train') +'/'
target_test = os.path.join(options.rootpath, options.datapath, options.targetpath, 'test') + '/'
DIMSIZE = int(options.dimsize)

def load_image(f):
    return cv2.imread(f,cv2.IMREAD_UNCHANGED)

def load_images_as_tensor(image_paths, dim, dtype=np.float32):
    
    data = np.ndarray(shape=(dim, dim, 6), dtype=dtype)

    for ix, img_path in enumerate(image_paths):
        im = load_image(img_path)
        im = im.astype(np.float32)
        data[:, :, ix] = cv2.resize(im, dsize=(dim, dim))
        del im

    return data

def dumpobj(file, obj):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

def convert(fn):
    fname = fn.split('/')[-1] + '.pk'
    fdir = fn.replace(fn.split('/')[-1], '')
    fdir = fdir.replace('data', 'data/{}'.format(options.targetpath))
    img_paths = [fn + f'{i}.png' for i in range(1,7)]
    t = load_images_as_tensor(img_paths, dim = DIMSIZE)
    t = np.round(t).astype(np.uint8)
    try:
        os.makedirs(fdir)
    except:
        1
    dumpobj(fdir+fname, t)

root = root_train
target = target_train
fns = []
genes = os.listdir(root)
for g in tqdm(genes):
    plate = os.listdir(root + g + '/')
    for p in plate:
        files = os.listdir(root + g + '/' + p + '/')
        files2 = list(set([root + g + '/' + p + '/' + f[:-5] for f in files]))
        for f in files2:
            fns += [f]

num_cores = 16
from joblib import Parallel, delayed

Parallel(n_jobs=num_cores, prefer="threads")(delayed(convert)(i) for i in tqdm(fns))

root = root_test
target = target_test
fns = []
genes = os.listdir(root)
for g in tqdm(genes):
    plate = os.listdir(root + g + '/')
    for p in plate:
        files = os.listdir(root + g + '/' + p + '/')
        files2 = list(set([root + g + '/' + p + '/' + f[:-5] for f in files]))
        for f in files2:
            fns += [f]

Parallel(n_jobs=num_cores, prefer="threads")(delayed(convert)(i) for i in tqdm(fns))

# ['/share/dhanley2/recursion/data/test/HEPG2-08/Plate1/B13_s1_w', '/share/dhanley2/recursion/data/test/HEPG2-08/Plate1/K22_s1_w', '/share/dhanley2/recursion/data/test/HEPG2-08/Plate1/B20_s1_w', '/share/dhanley2/recursion/data/test/HEPG2-08/Plate1/C09_s1_w', '/share/dhanley2/recursion/data/test/HEPG2-08/Plate1/M20_s2_w']
