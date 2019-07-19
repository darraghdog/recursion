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
parser.add_option('-o', '--datapath', action="store", dest="datapath", help="root directory", default="128X128X6")
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
data_train = os.path.join(options.rootpath, 'data', options.datapath, 'train') +'/'
data_test = os.path.join(options.rootpath, 'data', options.datapath, 'test') + '/'
DIMSIZE = int(options.dimsize)
        
def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

os.listdir('/share/dhanley2/recursion/data/128X128X6/train/')

logger.info('Traverse directory {} : time {}'.format(data_train, datetime.datetime.now().time()))

# Directories traverse 
normdict = {}
for data_dir in [data_test, data_train]:
    for t, (dir_, subdir_, files_) in enumerate(os.walk(data_dir)):
        if subdir_ != []:
            continue
        logger.info('Process directories : {} {}'.format(dir_, subdir_))
        experiment = '/'.join(dir_.split('/')[-2:])
        normdict[experiment] = {'mean' : [], 'std' : []}
        for file_ in files_:
            if '.pk' in file_:
                fname = os.path.join(dir_, file_)
                try : 
                    im = loadobj(fname)
                    normdict[experiment]['mean'].append(im.mean((0,1)))
                    normdict[experiment]['std'].append(im.std((0,1)))
                except:
                    logger.info('Error loading : {}'.format(fname))
        normdict[experiment]['mean'] = sum(normdict[experiment]['mean'])/len(normdict[experiment]['mean'])   
        normdict[experiment]['std'] = sum(normdict[experiment]['std'])/len(normdict[experiment]['std']) 
        normdict[experiment]['mean'] /= 255
        normdict[experiment]['std'] /= 255
        logger.info('Experiment Stats : {}'.format(normdict[experiment]))
logger.info('Results : {}'.format(normdict))

logger.info('Output files : time {}'.format(datetime.datetime.now().time()))

outfile = os.path.join(options.rootpath, 'data', 'experiment_normalizations{}.p'.format(options.datapath))
dumpobj(outfile, normdict)
            
#normfile = '/Users/dhanley2/Documents/Personal/recursion/data/experiment_normalizations_128x128x3.p'
#with open(normfile,'rb') as f:
#    norms = pickle.load(f) 
