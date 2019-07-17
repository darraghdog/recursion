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
!pip install albumentations
import albumentations as A
import torch
import random
import pickle

# Print info about environments
parser = optparse.OptionParser()
parser.add_option('-r', '--rootpath', action="store", dest="rootpath", help="root directory", default="/share/dhanley2/recursion")
parser.add_option('-d', '--datapath', action="store", dest="datapath", help="root directory", default="/share/dhanley2/recursion/data")
parser.add_option('-o', '--targetpath', action="store", dest="targetpath", help="root directory", default="/share/dhanley2/recursion/data/128X128X3")



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

root_train = os.path.join(options.datapath, 'train') + '/'
root_test = os.path.join(options.datapath, 'test') + '/'
target_train = os.path.join(options.targetpath, 'train') +'/'
target_test = os.path.join(options.targetpath, 'test') + '/'


print(root_train)
print(root_test)
print(target_train)
print(target_test)



