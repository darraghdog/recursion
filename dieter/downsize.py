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


# DEFAULT_BASE_PATH = 'gs://rxrx1-us-central1'
# DEFAULT_METADATA_BASE_PATH = os.path.join(DEFAULT_BASE_PATH, 'metadata')
# DEFAULT_IMAGES_BASE_PATH = os.path.join(DEFAULT_BASE_PATH, 'images')
DEFAULT_CHANNELS = (1, 2, 3, 4, 5, 6)
RGB_MAP = {
    1: {
        'rgb': np.array([19, 0, 249]),
        'range': [0, 51]
    },
    2: {
        'rgb': np.array([42, 255, 31]),
        'range': [0, 107]
    },
    3: {
        'rgb': np.array([255, 0, 25]),
        'range': [0, 64]
    },
    4: {
        'rgb': np.array([45, 255, 252]),
        'range': [0, 191]
    },
    5: {
        'rgb': np.array([250, 0, 253]),
        'range': [0, 89]
    },
    6: {
        'rgb': np.array([254, 255, 40]),
        'range': [0, 191]
    }
}

def convert_tensor_to_rgb(t, channels=DEFAULT_CHANNELS, vmax=255, rgb_map=RGB_MAP):
    """
    Converts and returns the image data as RGB image

    Parameters
    ----------
    t : np.ndarray
        original image data
    channels : list of int
        channels to include
    vmax : int
        the max value used for scaling
    rgb_map : dict
        the color mapping for each channel
        See rxrx.io.RGB_MAP to see what the defaults are.

    Returns
    -------
    np.ndarray the image data of the site as RGB channels
    """
    colored_channels = []
    for i, channel in enumerate(channels):
        x = (t[:, :, i] / vmax) / \
            ((rgb_map[channel]['range'][1] - rgb_map[channel]['range'][0]) / 255) + \
            rgb_map[channel]['range'][0] / 255
        x = np.where(x > 1., 1., x)
        x_rgb = np.array(
            np.outer(x, rgb_map[channel]['rgb']).reshape(512, 512, 3),
            dtype=int)
        colored_channels.append(x_rgb)
    im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    im = np.where(im > 255, 255, im)
    return im


def load_image(f):
    return cv2.imread(f,cv2.IMREAD_UNCHANGED)

def load_images_as_tensor(image_paths, dtype=np.float):
    n_channels = len(image_paths)

    data = np.ndarray(shape=(512, 512, n_channels), dtype=dtype)

    for ix, img_path in enumerate(image_paths):
        data[:, :, ix] = load_image(img_path)

    return data

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

fn = fns[0]
#target = 'input/preprocessed/128x128x3/train/'
if not os.path.exists(target):
    os.makedirs(target)

# create folders
def create_target_folder(fn):
    new_fn = target + '/'.join(fn.split('/')[2:]) + '.png'
    new_dir = '/'.join(new_fn.split('/')[:-1]) + '/'
    if not os.path.exists(new_dir):
       os.makedirs(new_dir)

for fn in tqdm(fns):
    create_target_folder(fn)

def convert(fn):
    new_fn = target + '/'.join(fn.split('/')[2:]) + '.png'
    new_dir = '/'.join(new_fn.split('/')[:-1]) + '/'
    if not os.path.exists(new_fn):
        img_paths = [fn + f'{i}.png' for i in range(1,7)]
        t = load_images_as_tensor(img_paths)
        im = convert_tensor_to_rgb(t)
        im = im.astype(np.float32)
        a = cv2.resize(im, dsize=(128, 128))
        b = np.round(a).astype(np.uint8)

        #if not os.path.exists(new_dir):
        #   os.makedirs(new_dir)
        cv2.imwrite(new_fn,b)

num_cores = 8
from joblib import Parallel, delayed
Parallel(n_jobs=num_cores, prefer="threads")(delayed(convert)(i) for i in tqdm(fns))

root = root_test
target = target_test
if not os.path.exists(target):
    os.makedirs(target)

fns = []
genes = os.listdir(root)
for g in tqdm(genes):
    plate = os.listdir(root + g + '/')
    for p in plate:
        files = os.listdir(root + g + '/' + p + '/')
        files2 = list(set([root + g + '/' + p + '/' + f[:-5] for f in files]))
        for f in files2:
            fns += [f]

for fn in tqdm(fns):
    create_target_folder(fn)

Parallel(n_jobs=num_cores, prefer="threads")(delayed(convert)(i) for i in tqdm(fns))


