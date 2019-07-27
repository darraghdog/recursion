# https://stackoverflow.com/a/28520445
import pandas as pd
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import optparse
import os, sys
from PIL import Image,ImageStat
import math
import cv2
import skimage.transform
import skimage.filters
import skimage.morphology
import scipy.stats
import pickle as pickle
import sys
#PATH = '/Users/dhanley2/Documents/Personal/recursion/data'
#sys.path.append(PATH+'/../../rxrx1-utils')
#import rxrx.io as rio

# Print info about environments
parser = optparse.OptionParser()
parser.add_option('-r', '--rootpath', action="store", dest="rootpath", help="root directory", default="/share/dhanley2/recursion/")
parser.add_option('-i', '--imgpath', action="store", dest="imgpath", help="root directory", default="data/mount/512X512X6/")

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

ROOT = options.rootpath
path_data = os.path.join(ROOT, 'data')
path_img = os.path.join(ROOT, options.imgpath)
device = 'cuda'
print('Data path : {}'.format(path_data))
print('Image path : {}'.format(path_img))

def illum_stats_filename(output_dir, plate_name):
    return "{}/{}/{}.pkl".format(output_dir, plate_name, plate_name)


def percentile(prob, p):
    cum = np.cumsum(prob)
    pos = cum > p
    return np.argmax(pos)

#################################################
## ILLUMINATION CORRECTION FUNCTION
#################################################

def dumpobj(file, obj):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

#################################################
## COMPUTATION OF ILLUMINATION STATISTICS
#################################################

# Build pixel histogram for each channel
class IlluminationStatistics():
    def __init__(self, bits, channels, down_scale_factor, median_filter_size, name=""):
        self.depth = 2 ** bits
        self.channels = channels
        self.name = name
        self.down_scale_factor = down_scale_factor
        self.median_filter_size = median_filter_size
        self.hist = np.zeros((len(channels), self.depth), dtype=np.float64)
        self.count = 0
        self.expected = 1
        self.mean_image = None
        self.original_image_size = None

    # Accumulate the mean image. Useful for illumination correction purposes
    def addToMean(self, img):
        # Rescale original image to half
        if self.mean_image is None:
            self.mean_image = np.zeros_like(img, dtype=np.float64)
        # Add image to current mean values
        self.mean_image += img
        return

    # Compute global statistics on pixels. 
    def computeStats(self):
        # Initialize counters
        bins = np.linspace(0, self.depth - 1, self.depth)
        mean = np.zeros((len(self.channels)))
        lower = np.zeros((len(self.channels)))
        upper = np.zeros((len(self.channels)))
        self.mean_image /= self.count

        # Compute percentiles and histogram
        for i in range(len(self.channels)):
            probs = self.hist[i] / self.hist[i].sum()
            mean[i] = (bins * probs).sum()
            lower[i] = percentile(probs, 0.0001)
            upper[i] = percentile(probs, 0.9999)
        stats = {"mean_values": mean, "upper_percentiles": upper, "lower_percentiles": lower, "histogram": self.hist,
                 "mean_image": self.mean_image, "channels": self.channels, "original_size": self.original_image_size}
        # Compute illumination correction function and add it to the dictionary
        correct = IlluminationCorrection(stats, self.channels, self.original_image_size)
        correct.compute_all(self.median_filter_size)
        stats["illum_correction_function"] = correct.illum_corr_func

        # Plate ready
        print("Plate " + self.name + " done")
        return stats

    def processImage(self, index, img):
        if self.original_image_size is None:
            self.original_image_size = img.shape
        self.addToMean(img)
        self.count += 1
        for i in range(len(self.channels)):
            counts = np.histogram(img[:, :, i], bins=self.depth, range=(0, self.depth))[0]
            self.hist[i] += counts.astype(np.float64)
            
            
class IlluminationCorrection(object):
    def __init__(self, stats, channels, target_dim):
        self.stats = stats
        self.channels = channels
        self.target_dim = (target_dim[0], target_dim[1])
        self.illum_corr_func = np.zeros((self.target_dim[0], self.target_dim[1], len(self.channels)))

    # Based on Sing et al. 2014 paper
    def channel_function(self, mean_channel, disk_size):
        #TODO: get np.type from other source or parameterize or compute :/
        # We currently assume 16 bit images
        operator = skimage.morphology.disk(disk_size)
        filtered_channel = skimage.filters.median(mean_channel.astype(np.uint16), operator)
        robust_minimum = scipy.stats.scoreatpercentile(filtered_channel, 2)
        filtered_channel = np.maximum(filtered_channel, robust_minimum)
        illum_corr_func = filtered_channel / robust_minimum
        return illum_corr_func

    def compute_all(self, median_filter_size):
        disk_size = median_filter_size / 2  # From diameter to radius
        for ch in range(len(self.channels)):
            self.illum_corr_func[:, :, ch] = self.channel_function(self.stats["mean_image"][:, :, ch], disk_size)

    def apply(self, image):
        return image / self.illum_corr_func

logger.info('Load Path names')
cols = ['experiment', 'plate']
trncdf = pd.read_csv( os.path.join(path_data, 'train_controls.csv'), usecols = cols)
tstcdf = pd.read_csv( os.path.join(path_data, 'test_controls.csv'), usecols = cols)
trncdf['type'] = 'train'
tstcdf['type'] = 'test'
alldf = pd.concat([trncdf, tstcdf]).drop_duplicates().reset_index(drop=True)

#alldf = alldf[alldf['experiment'] == 'HEPG2-01']

statdict = {}
for t, row in alldf.iterrows():
    img_dir = '{}/{}/{}/Plate{}'.format(options.imgpath, row[-1], *row[:2])
    img_names = [i for i in os.listdir(os.path.join(path_data, img_dir))]
    illuminstat = IlluminationStatistics(bits = 8, 
                           channels=[0,1,2,3,4,5], 
                           down_scale_factor=1, 
                           median_filter_size =24, 
                           name='\t'+img_dir)
    for t, imf in enumerate(tqdm(img_names)):  
        fname = os.path.join(PATH, img_dir, imf)
        img = loadobj(fname)
        illuminstat.processImage(0, img)  
    stats = illuminstat.computeStats()
    statdict[img_dir] = stats
    
outfile = path_data+'/illumsttats_{}.pk'.format(options.imgpath)
dumpobj(outfile , statdict)

'''
img= loadobj(imnames[100])
x1 = rio.convert_tensor_to_rgb(img, dim = 256)
x2 = rio.convert_tensor_to_rgb(np.uint8(img/stats['illum_correction_function']), dim = 256)

Image.fromarray(np.uint8(x1))
Image.fromarray(np.uint8(x2))
'''


