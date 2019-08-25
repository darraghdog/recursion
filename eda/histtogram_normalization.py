# https://stackoverflow.com/a/28520445
import pandas as pd
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
from PIL import Image
from PIL import ImageStat
import math

def image_histogram(image, number_bins=256):
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    return cdf, bins

def histogram_equalization(image,  bins, cdf):
    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)

PATH = '/Users/dhanley2/Documents/Personal/recursion/data'
img_dir = os.path.join(PATH, 'train/HEPG2-01/Plate1')
img_names = [i for i in os.listdir(img_dir) if '1.png' in i]

imf = os.path.join(PATH, img_dir, img_names[0])

import cv2

def img2np(imf):
    return np.array(Image.open(os.path.join(PATH, img_dir, imf)))

meanimg = np.mean([img2np(i) for i in img_names], 0)
Image. fromarray(np.uint8((meanimg)*100))


meanimg
pd.Series(10*((meanimg).flatten())).value_counts()

import random
rand = random.randint(0, 300)
for t, imf in enumerate(img_names) :
    if t==rand:
        break
    img = Image.open(os.path.join(PATH, img_dir, imf))
    cl = clahe.apply(np.array(img))
    print((cl>245.).sum(), '\t\t',(cl<10).sum(), '\t\t', (np.array(img)>245.).sum(),  '\t\t',(np.array(img)<10).sum())
    Image. fromarray(cl)

#-----Reading the image-----------------------------------------------------






Image. fromarray(cl)
(cl==255.).sum()
(cl==0).sum()

cl = clahe.apply(img)

#-----Converting image to LAB Color model----------------------------------- 
lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
Image. fromarray(lab)

#-----Splitting the LAB image to different channels-------------------------
l, a, b = cv2.split(lab)
cv2.imshow('l_channel', l)
cv2.imshow('a_channel', a)
cv2.imshow('b_channel', b)

#-----Applying CLAHE to L-channel-------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
cv2.imshow('CLAHE output', cl)

#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg = cv2.merge((cl,a,b))
cv2.imshow('limg', limg)

#-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imshow('final', final)