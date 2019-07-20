#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 17:36:07 2019

@author: dhanley2
"""



from PIL import Image

def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)
PATH='/Users/dhanley2/Documents/Personal/recursion/data/tmp'

# 256X256X6/train/cgan/HEPG2-01/Plate1/O*.pk
#trndf = pd.read_csv(os.path.join(PATH,'../train.csv'))
#trndf[(trndf['experiment']=='HEPG2-01' ) & (trndf['plate']==1)]# & trndf['well'] == 'O02']

imgorig = loadobj(os.path.join(PATH, 'O02_s1_w.pk'))
imggan = loadobj(os.path.join(PATH, 'gan/O02_s1_w.pk'))
imgtrgt = loadobj(os.path.join(PATH, 'target/O02_s1_w.pk'))

Image.fromarray(imgorig[:,:,:3])#.show()
Image.fromarray(imggan[:,:,:3])
Image.fromarray(imgtrgt[:,:,:3])

Image.fromarray(imgorig[:,:,4:])
Image.fromarray(imggan[:,:,4:].astype(np.uint8))


imggan[:,:,:3].shape