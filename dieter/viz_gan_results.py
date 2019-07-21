import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image

def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)
PATH='/Users/dhanley2/Documents/Personal/recursion/data/tmp'
LOCALPATH='/Users/dhanley2/Documents/Personal/recursion/data/train/HEPG2-01/Plate1'

# 256X256X6/train/cgan/HEPG2-01/Plate1/O*.pk
trndf = pd.read_csv(os.path.join(PATH,'../train.csv'))
trndf[(trndf['experiment']=='HEPG2-01' ) & (trndf['plate']==1) & (trndf['well'] == 'O02')]


imgpng = [Image.open(os.path.join(LOCALPATH, 'O02_s1_w{}.png'.format(i))) for i in range(1,7)]
imgpng = [np.array(i) for i in imgpng]
imgpng[0].shape

dir(imgpng[0].resize((256,256)))

imgorig = loadobj(os.path.join(PATH, 'O02_s1_w.pk'))
imggan = loadobj(os.path.join(PATH, 'gan/O02_s1_w.pk'))
imgtrgt = loadobj(os.path.join(PATH, 'target/O02_s1_w.pk'))

imgorig[:,:,0]==imgpng[0]
imgorig[:,:,0].shape

Image.fromarray(imgorig[:,:,:3])#.show()
Image.fromarray(imggan[:,:,:3])
Image.fromarray(imgtrgt[:,:,:3])

Image.fromarray(imgorig[:,:,4:])
Image.fromarray(imggan[:,:,4:].astype(np.uint8))


imggan[:,:,:3].shape