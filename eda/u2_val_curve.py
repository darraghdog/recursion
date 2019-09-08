#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 06:59:25 2019

@author: dhanley2
"""
import os
import pandas as pd
import numpy as np

path_data = '/Users/dhanley2/Documents/Personal/recursion/sub/tts/'
traindf = pd.read_csv( os.path.join( path_data, 'train.csv'))#.iloc[:3000]
folds = pd.read_csv( os.path.join( path_data, 'folds.csv'))
traindf = traindf.merge(folds, on = 'experiment')
traindf[['experiment', 'fold']].drop_duplicates()

u2df = pd.read_csv('/Users/dhanley2/Documents/Personal/u2os.csv')
u2df = u2df.astype(np.float32)

u2df.columns = ['{} {}'.format(b, 'OOF acc') for (a,b) in zip(u2df.columns, ['U2OS-01', 'U2OS-02', 'U2OS-03'])]

u2df.plot.line()


u2df.index=list(range(90, u2df.shape[0]+90))
u2df.rolling(1,axis =0).mean().plot.line(title='Logits Argmax Accuracy')