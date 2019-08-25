#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 19:13:07 2019

@author: dhanley2
"""


import os
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

PATH = '/Users/dhanley2/Documents/Personal/recursion/data'

trnctrldf = pd.read_csv(os.path.join(PATH, 'train_controls.csv'))
trndf = pd.read_csv(os.path.join(PATH, 'train.csv'))
tstctrldf = pd.read_csv(os.path.join(PATH, 'test_controls.csv'))
tstdf = pd.read_csv(os.path.join(PATH, 'test.csv'))
statsdf = pd.read_csv(os.path.join(PATH, 'stats.csv'))

statsdf['experiment'] = statsdf['FileName'].apply(lambda x: x.split('/')[-3])
statsdf['plate'] = statsdf['FileName'].apply(lambda x: x.split('/')[-2])
statsdf['fname'] = statsdf['FileName'].apply(lambda x: x.split('/')[-1])

statsgrpdf = statsdf.groupby(['experiment', 'plate', 'Channel'])['Mean', 'Std'].mean().unstack()
statsgrpdf.to_csv(os.path.join(PATH, 'stats_grp.csv'))

trnctrldf.head()
trndf.head()

for df in [trndf, trnctrldf]:
    print(50*'-')
    print(df[(df['experiment']=='HEPG2-01') & \
             (df['plate']==1) & \
             (df['id_code']=='HEPG2-01_1_B02')])

trndf[(trndf['experiment']=='HEPG2-01') & (trndf['plate']==1)]
trnctrldf[(trnctrldf['experiment']=='HEPG2-01') & (trnctrldf['plate']==1)]

trnctrldf['experiment'].value_counts()

trnctrldf[trnctrldf['experiment'] \
          .str.contains('HEPG2')] \
          .groupby(['experiment', 'well_type', 'plate']).count()

trndf[trndf['experiment'] \
          .str.contains('HEPG2')] \
          .groupby(['experiment', 'plate']).count()


tstctrldf[tstctrldf['experiment'] \
          .str.contains('HEPG2')] \
          .groupby(['experiment', 'well_type', 'plate']).count()

tstdf[tstdf['experiment'] \
          .str.contains('HEPG2')] \
          .groupby(['experiment', 'plate']).count()