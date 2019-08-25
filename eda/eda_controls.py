import pandas as pd
import numpy as np
import os
PATH = '/Users/dhanley2/Documents/Personal/recursion/data'

trndf = pd.read_csv(os.path.join(PATH, 'train.csv'))
tstdf = pd.read_csv(os.path.join(PATH, 'test.csv'))
trnctrldf = pd.read_csv(os.path.join(PATH, 'train_controls.csv'))
tstctrldf = pd.read_csv(os.path.join(PATH, 'test_controls.csv'))
statsdf = pd.read_csv(os.path.join(PATH, 'stats.csv'))


trnctrldf[trnctrldf['experiment']=='HEPG2-01'].head(10)
trnctrldf[trnctrldf['experiment']=='HEPG2-02'].head(10)
trndf[trndf['experiment']=='HEPG2-01'].head(10)
trndf[trndf['experiment']=='HEPG2-02'].head(10)


trnctrldf['sirna'].value_counts()
trndf['sirna'].value_counts().sort_index()




pd.concat([trndf.experiment, tstdf.experiment]).apply(lambda x: x.split('-')[0]).value_counts()

CLASSES = ['HEPG2', 'HUVEC', 'RPE', 'U2OS']

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing  import LabelEncoder

ohc = OneHotEncoder(sparse=False)
lenc = LabelEncoder()

expsall = pd.concat([trndf.experiment, tstdf.experiment]).apply(lambda x: x.split('-')[0])
expsall = lenc.fit_transform(expsall)

ohc.fit(np.expand_dims(expsall, 1))

expsbatch = trndf.iloc[[i*2000 for i in range(16)]].experiment.apply(lambda x: x.split('-')[0]).values
expsbatch = lenc.transform(expsbatch)
expsbatch = ohc.transform(np.expand_dims(expsbatch, 1))
expsbatch


expsbatch = lenc.transform(['HUVEC'])
expsbatch = ohc.transform(np.expand_dims(expsbatch, 1))
expsbatch


trndf[trndf['well']=='B03'].tail(10)
trnctrldf[trnctrldf['well']=='B02'].tail(10)

trndf[trndf['well']=='B03'].shape
trnctrldf[trnctrldf['well']=='B02'].shape

statsdf['experiment'] = statsdf['FileName'].apply(lambda x: x.split('/')[-3])
statsdf['plate'] = statsdf['FileName'].apply(lambda x: x.split('/')[-2])
statsdf['fname'] = statsdf['FileName'].apply(lambda x: x.split('/')[-1])
statsgrpdf = statsdf.groupby(['experiment', 'plate', 'Channel'])['Mean', 'Std'].mean()

#statsgrpdf = statsdf.groupby(['experiment', 'plate', 'Channel'])['Mean', 'Std'].mean().unstack()
statsgrpdf.to_csv(os.path.join(PATH, 'stats_grp.csv'))


statsgrpdf.loc[('HEPG2-01', 'Plate1', 1)]['Mean']

trnctrldf[(trnctrldf['experiment']== 'RPE-05') & (trnctrldf['plate']== 3)]
trndf[(trndf['experiment']== 'RPE-05') & (trndf['plate']== 3)].head(60)

