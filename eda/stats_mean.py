import pandas as pd
import pickle
import numpy as np

import os
PATH = '/Users/dhanley2/Documents/Personal/recursion/data'
path_data = PATH
traindf = pd.read_csv( os.path.join( path_data, 'train.csv'))#.iloc[:3000]
testdf  = pd.read_csv( os.path.join( path_data, 'test.csv'))
train_ctrl = pd.read_csv( os.path.join( path_data, 'train_controls.csv'))#.iloc[:3000]
test_ctrl = pd.read_csv( os.path.join( path_data, 'test_controls.csv'))#.iloc[:3000]


trnix = train_ctrl['well_type']=='negative_control'
tstix = test_ctrl['well_type']=='negative_control'
negdf = pd.concat([train_ctrl[trnix].reset_index(drop=True),
                   test_ctrl[tstix].reset_index(drop=True)])
train_ctrl = train_ctrl[~trnix].reset_index(drop=True)
test_ctrl = test_ctrl[~tstix].reset_index(drop=True)
negdf = negdf.set_index(['experiment', 'plate'])
a  = negdf.loc['HUVEC-24', 4].sample(n=1).iloc[0]['well']


testcdf['well_type'].value_counts()
traincdf['well'].value_counts().sort_index()

logger.info('Load Path names')
cols = ['experiment', 'plate']
trncdf = pd.read_csv( os.path.join(path_data, 'train_controls.csv'), usecols = cols)
tstcdf = pd.read_csv( os.path.join(path_data, 'test_controls.csv'), usecols = cols)
trncdf['type'] = 'train'
tstcdf['type'] = 'test'
alldf = pd.concat([trncdf, tstcdf]).drop_duplicates().reset_index(drop=True)

for t,row in alldf.iterrows():
    row
'/'.join(map(str, row.tolist()))

cols = ['experiment', 'plate', 'well']
alltrndf = pd.concat([traindf[cols], traincdf[cols]], 0 )
alltrndf.groupby(['experiment', 'plate'])['well'].nunique()

a = pd.Series(list(range(10))+[7,7,8])
def dupe(a):
    return str(a.value_counts()[a.value_counts()>1].to_dict())



np.uint8(np.random.rand(16,16)*2)


traindf = pd.read_csv( os.path.join( path_data, 'train.csv'))#.iloc[:3000]
traindf['exptype'] = traindf['experiment'].apply(lambda x: x.split('-')[0])
traindf['wellcol'] = traindf['well'].apply(lambda x: ord(x[0])-ord('A'))
traindf['wellrow'] = traindf['well'].apply(lambda x: int(x[1:]))
traindf['wellrow'] = traindf['wellrow']
traindf['wellcol'] = traindf['wellcol']
nuniqdf = traindf.groupby(['exptype', 'wellcol', 'wellrow'])['sirna'].nunique().unstack()
countdf = traindf.groupby(['exptype', 'wellcol', 'wellrow'])['sirna'].count().unstack()
print((nuniqdf.fillna(-1)!=countdf.fillna(-1)).values.sum())

a = traindf.groupby(['wellcol', 'wellrow'])['sirna'].apply(lambda x: dupe(x)).unstack()#.transpose()

traindf.groupby(['wellcol', 'wellrow'])['sirna']

pd.Series(nuniqdf.values.flatten()).hist()

nuniqdf.iloc[np.where(nuniqdf.values==114)]


def sirnapos(df, s):
    return df[df['experiment'].str.contains('HEPG2')].groupby(['experiment', 'plate', 'wellcol', 'wellrow'])['sirna'].apply(lambda x: (x==s).sum()).unstack()

sirnapos(traindf[traindf['experiment']=='HEPG2-01'], 100)
a.to_csv(os.path.join(PATH, '../../sirna_posn.csv'))


nuniqdf = traindf.groupby(['experiment', 'wellcol', 'wellrow'])['sirna'].nunique().unstack()
countdf = traindf.groupby(['experiment', 'wellcol', 'wellrow'])['sirna'].count().unstack()
print((nuniqdf.fillna(-1)!=countdf.fillna(-1)).values.sum())


subdf = pd.read_csv( os.path.join( '/Users/dhanley2/Downloads/blend_submission_2.csv'))#.iloc[:3000]
subdf['experiment'] = subdf['id_code'].apply(lambda x: x.split('_')[0])
subdf['exptype'] = subdf['experiment'].apply(lambda x: x.split('-')[0])
subdf['plate'] = subdf['id_code'].apply(lambda x: x.split('_')[1])
subdf['well'] = subdf['id_code'].apply(lambda x: x.split('_')[-1])
subdf['wellcol'] = subdf['well'].apply(lambda x: ord(x[0])-ord('A'))
subdf['wellrow'] = subdf['well'].apply(lambda x: int(x[1:]))
subdf['wellrow'] = subdf['wellrow']
subdf['wellcol'] = subdf['wellcol']
nuniqdf1 = subdf.groupby(['wellcol', 'wellrow'])['sirna'].nunique().unstack()
countdf1 = subdf.groupby(['wellcol', 'wellrow'])['sirna'].count().unstack()
pd.Series(nuniqdf1.values.flatten()).hist()

nuniqdf1.iloc[np.where(nuniqdf.values==114)]


nuniqdf1.shape()

print((nuniqdf1.fillna(-1)!=countdf1.fillna(-1)).values.sum())

ix = np.where((nuniqdf1.fillna(-1)!=countdf1.fillna(-1)))
nuniqdf1.iloc[ix]
countdf1.iloc[ix]

countdf1.iloc[ix]!=nuniqdf1.iloc[ix]


nuniqdf1.fillna(-1)[(nuniqdf1.fillna(-1)!=countdf1.fillna(-1)).values]

nuniqdf1.loc['HUVEC-17']

np.where((nuniqdf1.fillna(-1)!=countdf1.fillna(-1)).values)

subdf.sort_values(['experiment', 'well', 'plate'])

nuniqdf.transpose()
countdf.transpose()


traindf[(traindf.experiment=='HEPG2-01')&(traindf.well.str.contains('B0'))]

traindf[['experiment', 'sirna', 'well']]

traindf.shape
add_sites(traindf).shape
def add_sites(df):
    df1 = df.copy()
    df2 = df.copy()
    df1['site'] = 1
    df2['site'] = 2
    return pd.concat([df1, df2], 0)

statsdf = pd.read_csv(os.path.join(PATH, 'stats.csv'))
statsdf = pd.read_csv(os.path.join(PATH, 'stats_256.csv'))
statsdf['experiment'] = statsdf['FileName'].apply(lambda x: x.split('/')[-3])
statsdf['plate'] = statsdf['FileName'].apply(lambda x: x.split('/')[-2])
statsdf['fname'] = statsdf['FileName'].apply(lambda x: x.split('/')[-1])
statsdf['channel'] = statsdf['FileName'].apply(lambda x: int(x[-5]))

statsdf['well'] = statsdf['fname'].apply(lambda x : x.split('_')[0])
statsdf['site'] = statsdf['fname'].apply(lambda x : int(x.split('_')[1][1]))
statsdf['wellrow'] = statsdf['well'].apply(lambda x : x[0])
statsdf['wellcol'] = statsdf['well'].apply(lambda x : int(x[1:]))


statsgrpdf1 = statsdf.groupby(['experiment', 'plate', 'Channel'])['Mean', 'Std'].mean()
statsgrpdf1 = statsdf.groupby(['experiment', 'Channel'])['Min', 'Max'].mean()
statsgrpdf1.hist()
statsgrpdf1.plot.line(figsize=(20,5))
statsgrpdf1.unstack().to_csv(os.path.join(PATH, '../../minmaxchannel.csv'))

statsgrpdf2 = statsdf.groupby(['experiment', 'Channel'])['Mean', 'Std'].mean()
statsgrpdf3 = statsdf.groupby(['site'])['Mean', 'Std'].mean()
statsgrpdf3 = statsdf.groupby(['channel', 'wellrow', 'wellcol'])['Mean', 'Std'].mean().unstack()

statsgrpdf2.unstack().to_csv(os.path.join(PATH, '../../expchannel.csv'))
statsgrpdf3.transpose().to_csv(os.path.join(PATH, '../../heatchannel.csv'))

statsgrpdf1
statsgrpdf2
statsgrpdf1.loc[['HEPG2-02']]
statsgrpdf2.loc[['HEPG2-02']]


statsgrpdf2.loc[['HEPG2-{}'.format(str(i).zfill(2)) for i in range(1,12)]]
statsgrpdf2.loc[['HUVEC-{}'.format(str(i).zfill(2)) for i in range(1,25)]]
statsgrpdf2.loc[['RPE-{}'.format(str(i).zfill(2)) for i in range(1,12)]]
statsgrpdf2.loc[['U2OS-{}'.format(str(i).zfill(2)) for i in range(1,6)]]

statsgrpdf1.loc[['U2OS-04']]




#statsgrpdf = statsdf.groupby(['experiment', 'plate', 'Channel'])['Mean', 'Std'].mean().unstack()
statsgrpdf1.unstack().to_csv(os.path.join(PATH, 'stats_grp_256.csv'))


statsdf['experiment'] = statsdf['FileName'].apply(lambda x: x.split('/')[-3])
statsdf['plate'] = statsdf['FileName'].apply(lambda x: x.split('/')[-2])
statsdf['fname'] = statsdf['FileName'].apply(lambda x: x.split('/')[-1])
statsgrpdf = statsdf.groupby(['experiment', 'plate', 'Channel'])['Mean', 'Std'].mean()
experiment_, plate_, _ = ['HEPG2-01', 'Plate1', 'B03_s1_w1.png']
stats_dict = statsgrpdf.loc[(experiment_, plate_)].to_dict()
print(stats_dict)

statsgrpdf.loc[('HEPG2-01', 'Plate1', 1)]['Mean']

statsdf.groupby(['plate', 'Channel'])['Mean', 'Std'].mean() \
    .reset_index() \
    .sort_values(['Channel', 'plate'])
    
statsdf.iloc[0]


def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)
    
def softmax(x)  : return np.exp(x)/sum(np.exp(x))

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm
    
valprob = loadobj(os.path.join(PATH, '../eda/files/val_prods_fold0.pk'))
valprob = sum(valprob )/len(valprob )
valprob.shape
train_dfall = pd.read_csv( os.path.join( PATH, 'train.csv'))#.iloc[:3000]
folddf  = pd.read_csv( os.path.join( PATH, 'folds.csv'))
train_dfall = pd.merge(train_dfall, folddf, on = 'experiment' )

valdf = train_dfall[train_dfall.fold==0]
valdf.head()
valdf.shape
valpdf.iloc[:3,:3].values.argsort(1)
valpdf = pd.DataFrame(valprob, columns = range(valprob.shape[1]))

softmax(valprob)


(valdf.sirna == valprob[:,:1108].argmax(1)).value_counts()
valpdf['sirna'] = valdf.sirna
valpdf = valpdf.groupby(['sirna']).mean()

valpdf.sum(1).hist(bins = 100)
valpdf.values.argsort(1)

pd.DataFrame(valpdf.values.argsort(1)).to_csv(os.path.join(PATH, 'valpdf.csv'))