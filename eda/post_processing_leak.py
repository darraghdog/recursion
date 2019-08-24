#usage python post_processing.py ./submit_17.txt ./submit_17_submit_postprocess.csv

import argparse
import os
import torch
import tqdm
import numpy as np
import pandas as pd


def _get_train_groups():
    '''
    # We store the groups in the script so we do not need to reload train, but below is how we get them
    traindf = pd.read_csv('train.csv')
    plate_groups = np.zeros((1108,4), int)
    for sirna in range(1108):
        grp = traindf.loc[traindf.sirna==sirna,:].plate.value_counts().index.values
        assert len(grp) == 3
        plate_groups[sirna,0:3] = grp
        plate_groups[sirna,3] = 10 - grp.sum()
    '''
    grp1 = [4,1,2,1,3,1,1,2,1,4,1,2,4,4,4,4,4,4,4,2,2,3,2,4,3,2,1,2,2,3,4,4,3,1,3,3,4,4,3,3,1,2,2,2,2,4,1,2,1,3,4,3,2,3,4,1,1,2,4,1,2,2,3,2,2,3,2,4,4,4,1,4,2,4,1,2,3,1,2,4,1,2,4,2,3,4,3,2,1,1,3,1,2,2,1,2,3,4,3,3,2,2,1,4,4,1,4,4,2,1,4,2,1,1,3,3,4,2,2,3,3,3,3,3,1,3,1,3,4,3,2,2,2,1,3,3,1,4,2,4,4,4,3,4,1,3,2,1,3,2,4,3,4,2,4,1,4,1,1,3,2,4,1,3,2,2,2,3,3,2,3,3,1,3,3,2,4,1,2,4,3,1,2,3,3,3,3,4,1,2,3,3,1,4,2,1,2,3,3,4,4,4,4,1,4,4,2,2,4,4,3,1,1,4,3,3,2,1,2,2,4,1,1,4,1,2,4,4,2,3,4,4,2,2,3,4,2,3,1,1,4,4,1,4,3,2,3,4,3,4,3,4,3,1,1,3,1,2,1,3,4,4,2,2,4,3,1,3,1,2,4,3,1,4,1,4,1,1,3,3,4,1,1,3,2,2,4,1,4,3,1,3,4,1,4,1,1,4,4,2,3,1,1,2,4,2,4,3,1,1,3,3,1,1,3,2,4,2,4,3,1,4,3,4,1,4,3,2,4,2,2,4,4,2,2,3,2,3,3,1,1,1,2,4,4,1,1,3,1,2,1,2,1,2,4,1,4,2,4,3,1,1,3,3,2,3,1,4,3,3,4,1,1,1,4,3,2,3,4,1,3,2,3,2,2,1,4,3,1,1,2,1,1,2,3,1,3,1,1,1,1,3,3,4,2,3,4,2,2,2,3,1,2,1,2,1,4,1,4,4,4,2,3,2,2,2,4,4,2,2,4,1,3,4,2,3,1,3,3,1,4,4,1,3,3,1,3,2,3,3,1,4,2,3,2,4,1,2,4,1,1,2,2,3,3,3,2,2,4,3,4,1,4,2,4,2,4,3,1,2,3,1,1,2,3,3,1,3,4,1,3,3,2,4,3,2,4,1,4,2,1,3,2,1,3,1,1,2,1,4,3,2,3,1,1,3,3,2,4,4,3,2,4,1,4,3,4,3,4,2,3,4,1,3,4,2,2,1,1,2,4,1,2,1,2,3,3,1,3,1,2,3,1,2,3,1,1,1,2,1,4,4,2,2,3,1,1,1,4,2,3,2,1,4,4,4,1,1,3,2,2,1,1,3,2,1,3,4,1,4,3,4,4,3,4,3,4,2,1,3,2,2,4,3,1,1,1,1,3,4,4,1,4,1,3,2,3,3,1,4,1,4,1,2,1,3,1,3,4,3,4,4,3,3,4,4,3,3,4,2,2,4,2,3,3,4,3,1,3,3,2,2,3,1,2,1,4,1,3,2,1,2,2,4,3,3,3,2,4,2,1,3,3,1,3,4,1,3,3,3,4,1,2,2,4,2,3,1,4,1,3,3,1,4,2,1,4,2,4,1,1,2,4,1,1,1,2,3,1,2,4,4,4,3,2,1,3,2,3,3,4,2,2,2,2,4,2,3,4,2,2,2,2,2,1,2,2,4,3,2,3,4,1,1,1,1,4,2,3,3,1,2,2,2,2,3,4,1,4,3,1,2,1,3,1,1,2,4,3,3,1,4,4,4,3,1,2,3,4,2,4,2,3,2,1,4,1,2,2,2,2,1,4,1,4,3,3,1,1,4,3,2,3,4,2,3,1,3,3,3,1,1,4,2,4,3,3,4,3,3,3,3,4,4,4,1,2,2,1,3,2,1,4,4,3,4,4,2,1,4,1,4,3,4,2,1,4,1,3,2,4,2,2,1,4,4,2,2,3,2,3,3,4,3,3,2,3,2,1,3,1,2,4,3,2,1,2,1,4,2,4,1,4,1,1,4,3,2,3,1,1,1,4,2,1,3,2,1,2,4,1,3,2,2,4,2,4,1,3,2,1,4,4,2,1,2,3,2,2,3,2,1,3,2,4,4,3,2,4,4,4,4,2,3,2,4,3,3,1,1,3,2,3,3,3,1,2,3,4,4,1,2,2,4,4,3,2,2,4,1,1,3,1,3,4,2,2,2,2,3,3,4,1,4,1,2,4,3,1,4,4,1,1,1,1,4,1,1,2,3,3,2,3,4,4,3,2,4,4,3,3,4,2,2,4,4,1,3,1,4,4,4,3,1,3,1,2,2,3,2,1,4,4,1,2,1,4,1,1,3,2,1,2,2,3,2,4,4,4,1,2,1,1,2,2,4,2,2,4,2,4,3,2,3,4,4,1,3,2,3,1,4,1,2,2,1,1,1,3,4,2,3,4,4,3,4,1,2,2,3,2,4,4,2,1,3,3,3,2,1,2,3,2,3,4,4,1,4,2,3,1,1,1,4,1,3,1,4]
    grp2 = [2,3,4,3,1,3,3,4,3,2,3,4,2,2,2,2,2,2,2,4,4,1,4,2,1,4,3,4,4,1,2,2,1,3,1,1,2,2,1,1,3,4,4,4,4,2,3,4,3,1,2,1,4,1,2,3,3,4,2,3,4,4,1,4,4,1,4,2,2,2,3,2,4,2,3,4,1,3,4,2,3,4,2,4,1,2,1,4,3,3,1,3,4,4,3,4,1,2,1,1,4,4,3,2,2,3,2,2,4,3,2,4,3,3,1,1,2,4,4,1,1,1,1,1,3,1,3,1,2,1,4,4,4,3,1,1,3,2,4,2,2,2,1,2,3,1,4,3,1,4,2,1,2,4,2,3,2,3,3,1,4,2,3,1,4,4,4,1,1,4,1,1,3,1,1,4,2,3,4,2,1,3,4,1,1,1,1,2,3,4,1,1,3,2,4,3,4,1,1,2,2,2,2,3,2,2,4,4,2,2,1,3,3,2,1,1,4,3,4,4,2,3,3,2,3,4,2,2,4,1,2,2,4,4,1,2,4,1,3,3,2,2,3,2,1,4,1,2,1,2,1,2,1,3,3,1,3,4,3,1,2,2,4,4,2,1,3,1,3,4,2,1,3,2,3,2,3,3,1,1,2,3,3,1,4,4,2,3,2,1,3,1,2,3,2,3,3,2,2,4,1,3,3,4,2,4,2,1,3,3,1,1,3,3,1,4,2,4,2,1,3,2,1,2,3,2,1,4,2,4,4,2,2,4,4,1,4,1,1,3,3,3,4,2,2,3,3,1,3,4,3,4,3,4,2,3,2,4,2,1,3,3,1,1,4,1,3,2,1,1,2,3,3,3,2,1,4,1,2,3,1,4,1,4,4,3,2,1,3,3,4,3,3,4,1,3,1,3,3,3,3,1,1,2,4,1,2,4,4,4,1,3,4,3,4,3,2,3,2,2,2,4,1,4,4,4,2,2,4,4,2,3,1,2,4,1,3,1,1,3,2,2,3,1,1,3,1,4,1,1,3,2,4,1,4,2,3,4,2,3,3,4,4,1,1,1,4,4,2,1,2,3,2,4,2,4,2,1,3,4,1,3,3,4,1,1,3,1,2,3,1,1,4,2,1,4,2,3,2,4,3,1,4,3,1,3,3,4,3,2,1,4,1,3,3,1,1,4,2,2,1,4,2,3,2,1,2,1,2,4,1,2,3,1,2,4,4,3,3,4,2,3,4,3,4,1,1,3,1,3,4,1,3,4,1,3,3,3,4,3,2,2,4,4,1,3,3,3,2,4,1,4,3,2,2,2,3,3,1,4,4,3,3,1,4,3,1,2,3,2,1,2,2,1,2,1,2,4,3,1,4,4,2,1,3,3,3,3,1,2,2,3,2,3,1,4,1,1,3,2,3,2,3,4,3,1,3,1,2,1,2,2,1,1,2,2,1,1,2,4,4,2,4,1,1,2,1,3,1,1,4,4,1,3,4,3,2,3,1,4,3,4,4,2,1,1,1,4,2,4,3,1,1,3,1,2,3,1,1,1,2,3,4,4,2,4,1,3,2,3,1,1,3,2,4,3,2,4,2,3,3,4,2,3,3,3,4,1,3,4,2,2,2,1,4,3,1,4,1,1,2,4,4,4,4,2,4,1,2,4,4,4,4,4,3,4,4,2,1,4,1,2,3,3,3,3,2,4,1,1,3,4,4,4,4,1,2,3,2,1,3,4,3,1,3,3,4,2,1,1,3,2,2,2,1,3,4,1,2,4,2,4,1,4,3,2,3,4,4,4,4,3,2,3,2,1,1,3,3,2,1,4,1,2,4,1,3,1,1,1,3,3,2,4,2,1,1,2,1,1,1,1,2,2,2,3,4,4,3,1,4,3,2,2,1,2,2,4,3,2,3,2,1,2,4,3,2,3,1,4,2,4,4,3,2,2,4,4,1,4,1,1,2,1,1,4,1,4,3,1,3,4,2,1,4,3,4,3,2,4,2,3,2,3,3,2,1,4,1,3,3,3,2,4,3,1,4,3,4,2,3,1,4,4,2,4,2,3,1,4,3,2,2,4,3,4,1,4,4,1,4,3,1,4,2,2,1,4,2,2,2,2,4,1,4,2,1,1,3,3,1,4,1,1,1,3,4,1,2,2,3,4,4,2,2,1,4,4,2,3,3,1,3,1,2,4,4,4,4,1,1,2,3,2,3,4,2,1,3,2,2,3,3,3,3,2,3,3,4,1,1,4,1,2,2,1,4,2,2,1,1,2,4,4,2,2,3,1,3,2,2,2,1,3,1,3,4,4,1,4,3,2,2,3,4,3,2,3,3,1,4,3,4,4,1,4,2,2,2,3,4,3,3,4,4,2,4,4,2,4,2,1,4,1,2,2,3,1,4,1,3,2,3,4,4,3,3,3,1,2,4,1,2,2,1,2,3,4,4,1,4,2,2,4,3,1,1,1,4,3,4,1,4,1,2,2,3,2,4,1,3,3,3,2,3,1,3,2]
    grp3 = [3,4,1,4,2,4,4,1,4,3,4,1,3,3,3,3,3,3,3,1,1,2,1,3,2,1,4,1,1,2,3,3,2,4,2,2,3,3,2,2,4,1,1,1,1,3,4,1,4,2,3,2,1,2,3,4,4,1,3,4,1,1,2,1,1,2,1,3,3,3,4,3,1,3,4,1,2,4,1,3,4,1,3,1,2,3,2,1,4,4,2,4,1,1,4,1,2,3,2,2,1,1,4,3,3,4,3,3,1,4,3,1,4,4,2,2,3,1,1,2,2,2,2,2,4,2,4,2,3,2,1,1,1,4,2,2,4,3,1,3,3,3,2,3,4,2,1,4,2,1,3,2,3,1,3,4,3,4,4,2,1,3,4,2,1,1,1,2,2,1,2,2,4,2,2,1,3,4,1,3,2,4,1,2,2,2,2,3,4,1,2,2,4,3,1,4,1,2,2,3,3,3,3,4,3,3,1,1,3,3,2,4,4,3,2,2,1,4,1,1,3,4,4,3,4,1,3,3,1,2,3,3,1,1,2,3,1,2,4,4,3,3,4,3,2,1,2,3,2,3,2,3,2,4,4,2,4,1,4,2,3,3,1,1,3,2,4,2,4,1,3,2,4,3,4,3,4,4,2,2,3,4,4,2,1,1,3,4,3,2,4,2,3,4,3,4,4,3,3,1,2,4,4,1,3,1,3,2,4,4,2,2,4,4,2,1,3,1,3,2,4,3,2,3,4,3,2,1,3,1,1,3,3,1,1,2,1,2,2,4,4,4,1,3,3,4,4,2,4,1,4,1,4,1,3,4,3,1,3,2,4,4,2,2,1,2,4,3,2,2,3,4,4,4,3,2,1,2,3,4,2,1,2,1,1,4,3,2,4,4,1,4,4,1,2,4,2,4,4,4,4,2,2,3,1,2,3,1,1,1,2,4,1,4,1,4,3,4,3,3,3,1,2,1,1,1,3,3,1,1,3,4,2,3,1,2,4,2,2,4,3,3,4,2,2,4,2,1,2,2,4,3,1,2,1,3,4,1,3,4,4,1,1,2,2,2,1,1,3,2,3,4,3,1,3,1,3,2,4,1,2,4,4,1,2,2,4,2,3,4,2,2,1,3,2,1,3,4,3,1,4,2,1,4,2,4,4,1,4,3,2,1,2,4,4,2,2,1,3,3,2,1,3,4,3,2,3,2,3,1,2,3,4,2,3,1,1,4,4,1,3,4,1,4,1,2,2,4,2,4,1,2,4,1,2,4,4,4,1,4,3,3,1,1,2,4,4,4,3,1,2,1,4,3,3,3,4,4,2,1,1,4,4,2,1,4,2,3,4,3,2,3,3,2,3,2,3,1,4,2,1,1,3,2,4,4,4,4,2,3,3,4,3,4,2,1,2,2,4,3,4,3,4,1,4,2,4,2,3,2,3,3,2,2,3,3,2,2,3,1,1,3,1,2,2,3,2,4,2,2,1,1,2,4,1,4,3,4,2,1,4,1,1,3,2,2,2,1,3,1,4,2,2,4,2,3,4,2,2,2,3,4,1,1,3,1,2,4,3,4,2,2,4,3,1,4,3,1,3,4,4,1,3,4,4,4,1,2,4,1,3,3,3,2,1,4,2,1,2,2,3,1,1,1,1,3,1,2,3,1,1,1,1,1,4,1,1,3,2,1,2,3,4,4,4,4,3,1,2,2,4,1,1,1,1,2,3,4,3,2,4,1,4,2,4,4,1,3,2,2,4,3,3,3,2,4,1,2,3,1,3,1,2,1,4,3,4,1,1,1,1,4,3,4,3,2,2,4,4,3,2,1,2,3,1,2,4,2,2,2,4,4,3,1,3,2,2,3,2,2,2,2,3,3,3,4,1,1,4,2,1,4,3,3,2,3,3,1,4,3,4,3,2,3,1,4,3,4,2,1,3,1,1,4,3,3,1,1,2,1,2,2,3,2,2,1,2,1,4,2,4,1,3,2,1,4,1,4,3,1,3,4,3,4,4,3,2,1,2,4,4,4,3,1,4,2,1,4,1,3,4,2,1,1,3,1,3,4,2,1,4,3,3,1,4,1,2,1,1,2,1,4,2,1,3,3,2,1,3,3,3,3,1,2,1,3,2,2,4,4,2,1,2,2,2,4,1,2,3,3,4,1,1,3,3,2,1,1,3,4,4,2,4,2,3,1,1,1,1,2,2,3,4,3,4,1,3,2,4,3,3,4,4,4,4,3,4,4,1,2,2,1,2,3,3,2,1,3,3,2,2,3,1,1,3,3,4,2,4,3,3,3,2,4,2,4,1,1,2,1,4,3,3,4,1,4,3,4,4,2,1,4,1,1,2,1,3,3,3,4,1,4,4,1,1,3,1,1,3,1,3,2,1,2,3,3,4,2,1,2,4,3,4,1,1,4,4,4,2,3,1,2,3,3,2,3,4,1,1,2,1,3,3,1,4,2,2,2,1,4,1,2,1,2,3,3,4,3,1,2,4,4,4,3,4,2,4,3]
    grp4 = [1,2,3,2,4,2,2,3,2,1,2,3,1,1,1,1,1,1,1,3,3,4,3,1,4,3,2,3,3,4,1,1,4,2,4,4,1,1,4,4,2,3,3,3,3,1,2,3,2,4,1,4,3,4,1,2,2,3,1,2,3,3,4,3,3,4,3,1,1,1,2,1,3,1,2,3,4,2,3,1,2,3,1,3,4,1,4,3,2,2,4,2,3,3,2,3,4,1,4,4,3,3,2,1,1,2,1,1,3,2,1,3,2,2,4,4,1,3,3,4,4,4,4,4,2,4,2,4,1,4,3,3,3,2,4,4,2,1,3,1,1,1,4,1,2,4,3,2,4,3,1,4,1,3,1,2,1,2,2,4,3,1,2,4,3,3,3,4,4,3,4,4,2,4,4,3,1,2,3,1,4,2,3,4,4,4,4,1,2,3,4,4,2,1,3,2,3,4,4,1,1,1,1,2,1,1,3,3,1,1,4,2,2,1,4,4,3,2,3,3,1,2,2,1,2,3,1,1,3,4,1,1,3,3,4,1,3,4,2,2,1,1,2,1,4,3,4,1,4,1,4,1,4,2,2,4,2,3,2,4,1,1,3,3,1,4,2,4,2,3,1,4,2,1,2,1,2,2,4,4,1,2,2,4,3,3,1,2,1,4,2,4,1,2,1,2,2,1,1,3,4,2,2,3,1,3,1,4,2,2,4,4,2,2,4,3,1,3,1,4,2,1,4,1,2,1,4,3,1,3,3,1,1,3,3,4,3,4,4,2,2,2,3,1,1,2,2,4,2,3,2,3,2,3,1,2,1,3,1,4,2,2,4,4,3,4,2,1,4,4,1,2,2,2,1,4,3,4,1,2,4,3,4,3,3,2,1,4,2,2,3,2,2,3,4,2,4,2,2,2,2,4,4,1,3,4,1,3,3,3,4,2,3,2,3,2,1,2,1,1,1,3,4,3,3,3,1,1,3,3,1,2,4,1,3,4,2,4,4,2,1,1,2,4,4,2,4,3,4,4,2,1,3,4,3,1,2,3,1,2,2,3,3,4,4,4,3,3,1,4,1,2,1,3,1,3,1,4,2,3,4,2,2,3,4,4,2,4,1,2,4,4,3,1,4,3,1,2,1,3,2,4,3,2,4,2,2,3,2,1,4,3,4,2,2,4,4,3,1,1,4,3,1,2,1,4,1,4,1,3,4,1,2,4,1,3,3,2,2,3,1,2,3,2,3,4,4,2,4,2,3,4,2,3,4,2,2,2,3,2,1,1,3,3,4,2,2,2,1,3,4,3,2,1,1,1,2,2,4,3,3,2,2,4,3,2,4,1,2,1,4,1,1,4,1,4,1,3,2,4,3,3,1,4,2,2,2,2,4,1,1,2,1,2,4,3,4,4,2,1,2,1,2,3,2,4,2,4,1,4,1,1,4,4,1,1,4,4,1,3,3,1,3,4,4,1,4,2,4,4,3,3,4,2,3,2,1,2,4,3,2,3,3,1,4,4,4,3,1,3,2,4,4,2,4,1,2,4,4,4,1,2,3,3,1,3,4,2,1,2,4,4,2,1,3,2,1,3,1,2,2,3,1,2,2,2,3,4,2,3,1,1,1,4,3,2,4,3,4,4,1,3,3,3,3,1,3,4,1,3,3,3,3,3,2,3,3,1,4,3,4,1,2,2,2,2,1,3,4,4,2,3,3,3,3,4,1,2,1,4,2,3,2,4,2,2,3,1,4,4,2,1,1,1,4,2,3,4,1,3,1,3,4,3,2,1,2,3,3,3,3,2,1,2,1,4,4,2,2,1,4,3,4,1,3,4,2,4,4,4,2,2,1,3,1,4,4,1,4,4,4,4,1,1,1,2,3,3,2,4,3,2,1,1,4,1,1,3,2,1,2,1,4,1,3,2,1,2,4,3,1,3,3,2,1,1,3,3,4,3,4,4,1,4,4,3,4,3,2,4,2,3,1,4,3,2,3,2,1,3,1,2,1,2,2,1,4,3,4,2,2,2,1,3,2,4,3,2,3,1,2,4,3,3,1,3,1,2,4,3,2,1,1,3,2,3,4,3,3,4,3,2,4,3,1,1,4,3,1,1,1,1,3,4,3,1,4,4,2,2,4,3,4,4,4,2,3,4,1,1,2,3,3,1,1,4,3,3,1,2,2,4,2,4,1,3,3,3,3,4,4,1,2,1,2,3,1,4,2,1,1,2,2,2,2,1,2,2,3,4,4,3,4,1,1,4,3,1,1,4,4,1,3,3,1,1,2,4,2,1,1,1,4,2,4,2,3,3,4,3,2,1,1,2,3,2,1,2,2,4,3,2,3,3,4,3,1,1,1,2,3,2,2,3,3,1,3,3,1,3,1,4,3,4,1,1,2,4,3,4,2,1,2,3,3,2,2,2,4,1,3,4,1,1,4,1,2,3,3,4,3,1,1,3,2,4,4,4,3,2,3,4,3,4,1,1,2,1,3,4,2,2,2,1,2,4,2,1]
    plate_groups = np.array([grp1, grp2, grp3, grp4]).T
    return plate_groups

def _get_predicts(predicts, coefficients):
    return torch.einsum("ij,j->ij", (predicts, coefficients))


def _get_labels_distribution(predicts, coefficients):
    predicts = _get_predicts(predicts, coefficients)
    labels = predicts.argmax(dim=-1)
    counter = torch.bincount(labels, minlength=predicts.shape[1])
    return counter


def _compute_score_with_coefficients(predicts, coefficients):
    counter = _get_labels_distribution(predicts, coefficients).float()
    counter = counter * 100 / len(predicts)
    #max_scores = torch.ones(len(coefficients)).cuda().float() * 100 / len(coefficients)
    max_scores = torch.ones(len(coefficients)).float() * 100 / len(coefficients)
    result, _ = torch.min(torch.cat([counter.unsqueeze(0), max_scores.unsqueeze(0)], dim=0), dim=0)

    return float(result.sum().cpu())


def _find_best_coefficients(predicts, coefficients, alpha=0.001, iterations=100):
    best_coefficients = coefficients.clone()
    best_score = _compute_score_with_coefficients(predicts, coefficients)

    for _ in tqdm.trange(iterations):
        counter = _get_labels_distribution(predicts, coefficients)
        label = int(torch.argmax(counter).cpu())
        coefficients[label] -= alpha
        score = _compute_score_with_coefficients(predicts, coefficients)
        if score > best_score:
            best_score = score
            best_coefficients = coefficients.clone()

    return best_coefficients


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--start_alpha", type=float, default=0.01)
    parser.add_argument("--min_alpha", type=float, default=0.0001)

    args = parser.parse_args()
    
    #load to pandas df and convert to npy
    df = pd.read_csv(args.input_path)
    df[['expt','tray', 'cell']] = df['id_code'].str.split('_',expand=True)
    unique_expt = np.unique(df['expt'])
    
    # Process the train leak
    plate_groups = _get_train_groups()
    
    # Assign groups to plates
    all_test_exp = df.expt.unique()
    group_plate_probs = np.zeros((len(all_test_exp),4))
    for idx in range(len(all_test_exp)):
        preds = df.loc[df.expt == all_test_exp[idx]].values[:,:1108].argmax(1)
        pp_mult = np.zeros((len(preds),1108))
        pp_mult[range(len(preds)),preds] = 1
        sub_test = df.loc[df.expt == all_test_exp[idx],:]
        assert len(pp_mult) == len(sub_test)
        for j in range(4):
            mask = np.repeat(plate_groups[np.newaxis, :, j], len(pp_mult), axis=0) == \
                   np.repeat(sub_test.tray.values[:, np.newaxis].astype(np.int8), 1108, axis=1)
            group_plate_probs[idx,j] = np.array(pp_mult)[mask].sum()/len(pp_mult)
    exp_to_group = group_plate_probs.argmax(1) # test group assignments
    
    # Mask out the empty groups
    def _select_plate_group(pp_mult, idx):
        # Create a mask to zero out values
        sub_test = df.loc[df.expt == all_test_exp[idx],:]
        assert len(pp_mult) == len(sub_test)
        mask = np.repeat(plate_groups[np.newaxis, :, exp_to_group[idx]], len(pp_mult), axis=0) != \
               np.repeat(sub_test.tray.values[:, np.newaxis].astype(np.int8), 1108, axis=1)
        pp_mult[mask] = 0
        return pp_mult
    
    probsleak = df.iloc[:,:1108].values
    for t, idx in enumerate(range(len(all_test_exp))):
        indices = (df.expt == all_test_exp[idx])    
        preds = probsleak[indices,:1108].copy()    
        probsleak[indices,:1108] = _select_plate_group(preds, idx)
    df.iloc[:,:1108] = probsleak

    
    #loop through each id_code and find balanced labels
    predslist = []   #list for final prediction
    idcodes = []    #idcodes
    for expt in unique_expt:
        dfunique = df[df['expt']==expt]   #new df for specific expt
        logits = dfunique.to_numpy()           #convert to np array
        idcodes.extend(dfunique['id_code'].tolist())
        logits = logits[:,:1108]                    #only select logits
        logits = logits.astype('float32')              
        y = torch.from_numpy(logits)#.cuda()           #convert to torch tensor
        alpha = args.start_alpha
        print("Starting id_code {}".format(expt))
        #coefs = torch.ones(y.shape[1]).cuda().float()
        coefs = torch.ones(y.shape[1]).float()
        last_score = _compute_score_with_coefficients(y, coefs)
        print("Start score", last_score)
        while alpha >= args.min_alpha:
            if last_score >100:
                print("Score: {}, alpha: {}".format(last_score, alpha))
                alpha = args.min_alpha-0.1
                continue
            coefs = _find_best_coefficients(y, coefs, iterations=3000, alpha=alpha)
            new_score = _compute_score_with_coefficients(y, coefs)

            if new_score <= last_score:
                alpha *= 0.5

            last_score = new_score

            print("Score: {}, alpha: {}".format(last_score, alpha))

        predicts = _get_predicts(y, coefs)   #new final preds in torch format
        predicts = predicts.cpu()
        np_predicts = predicts.numpy()
        np_predicts_argmax = np.argmax(np_predicts, axis=1)
        predslist.extend(np_predicts_argmax)

    #save submission
    df_submit = pd.DataFrame({'id_code': idcodes, 'sirna': predslist})
    df_submit.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    main()
