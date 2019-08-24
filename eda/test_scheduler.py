# https://github.com/ildoonet/pytorch-gradual-warmup-lr
# https://github.com/PavelOstyakov/predictions_balancing/blob/master/run.py
import pickle
import argparse
import os
import torch
import math
import pandas as pd
import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import os
PATH = '/Users/dhanley2/Documents/Personal/recursion'
os.chdir(PATH)
      
from logs import get_logger
from utils import dumpobj, loadobj, GradualWarmupScheduler

EPOCHS = 100
lrmult = 20
lr = 0.000025
cutmix_prob = 1.0
            
model = torchvision.models.densenet121(pretrained=True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)


def get_curr_para(epoch, para_min=0.00, para_max=1.0, peakepoch=70):
    para_curr = para_max - (para_max - para_min) * (1 + math.cos(math.pi * epoch / peakepoch)) / 2
    para_curr = para_curr if epoch<peakepoch else para_max
    return para_curr


# Cutmix
pd.Series(get_curr_para(epoch) for epoch in range(100)).plot.line()


# Sprinkles probability
pd.Series(get_curr_para(epoch, para_max=0.5, peakepoch=70) for epoch in range(100)).plot.line()


# sprinkles hole size
pd.Series(int(get_curr_para(epoch, para_max=32, peakepoch=100)) for epoch in range(100)).plot.line()

# Number of holes
pd.Series(int(get_curr_para(epoch, para_min=4, para_max=8, peakepoch=50)) for epoch in range(100)).plot.line()
