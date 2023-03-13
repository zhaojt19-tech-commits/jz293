import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import sparse, stats
from MyDataset_10K import *
import os

if __name__ == '__main__':
    # Create two lists of random values
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [2, 1, 2, 4.5, 7, 6.5, 6, 9, 9.5]
    print(stats.pearsonr(x, y))
    # 0.9412443251336238
    print(stats.spearmanr(x, y))
    # 0.903773601456181

    
    
    