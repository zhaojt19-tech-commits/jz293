from torch.utils import data
import numpy as np
import torch
from scipy import sparse
import random


class Brain_Dataset(data.Dataset):
    def __init__(self, mode='train', transform=None, id = 0):
        self.transform = transform
        self.mode = mode
        self.feature = []
        self.label = []
        if(mode == 'train'):
            feature = sparse.load_npz('../CVAE/Sample/feature_non_blank.npz')
            feature = feature.toarray().reshape((-1, 1, 109, 89))
            label = np.load('../CVAE/Sample/label_non_blank.npy')
            label_0 = label[:, 0]
            index_201 = np.array(np.where((label_0 >= id*200) & (label_0 <= (id+1)*200+1))).ravel()
            index_1800 = np.array(np.where((label_0 < id*200) | (label_0 > (id+1)*200+1))).ravel()

            temp = [i for i in range(index_201.shape[0])]
            test = random.sample(range(index_201.shape[0]), int(index_201.shape[0]*0.5))
            train = list(set(temp) - set(test))

            index_test = index_201[test]
            index_train = np.concatenate((index_1800, index_201[train]))
            np.save('../CVAE/Sample/index_test_'+str(id), index_test)
            self.feature_train = feature[index_train]
            self.label_train = label[index_train]
        if(mode == 'test'):
            feature = sparse.load_npz('../CVAE/Sample/feature_non_blank.npz')
            label = np.load('../CVAE/Sample/label_non_blank.npy')
            index_test = np.load('../CVAE/Sample/index_test_'+str(id) + '.npy')
            index_temp = [i for i in range(label.shape[0])]
            index_train = list(set(index_temp) - set(index_test))
            index_train.sort()
            index_train = np.array(index_train)
            feature = feature.toarray().reshape((-1, 1, 109, 89))
            self.feature_test = feature[index_test]
            self.feature_train = feature[index_train]
            self.label_test = label[index_test]
            self.label_train = label[index_train]

    def __getitem__(self, index:int):
        if(self.mode == 'train'):
            return self.feature_train[index], self.label_train[index]
        if(self.mode == 'test'):
            return self.feature_test[index], self.label_test[index]
    def __len__(self):
        if(self.mode == 'train'):
            return self.feature_train.shape[0]
        if(self.mode == 'test'):
            return self.feature_test.shape[0]




