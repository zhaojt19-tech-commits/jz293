from torch.utils import data
import numpy as np
import torch
from scipy import sparse
import random
import numpy as np
import torch


name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]

class Brain_Dataset(data.Dataset):
    def __init__(self, mode='train', transform=None, name_id=0, seed=0, proportion=0.05, squeeze=True):
        self.mode = mode
        self.transform = transform
        self.squeeze = squeeze
         # 读取的原始数据
        self.feature = []
        self.label = []

        name = name_list[name_id]
        shape = shape_list[name_id]
        
        # # 设置随机数种子
        # random.seed(seed)

        dir_image = '../Data/Data_non_blank/image_' + name + '.npz'
        dir_label = '../Data/Data_non_blank/label_' + name + '.npy'

        # 读取全部非空白图片
        feature = sparse.load_npz(dir_image)
        feature = feature.toarray().reshape((-1, shape[2], shape[3]))
        # 读取全部非空白图片的标签
        label = np.load(dir_label)
        
        # 数据集全部非空白图片数量
        dataset_size = feature.shape[0]
        
        # 抽取专用于测试集的slice id
        half = int(shape[1] * proportion)
        mid = int(shape[1] / 2)
        if(half == 0):
            low = mid - 1
            high = mid
        else:
            low = mid - half
            high = mid + half

        slice_test = np.array([i for i in range(low, high, 1)])
        # print(slice_test)
        id_list = []
        for id in range(dataset_size):
            if(label[id][1] in slice_test):
                id_list.append(id)
        
        id_list = np.array(id_list)
        # self.picture_num_train = dataset_size - self.picture_num_test
        self.feature_train = np.delete(feature, id_list, 0)
        self.label_train = np.delete(label, id_list, 0)
        # self.picture_num_test = id_list.shape[0]
        if (mode == 'test'):
            self.feature_test = feature[id_list]
            self.label_test = label[id_list]
        
            # 去掉训练集中没有的gene
            gene_train = self.label_train[:, 0]
            gene_test = self.label_test[:, 0]
            id_list = []
            for id in range(len(gene_test)):
                if((gene_test[id] not in gene_train)):
                    id_list.append(id)
            self.feature_test = np.delete(self.feature_test, id_list, 0)
            self.label_test = np.delete(self.label_test, id_list, 0)
        
            self.picture_num_test = self.label_test.shape[0]

        self.picture_num_train = self.label_train.shape[0]
        
        # 对数据进行归一化处理(按slice)
        if(squeeze):
            max = np.max(feature)
            min = np.min(feature)
            self.feature_train = (self.feature_train - min) / (max - min)
            self.feature_train = self.feature_train.reshape(-1, 1, shape[2], shape[3])
            # self.max_list_train = np.max(self.feature_train, axis=1).reshape(-1, 1)
            # self.feature_train = self.feature_train / self.max_list_train
            # self.feature_train = self.feature_train.reshape(-1, 1, shape[2], shape[3])
            if(mode == 'test'):
                self.feature_test = (self.feature_test - min) / (max - min)
                self.feature_test = self.feature_test.reshape(-1, 1, shape[2], shape[3])
                # self.max_list_test = np.max(self.feature_test, axis=1).reshape(-1, 1)
                # self.feature_test = self.feature_test / self.max_list_test
                # self.feature_test = self.feature_test.reshape(-1, 1, shape[2], shape[3])
        
        # print(slice_test)
        # print(self.picture_num_train)
        # print(self.feature_train.shape[0])
        # print(self.picture_num_test)
        # print(self.feature_test.shape[0])
        # print(dataset_size - self.picture_num_train - self.picture_num_test)
        # print(self.picture_num_train / (self.picture_num_train + self.picture_num_test))
        # print(self.picture_num_test / (self.picture_num_train + self.picture_num_test))
        
    def __getitem__(self, index:int):
        if(self.mode == 'train'):
            return self.feature_train[index], self.label_train[index]
        if(self.mode == 'test'):
            return self.feature_test[index], self.label_test[index]
    
    def __len__(self):
        if(self.mode == 'train'):
            return self.picture_num_train
        if(self.mode == 'test'):
            return self.picture_num_test
