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
        
        # 设置随机数种子
        random.seed(seed)

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
        half = int(shape[1] * 0.05)
        mid = int(shape[1] / 2)
        if(half == 0):
            low = mid - 1
            high = mid
        else:
            low = mid - half
            high = mid + half
        
        pair_train = int(shape[1] * proportion)

        slice_train = np.array([i for i in range(pair_train)] + [i for i in range(shape[1]-pair_train, shape[1], 1)])

        slice_test = np.array([i for i in range(low, high, 1)])
        
        
        index_test = []
        for id in range(dataset_size):
            if(label[id][1] in slice_test):
                index_test.append(id)
        
        index_train = []
        for id in range(dataset_size):
            if(label[id][1] in slice_train):
                index_train.append(id)
        
        index_test = np.array(index_test)
        index_train = np.array(index_train)

        self.picture_num_test = index_test.shape[0]
        if (mode == 'test'):
            self.feature_test = feature[index_test]
            self.label_test = label[index_test]
        
        self.picture_num_train = index_train.shape[0]
        self.feature_train = feature[index_train]
        self.label_train = label[index_train]

        # 打印信息
        print(slice_test)
        print(slice_train)
        print(dataset_size)
        print(self.picture_num_test)
        print(self.picture_num_train)
        print(self.picture_num_test / dataset_size)
        print(self.picture_num_train / dataset_size)
        
        # 对数据进行归一化处理(按slice)
        if(squeeze):
            self.feature_train = self.feature_train.reshape(-1, shape[2]*shape[3])
            self.max_list_train = np.max(self.feature_train, axis=1).reshape(-1, 1)
            self.feature_train = self.feature_train / self.max_list_train
            self.feature_train = self.feature_train.reshape(-1, 1, shape[2], shape[3])
            if(mode == 'test'):
                self.feature_test = self.feature_test.reshape(-1, shape[2]*shape[3])
                self.max_list_test = np.max(self.feature_test, axis=1).reshape(-1, 1)
                self.feature_test = self.feature_test / self.max_list_test
                self.feature_test = self.feature_test.reshape(-1, 1, shape[2], shape[3])
        
    def __getitem__(self, index:int):
        if(self.mode == 'train'):
            return self.feature_train[index], self.label_train[index], self.max_list_train[index]
        if(self.mode == 'test'):
            return self.feature_test[index], self.label_test[index], self.max_list_test[index]
    
    def __len__(self):
        if(self.mode == 'train'):
            return self.picture_num_train
        if(self.mode == 'test'):
            return self.picture_num_test


if __name__ == '__main__':
    # name_id = 1
    # seed = 1
    # T = Brain_Dataset(mode='test', name_id=name_id, seed=seed, squeeze=True, proportion=0.005)
    proportion_list = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    seed = 0
    for name_id in range(7):
        print(name_id)
        for proportion in proportion_list:
            T = Brain_Dataset(mode='test', name_id=name_id, seed=seed, squeeze=True, proportion=proportion)

    # train_loader = torch.utils.data.DataLoader(T, batch_size=8, shuffle=False)
    # for data, label, max in train_loader:
    #     # print(data.shape)
    #     print(label)
    #     # print(max.shape)
    #     break