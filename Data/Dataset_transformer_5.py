from torch.utils import data
import numpy as np
import torch
from scipy import sparse
import random
from torchvision.transforms import Normalize
from tqdm import tqdm

name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]

class Brain_Dataset(data.Dataset):
    def __init__(self, mode='train', transform=None, name_id=0, seed=0, proportion=0.2, squeeze=True):
        self.transform = transform
        self.mode = mode
        self.squeeze = squeeze
        name = name_list[name_id]
        shape = shape_list[name_id]

        dir_image = '../Data/Data_non_blank/image_' + name + '.npz'
        dir_label = '../Data/Data_non_blank/label_' + name + '.npy'

        # 读取的原始数据
        self.feature = []
        self.label = []
        # 处理后的数据
        self.data_train = []
        self.y_train = []
        self.t_train = []
        
        # 加载图片数据
        feature = sparse.load_npz(dir_image)
        feature = feature.toarray().reshape(-1, shape[2], shape[3])
        # 加载标签数据
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
        self.picture_num_train = index_train.shape[0]

        self.feature_train = feature[index_train]
        self.label_train = label[index_train]
        
        label_gene_train = self.label_train[:, 0]
        label_slice_train = self.label_train[:, 1]
        gene_empty_list = []
        self.gene_list = {}
        index = 0
        for gene in range(shape[0]):
            index_temp = np.array(np.where(label_gene_train == gene))
            if(index_temp.shape[1] == 0):
                gene_empty_list.append(gene)
                continue
            self.data_train.append(self.feature_train[index_temp].reshape(-1, shape[2], shape[3])) # N * 109 * 89
            self.y_train.append(np.array([gene] * index_temp.shape[1]).squeeze())# N
            self.t_train.append(label_slice_train[index_temp].squeeze())# N
            self.gene_list[gene] = index
            index = index + 1
            
        if (mode == 'test'):
            self.feature_test = feature[index_test]
            self.label_test = label[index_test]

            label_gene_test = self.label_test[:, 0]
            label_slice_test = self.label_test[:, 1]
            self.data_test = []
            self.y_test = []
            self.t_test = []
            label_gene_range = list(set(label_gene_test))
            label_gene_range.sort()
            for gene in label_gene_range:
                index_temp = np.array(np.where(label_gene_test == gene))
                if(index_temp.shape[1] == 0 or (gene in gene_empty_list)):
                    # print('empty')
                    continue
                self.data_test.append(self.feature_test[index_temp].reshape(-1, shape[2], shape[3]))
                self.y_test.append(np.array([gene] * index_temp.shape[1]).squeeze())
                self.t_test.append(label_slice_test[index_temp].squeeze())

        # 对数据进行压缩处理(按slice)
        if(squeeze):
            squeeze_list = []
            self.max_list_train = []
            for gene in self.data_train:
                temp = []
                gene_squ = []
                for slice in gene:
                    max = np.max(slice)
                    temp.append(max)
                    gene_squ.append(slice / max)
                temp = np.array(temp)
                gene_squ = np.array(gene_squ)
                self.max_list_train.append(temp)
                squeeze_list.append(gene_squ)
            self.data_train = squeeze_list
            if(mode == 'test'):
                squeeze_list = []
                self.max_list_test = []
                for gene in self.data_test:
                    temp = []
                    gene_squ = []
                    for slice in gene:
                        max = np.max(slice)
                        temp.append(max)
                        gene_squ.append(slice / max)
                    temp = np.array(temp)
                    gene_squ = np.array(gene_squ)
                    self.max_list_test.append(temp)
                    squeeze_list.append(gene_squ)
                self.data_test = squeeze_list
        # 打印信息
        print(slice_test)
        print(slice_train)
        print(dataset_size)
        print(self.picture_num_test)
        print(self.picture_num_train)
        print(self.picture_num_test / dataset_size)
        print(self.picture_num_train / dataset_size)
        
    def __getitem__(self, index: int):
        if (self.mode == 'train'):
            if(self.squeeze):
                return self.data_train[index], self.y_train[index], self.t_train[index], self.max_list_train[index]
            return self.data_train[index], self.y_train[index], self.t_train[index]
        if (self.mode == 'test'):
            if(self.squeeze):
                return self.data_test[index], self.y_test[index], self.t_test[index], self.max_list_test[index]
            return self.data_test[index], self.y_test[index], self.t_test[index]

    def __len__(self):
        if (self.mode == 'train'):
            return len(self.data_train)
        if (self.mode == 'test'):
            return len(self.data_test)
    def get_picture_num(self):
        if (self.mode == 'train'):
            return self.picture_num_train
        if (self.mode == 'test'):
            return self.picture_num_test

if __name__ == '__main__':
    proportion_list = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    seed = 0
    for name_id in range(7):
        print(name_id)
        for proportion in proportion_list:
            print(proportion)
            T = Brain_Dataset(mode='test', name_id=name_id, seed=seed, squeeze=True, proportion=proportion)
    



