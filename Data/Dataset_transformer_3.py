from torch.utils import data
import numpy as np
import torch
from scipy import sparse
import random
from torchvision.transforms import Normalize
from tqdm import tqdm

def initialize_random_seed(seed=0):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]

class Brain_Dataset(data.Dataset):
    def __init__(self, mode='train', transform=None, name_id=0, seed=0, proportion=0.1, squeeze=True):
        self.transform = transform
        self.mode = mode
        self.squeeze = squeeze
        name = name_list[name_id]
        shape = shape_list[name_id]

        # 设置随机数种子
        initialize_random_seed(seed)

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
        slice_all = [i for i in range(shape[1])]
        slice_test = np.array(random.sample(slice_all, int(shape[1]*proportion)))
        id_list = []
        for id in range(dataset_size):
            if(label[id][1] in slice_test):
                id_list.append(id)
        
        id_list = np.array(id_list)

        self.picture_num_test = id_list.shape[0]

        self.picture_num_train = dataset_size - self.picture_num_test
        index_all = [i for i in range(dataset_size)]
        index_train = np.array(list(set(index_all) - set(id_list)))
        self.feature_train = feature[index_train]
        self.label_train = label[index_train]

        if(squeeze):
            max = np.max(feature)
            min = np.min(feature)
            self.feature_train = (self.feature_train - min) / (max - min)
            self.feature_train = self.feature_train.reshape(-1, 1, shape[2], shape[3])
        #     self.max_list_train = np.max(self.feature_train, axis=1).reshape(-1, 1)
        #     self.feature_train = self.feature_train / self.max_list_train
        #     self.feature_train = self.feature_train.reshape(-1, 1, shape[2], shape[3])

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

        if(mode == 'test'):
            self.feature_test = feature[id_list]
            self.label_test = label[id_list]

            if(squeeze):
                self.feature_test = (self.feature_test - min) / (max - min)
                 self.feature_test = self.feature_test.reshape(-1, 1, shape[2], shape[3])
                # self.max_list_test = np.max(self.feature_test, axis=1).reshape(-1, 1)
                # self.feature_test = self.feature_test / self.max_list_test
                # self.feature_test = self.feature_test.reshape(-1, 1, shape[2], shape[3])

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
                    self.picture_num_test = self.picture_num_test - index_temp.shape[1]
                    # print('empty')
                    continue
                self.data_test.append(self.feature_test[index_temp].reshape(-1, shape[2], shape[3]))
                self.y_test.append(np.array([gene] * index_temp.shape[1]).squeeze())
                self.t_test.append(label_slice_test[index_temp].squeeze())

        # print(slice_test)
        # print(self.picture_num_train)
        # print(self.feature_train.shape[0])
        # print(self.picture_num_test)
        # print(self.feature_test.shape[0])
        # print(dataset_size - self.picture_num_train - self.picture_num_test)
        # print(self.picture_num_train / (self.picture_num_train + self.picture_num_test))
        # print(self.picture_num_test / (self.picture_num_train + self.picture_num_test))
    def __getitem__(self, index: int):
        if (self.mode == 'train'):
            if(self.squeeze):
                return self.data_train[index], self.y_train[index], self.t_train[index]
            return self.data_train[index], self.y_train[index], self.t_train[index]
        if (self.mode == 'test'):
            if(self.squeeze):
                return self.data_test[index], self.y_test[index], self.t_test[index]
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
    for seed in range(5):
        print(seed, end = '')
        print('++++++++++++++++++++++++++++')
        for name_id in range(7):
            name_id = 4
            T = Brain_Dataset(mode='test', name_id=name_id, seed=seed, squeeze=True, proportion=0.1)
            gene_train = set(T.label_train[:, 0])
            slice_train = set(T.label_train[:, 1])
            gene_test = set(T.label_test[:, 0])
            slice_test = set(T.label_test[:, 1])
            print(name_id, end=': ')
            print(gene_test.issubset(gene_train), end=' ')
            gene_cross = gene_train.intersection(gene_test)
            print(gene_test - gene_cross)
            print(slice_test.issubset(slice_train))
            break
        break
    



