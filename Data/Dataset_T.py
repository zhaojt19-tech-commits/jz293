from torch.utils import data
import numpy as np
import torch
from scipy import sparse
import random
from torchvision.transforms import Normalize
import torchvision.transforms as transforms
from tqdm import tqdm

def initialize_random_seed(seed=0):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]
id_list = [0, 1, 2, 3, 4]
part_list = ['full', 'brain']

class Brain_Dataset(data.Dataset):
    def __init__(self, mode='train', transform=None, seed=0, proportion=0.1, squeeze=True, part='full'):
        self.transform = transform
        self.mode = mode
        self.squeeze = squeeze
        self.part = part
        
         # 设置随机数种子
        initialize_random_seed(seed)


        if(part == 'full'):
            dir_image = '../Data/Data_time/image_full.npz'
            dir_label = '../Data/Data_time/label_full.npy'
            shape = [75, 70]
            num_gene = 2079
        if(part == 'brain'):
            dir_image = '../Data/Data_time/image_brain.npz'
            dir_label = '../Data/Data_time/label_brain.npy'
            shape = [40, 68]
            num_gene = 1985

        # 读取的原始数据
        self.feature = []
        self.label = []
        
        # 加载图片数据
        feature = sparse.load_npz(dir_image)
        feature = feature.toarray().reshape(-1, shape[0], shape[1])

        # 加载标签数据
        label = np.load(dir_label)

        # 数据集全部非空白图片数量
        dataset_size = feature.shape[0]

        # 划分训练集和测试集
        index_all = [i for i in range(dataset_size)]
        index_test = np.array(random.sample(index_all, int(proportion*dataset_size)))
        index_train = np.array(list(set(index_all) - set(index_test)))

        self.picture_num_train = index_train.shape[0]

        self.feature_train = feature[index_train]
        self.label_train = label[index_train]

        if (mode == 'test'):
            self.feature_test = feature[index_test]
            self.label_test = label[index_test]

            # 去掉训练集中没有的gene_id, slice_id, time
            label_time_train = self.label_train[:, 0]
            label_gene_train = self.label_train[:, 1]
            label_slice_train = self.label_train[:, 2]

            label_time_test = self.label_test[:, 0]
            label_gene_test = self.label_test[:, 1]
            label_slice_test = self.label_test[:, 2]

            id_list = []
            for id in range(len(label_gene_test)):
                if((label_gene_test[id] not in label_gene_train) or (label_slice_test[id] not in label_slice_train) or label_time_test[id] not in label_time_train):
                    id_list.append(id)

            self.feature_test = np.delete(self.feature_test, id_list, 0)
            self.label_test = np.delete(self.label_test, id_list, 0)

            self.picture_num_test = self.label_test.shape[0]
        
        # 对数据进行归一化处理(按slice)
        if(squeeze):
            self.feature_train = self.feature_train.reshape(-1, shape[0]*shape[1])
            self.max_list_train = np.max(self.feature_train, axis=1).reshape(-1, 1)
            self.feature_train = self.feature_train / self.max_list_train
            self.feature_train = self.feature_train.reshape(-1, 1, shape[0], shape[1])
            if(mode == 'test'):
                self.feature_test = self.feature_test.reshape(-1, shape[0]*shape[1])
                self.max_list_test = np.max(self.feature_test, axis=1).reshape(-1, 1)
                self.feature_test = self.feature_test / self.max_list_test
                self.feature_test = self.feature_test.reshape(-1, 1, shape[0], shape[1])
        
        # print(id_list)
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




if __name__ == '__main__':
    for seed in range(5):
        print(seed, end = '')
        print('++++++++++++++++++++++++++++')
        for name_id in range(2):
            T = Brain_Dataset(mode='test', seed=seed, squeeze=True, part=part_list[name_id])

            time_train = set(T.label_train[:, 0])
            gene_train = set(T.label_train[:, 1])
            slice_train = set(T.label_train[:, 2])

            time_test = set(T.label_test[:, 0])
            gene_test = set(T.label_test[:, 1])
            slice_test = set(T.label_test[:, 2])

            print(part_list[name_id], end=': ')
            print(gene_test.issubset(gene_train), end=' ')
            print(slice_test.issubset(slice_train), end = ' ')
            print(time_test.issubset(time_train))