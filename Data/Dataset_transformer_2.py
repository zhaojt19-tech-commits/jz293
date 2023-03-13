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
num_slice = 70

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


        # 划分训练集和测试集
        index_all = [i for i in range(dataset_size)]
        index_test = np.array(random.sample(index_all, int(proportion*dataset_size)))
        index_train = np.array(list(set(index_all) - set(index_test)))

        self.picture_num_train = index_train.shape[0]
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
            images = self.feature_train[index_temp].reshape(-1, shape[2], shape[3])

            ts = label_slice_train[index_temp].reshape(-1, 1)

            self.data_train.append(images) # 70 * 109 * 89
            self.y_train.append(np.array([gene] * index_temp.shape[1]).squeeze()) # 70
            self.t_train.append(ts) # 70
            self.gene_list[gene] = index
            index = index + 1

        if (mode == 'test'):
            self.feature_test = feature[index_test]
            self.label_test = label[index_test]

            # 去掉训练集中没有的gene与slice
            gene_train = self.label_train[:, 0]
            slice_train = self.label_train[:, 1]
            gene_test = self.label_test[:, 0]
            slice_test = self.label_test[:, 1]

            id_list = []
            for id in range(len(gene_test)):
                if((gene_test[id] not in gene_train) or (slice_test[id] not in slice_train)):
                    id_list.append(id)
            self.feature_test = np.delete(self.feature_test, id_list, 0)
            self.label_test = np.delete(self.label_test, id_list, 0)
            self.picture_num_test = self.label_test.shape[0]

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
                    continue
                images = self.feature_test[index_temp].reshape(-1, shape[2], shape[3])

                ts = label_slice_test[index_temp].reshape(-1, 1)
                
                self.data_test.append(images) # 70 * 109 * 89
                self.y_test.append(np.array([gene] * index_temp.shape[1]).squeeze()) # 70
                self.t_test.append(ts) # 70

        # print(gene_empty_list)
        # print(id_test)
        # print(id_list)
        # print(self.picture_num_train)
        # print(self.feature_train.shape[0])
        # print(self.picture_num_test)
        # print(self.feature_test.shape[0])
        # print(dataset_size - self.picture_num_train - self.picture_num_test)
        # print(self.picture_num_train / (self.picture_num_train + self.picture_num_test))
        # print(self.picture_num_test / (self.picture_num_train + self.picture_num_test))
        # print(slice_num_max)
    def __getitem__(self, index: int):
        if (self.mode == 'train'):
            if(self.squeeze):
                return self.data_train[index], self.y_train[index], self.t_train[index]
            return self.data_train[index], self.y_train[index], self.t_train[index]
        if (self.mode == 'test'):
            if(self.squeeze):
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
    # for seed in range(1):
    #     print(seed, end = '')
    #     print('++++++++++++++++++++++++++++')
    #     for name_id in range(7):
    #         T = Brain_Dataset(mode='test', name_id=name_id, seed=seed, squeeze=True)
            # gene_train = set(T.label_train[:, 0])
            # slice_train = set(T.label_train[:, 1])
            # gene_test = set(T.label_test[:, 0])
            # slice_test = set(T.label_test[:, 1])
            # print(name_id, end=': ')
            # print(gene_test.issubset(gene_train), end=' ')
            # print(slice_test.issubset(slice_train))
    
     dataset_test = Brain_Dataset(mode='test', transform=None, name_id=1, seed=0, proportion=0.1, squeeze=True)
     test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)
     for t, (images, labels, ts) in enumerate(test_loader):
        print(images.shape)
        print(labels.shape)
        print(ts.shape)
        break


