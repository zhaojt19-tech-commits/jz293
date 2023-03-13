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
# num_slice = 70

class Brain_Dataset(data.Dataset):
    def __init__(self, mode='train', transform=None, name_id=0, seed=0, proportion=0.1, squeeze=True, num_slice=70):
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
        self.mask_train = []
        self.num_non_blank_train = []
        
        # 加载图片数据
        feature = sparse.load_npz(dir_image)
        feature = feature.toarray().reshape(-1, shape[2], shape[3])
        # 加载标签数据
        label = np.load(dir_label)

        # 数据集全部非空白图片数量
        dataset_size = feature.shape[0]
        max = np.max(feature)
        min = np.min(feature)


        # 划分训练集和测试集
        index_all = [i for i in range(dataset_size)]
        index_test = np.array(random.sample(index_all, int(proportion*dataset_size)))
        index_train = np.array(list(set(index_all) - set(index_test)))

        self.picture_num_train = index_train.shape[0]
        self.feature_train = feature[index_train]
        self.label_train = label[index_train]

        if(squeeze):
            self.feature_train = (self.feature_train - min) / (max - min)
            # self.max_list_train = np.max(self.feature_train, axis=1).reshape(-1, 1)
            # self.feature_train = self.feature_train / self.max_list_train
            # self.feature_train = self.feature_train.reshape(-1, 1, shape[2], shape[3])
            
        
        label_gene_train = self.label_train[:, 0]
        label_slice_train = self.label_train[:, 1]
        gene_empty_list = []
        self.gene_list = {}
        index = 0

        # slice_num_max = 0
        for gene in range(shape[0]):
            index_temp = np.array(np.where(label_gene_train == gene))
            # if(slice_num_max < index_temp.shape[1]):
            #     slice_num_max = index_temp.shape[1]
            if(index_temp.shape[1] == 0):
                gene_empty_list.append(gene)
                continue
            images = self.feature_train[index_temp].reshape(-1, shape[2], shape[3])
            images_blank = np.zeros((num_slice-images.shape[0], shape[2]*shape[3])).reshape(-1, shape[2], shape[3])
            images_all = np.concatenate((images, images_blank), axis=0)

            ts = label_slice_train[index_temp].reshape(-1, 1)
            ts_blank = np.zeros(num_slice-images.shape[0]).reshape(-1, 1)
            ts_all = np.concatenate((ts, ts_blank), axis=0)
            
            temp = np.ones(images.shape[0]).reshape(-1, 1)
            mask = np.concatenate((temp, ts_blank), axis=0)

            self.data_train.append(images_all) # 70 * 109 * 89
            self.y_train.append(np.array([gene] * num_slice).squeeze()) # 70
            self.t_train.append(ts_all) # 70
            self.mask_train.append(mask) # 70
            self.num_non_blank_train.append(images.shape[0])
            self.gene_list[gene] = index
            index = index + 1

        self.data_train = np.array(self.data_train)
        self.y_train = np.array(self.y_train)
        self.t_train = np.array(self.t_train)
        self.mask_train = np.array(self.mask_train)
        self.num_non_blank_train = np.array(self.num_non_blank_train)

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
                # self.max_list_test = np.max(self.feature_test, axis=1).reshape(-1, 1)
                # self.feature_test = self.feature_test / self.max_list_test
                # self.feature_test = self.feature_test.reshape(-1, 1, shape[2], shape[3])

            label_gene_test = self.label_test[:, 0]
            label_slice_test = self.label_test[:, 1]
            self.data_test = []
            self.y_test = []
            self.t_test = []
            self.mask_test = []
            self.num_non_blank_test = []
            label_gene_range = list(set(label_gene_test))
            label_gene_range.sort()

            
            for gene in label_gene_range:
                index_temp = np.array(np.where(label_gene_test == gene))
                # if(slice_num_max < index_temp.shape[1]):
                #     slice_num_max = index_temp.shape[1]
                if(index_temp.shape[1] == 0 or (gene in gene_empty_list)):
                    # print('empty')
                    self.picture_num_test = self.picture_num_test - index_temp.shape[1]
                    continue
                images = self.feature_test[index_temp].reshape(-1, shape[2], shape[3])
                images_blank = np.zeros((num_slice-images.shape[0], shape[2]*shape[3])).reshape(-1, shape[2], shape[3])
                images_all = np.concatenate((images, images_blank), axis=0)

                ts = label_slice_test[index_temp].reshape(-1, 1)
                ts_blank = np.zeros(num_slice-images.shape[0]).reshape(-1, 1)
                ts_all = np.concatenate((ts, ts_blank), axis=0)
                
                temp = np.ones(images.shape[0]).reshape(-1, 1)
                mask = np.concatenate((temp, ts_blank), axis=0)

                self.data_test.append(images_all) # 70 * 109 * 89
                self.y_test.append(np.array([gene] * num_slice).squeeze()) # 70
                self.t_test.append(ts_all) # 70
                self.mask_test.append(mask) # 70
                self.num_non_blank_test.append(images.shape[0])
            
            self.data_test = np.array(self.data_test)
            self.y_test = np.array(self.y_test)
            self.t_test = np.array(self.t_test)
            self.mask_test = np.array(self.mask_test)
            self.num_non_blank_test = np.array(self.num_non_blank_test)

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
                return self.data_train[index], self.y_train[index], self.t_train[index], self.mask_train[index], self.num_non_blank_train[index]
            return self.data_train[index], self.y_train[index], self.t_train[index]
        if (self.mode == 'test'):
            if(self.squeeze):
                return self.data_test[index], self.y_test[index], self.t_test[index], self.mask_test[index], self.num_non_blank_test[index]
            return self.data_test[index], self.y_test[index], self.t_test[index]

    def __len__(self):
        if (self.mode == 'train'):
            return self.data_train.shape[0]
        if (self.mode == 'test'):
            return self.data_test.shape[0]

    def get_picture_num(self):
        if (self.mode == 'train'):
            return self.picture_num_train
        if (self.mode == 'test'):
            return self.picture_num_test

class Dataset_Transfer(data.Dataset):
    def __init__(self, mode='train', transform=None, name_id=0, seed=0, proportion=0.1, squeeze=True, num_slice=70):
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

        self.data_source = []
        self.y_source = []
        self.t_source = []
        self.mask_source = []
        self.num_non_blank_source = []

        self.data_target = []
        self.y_target = []
        self.t_target = []
        self.mask_target = []
        self.num_non_blank_target = []
        
        # 加载图片数据
        feature = sparse.load_npz(dir_image)
        feature = feature.toarray().reshape(-1, shape[2], shape[3])
        # 加载标签数据
        label = np.load(dir_label)

        # 数据集全部非空白图片数量
        dataset_size = feature.shape[0]
        max = np.max(feature)
        min = np.min(feature)

         # 划分训练集和测试集
        index_all = [i for i in range(dataset_size)]
        index_test = np.array(random.sample(index_all, int(proportion*dataset_size)))
        index_train = np.array(list(set(index_all) - set(index_test)))

        self.picture_num_train = index_train.shape[0]
        self.feature_train = feature[index_train]
        self.label_train = label[index_train]

        if(squeeze):
            self.feature_train = (self.feature_train - min) / (max - min)
            # self.max_list_train = np.max(self.feature_train, axis=1).reshape(-1, 1)
            # self.feature_train = self.feature_train / self.max_list_train
            # self.feature_train = self.feature_train.reshape(-1, 1, shape[2], shape[3])
            
        
        label_gene_train = self.label_train[:, 0]
        label_slice_train = self.label_train[:, 1]
            
        for gene in range(shape[0]):
            index_temp = np.array(np.where(label_gene_train == gene)).flatten()
            # print(index_temp.shape)
            if(index_temp.shape[0] <= 3):
                continue
            index_target = np.array(random.sample(list(index_temp), int(0.2*index_temp.shape[0])))
            index_source = np.array(list(set(index_temp) - set(index_target)))

            # print(index_target.shape[0])
            # print(index_source.shape[0])

            images = self.feature_train[index_source].reshape(-1, shape[2], shape[3])
            images_blank = np.zeros((num_slice-images.shape[0], shape[2]*shape[3])).reshape(-1, shape[2], shape[3])
            images_all = np.concatenate((images, images_blank), axis=0)

            ts = label_slice_train[index_source].reshape(-1, 1)
            ts_blank = np.zeros(num_slice-images.shape[0]).reshape(-1, 1)
            ts_all = np.concatenate((ts, ts_blank), axis=0)
            
            temp = np.ones(images.shape[0]).reshape(-1, 1)
            mask = np.concatenate((temp, ts_blank), axis=0)

            self.data_source.append(images_all) # 70 * 109 * 89
            self.y_source.append(np.array([gene] * num_slice).squeeze()) # 70
            self.t_source.append(ts_all) # 70
            self.mask_source.append(mask) # 70
            self.num_non_blank_source.append(images.shape[0])

            images = self.feature_train[index_target].reshape(-1, shape[2], shape[3])
            images_blank = np.zeros((num_slice-images.shape[0], shape[2]*shape[3])).reshape(-1, shape[2], shape[3])
            images_all = np.concatenate((images, images_blank), axis=0)

            ts = label_slice_train[index_target].reshape(-1, 1)
            ts_blank = np.zeros(num_slice-images.shape[0]).reshape(-1, 1)
            ts_all = np.concatenate((ts, ts_blank), axis=0)
            
            temp = np.ones(images.shape[0]).reshape(-1, 1)
            mask = np.concatenate((temp, ts_blank), axis=0)

            self.data_target.append(images_all) # 70 * 109 * 89
            self.y_target.append(np.array([gene] * num_slice).squeeze()) # 70
            self.t_target.append(ts_all) # 70
            self.mask_target.append(mask) # 70
            self.num_non_blank_target.append(images.shape[0])
    
        self.data_source = np.array(self.data_source)
        self.y_source = np.array(self.y_source)
        self.t_source = np.array(self.t_source)
        self.mask_source = np.array(self.mask_source)
        self.num_non_blank_source = np.array(self.num_non_blank_source)

        self.data_target = np.array(self.data_target)
        self.y_target = np.array(self.y_target)
        self.t_target = np.array(self.t_target)
        self.mask_target = np.array(self.mask_target)
        self.num_non_blank_target = np.array(self.num_non_blank_target)

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
        return self.data_target[index], self.y_target[index], self.t_target[index], self.mask_target[index], self.num_non_blank_target[index], self.data_source[index], self.y_source[index], self.t_source[index], self.mask_source[index], self.num_non_blank_source[index]

    def __len__(self):
        return self.data_source.shape[0]

    def get_picture_num(self):
        return self.num_non_blank_target.sum()

if __name__ == '__main__':
    for seed in range(3):
        print(seed, end = '')
        print('++++++++++++++++++++++++++++')
        for name_id in range(7):
            T = Brain_Dataset(mode='test', name_id=name_id, seed=seed, squeeze=True)
            gene_train = set(T.label_train[:, 0])
            slice_train = set(T.label_train[:, 1])
            gene_test = set(T.label_test[:, 0])
            slice_test = set(T.label_test[:, 1])
            print(name_id, end=': ')
            print(gene_test.issubset(gene_train), end=' ')
            print(slice_test.issubset(slice_train))
            print([np.min(T.feature_train), np.max(T.feature_train)])
            print([np.min(T.feature_test), np.max(T.feature_test)])
    
    #  dataset_test = Brain_Dataset(mode='test', transform=None, name_id=1, seed=0, proportion=0.1, squeeze=True)
    #  test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=3, shuffle=False)
    #  for t, (images, labels, ts, mask, num_non_blank) in enumerate(test_loader):
    #     print(images.shape)
    #     print(labels.shape)
    #     print(ts.shape)
    #     print(mask.shape)
    #     print(num_non_blank.shape)
    #     break


