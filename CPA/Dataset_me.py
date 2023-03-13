from torch.utils import data
import numpy as np
import torch
from scipy import sparse
import random
from torch.utils import data
import numpy as np
import torch
from scipy import sparse
import random
from torchvision.transforms import Normalize

name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]

class Brain_Dataset(data.Dataset):
    def __init__(self, mode='train', transform=None, name_id=0, id=0, squeeze=False):
        self.transform = transform
        self.mode = mode
        self.squeeze = squeeze
         # 读取的原始数据
        self.feature = []
        self.label = []

        name = name_list[name_id]
        shape = shape_list[name_id]

        dir_image = '../Data/Data_non_blank/image_' + name + '.npz'
        dir_label = '../Data/Data_non_blank/label_' + name + '.npy'
        dir_index = '../Data/Index_test/' + name + '/' + str(id) + '.npy'

        # 读取全部非空白图片
        feature = sparse.load_npz(dir_image)
        feature = feature.toarray().reshape((-1, shape[2], shape[3]))
        # 读取全部非空白图片的标签
        label = np.load(dir_label)
        # 读取第k折对应的训练集下标
        index_test = np.load(dir_index)
        temp_list = [i for i in range(label.shape[0])]
        index_train = np.array(list(set(temp_list) - set(index_test)))

        self.picture_num_train = index_train.shape[0]
        self.feature_train = feature[index_train]
        self.label_train = label[index_train]

        if (mode == 'test'):
            # 读取第k折对应的测试集下标
            self.picture_num_test = index_test.shape[0]
            self.feature_test = feature[index_test]
            self.label_test = label[index_test]

        # 对数据进行压缩处理(按slice)
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
    T = Brain_Dataset('test', name_id=0, id = 1, squeeze=True)
    train_loader = torch.utils.data.DataLoader(T, batch_size=32, shuffle=False)
    for data, label, max in train_loader:
        print(data.shape)
        print(label.shape)
        print(max.shape)
        break

