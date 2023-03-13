from torch.utils import data
import numpy as np
import torch
from scipy import sparse
import random


if __name__ == '__main__':
    id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for id in id_list:
        feature = sparse.load_npz('../CVAE/Sample/feature_non_blank.npz')
        feature = feature.toarray().reshape((-1, 1, 109, 89))
        label = np.load('../CVAE/Sample/label_non_blank.npy')
        label_0 = label[:, 0]
        index_201 = np.array(np.where((label_0 >= id * 200) & (label_0 <= (id + 1) * 200))).ravel()
        index_1800 = np.array(np.where((label_0 < id * 200) | (label_0 > (id + 1) * 200))).ravel()

        temp = [i for i in range(index_201.shape[0])]
        test = random.sample(range(index_201.shape[0]), int(index_201.shape[0] * 0.5))
        train = list(set(temp) - set(test))

        index_test = index_201[test]
        index_train = np.concatenate((index_1800, index_201[train]))
        
        # 验证代码逻辑正确性
        label_test_0 = label[index_test][:, 0]
        set_label = set(label_test_0)
        print(len(set_label))

        np.save('./Data/index_test_' + str(id), index_test)
        np.save('./Data/index_train_' + str(id), index_train)