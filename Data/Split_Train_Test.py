from torch.utils import data
import numpy as np
from scipy import sparse
import random
from tqdm import tqdm
import os

name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]
dir_img = './Data_non_blank/image_'
dir_label = './Data_non_blank/label_'
dir_index = './Index_test/' 


for name_id in tqdm(range(len(name_list))):
    name = name_list[name_id]
    shape = shape_list[name_id]
    image = sparse.load_npz(dir_img+name+'.npz')
    image = image.toarray()
    label = np.load(dir_label+name+'.npy')
    label_0 = label[:, 0]

    remain = shape[0] % 10
    interval = (shape[0]-remain) / 10

    dir = dir_index + name
    if not os.path.exists(dir):
        os.mkdir(dir)
    for id in tqdm(range(10)):
        index_1 = np.array(np.where((label_0 >= id * interval) & (label_0 < (id + 1) * interval + remain))).ravel()
        index_9 = np.array(np.where((label_0 < id * interval) | (label_0 >= (id + 1) * interval + remain))).ravel()

        temp = [i for i in range(index_1.shape[0])]
        test = random.sample(range(index_1.shape[0]), int(index_1.shape[0] * 0.5))
        train = list(set(temp) - set(test))

        index_test = index_1[test]
        # index_train = np.concatenate((index_9, index_1[train]))
        
        np.save(dir+'/'+str(id)+'.npy', index_test)
    
        # # 验证代码逻辑正确性
        # label_test_0 = label[index_test][:, 0]
        # set_label = set(label_test_0)
        # print(len(set_label))




