from scipy import sparse
import numpy as np
import random
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]
name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
dir_0 = './Data_raw/'
dir_1 = '_ish_data_density_csc.npz'
dir_2 = './Data_non_blank/image_'
dir_3 = './Data_non_blank/label_'
for id in range(len(name_list)):
    name = name_list[id]
    shape = shape_list[id]
    ish_arr = sparse.load_npz(dir_0 + name + dir_1)
    data = ish_arr.toarray().reshape(shape)
    num_data = shape[0] * shape[1]
    num_non_blank = 0
    image_non_blank = []
    label_non_blank = []
    # 取出所有非空白图片以及对应的标签
    # num_last = 0
    for gene in range(shape[0]):
        # num_last = num_non_blank
        for slice in range(shape[1]):
            if(np.max(data[gene][slice]) != 0):
                num_non_blank = num_non_blank + 1
                image_non_blank.append(data[gene][slice])
                label_non_blank.append([gene, slice])
        # print(num_non_blank - num_last, end=' ')
    
    print(name+'包含{}张图片，其中空白图片有{}张，非空白图片有{}张'.format(num_data, num_data-num_non_blank, num_non_blank))
    
    image_non_blank = np.array(image_non_blank).reshape(num_non_blank, shape[2]*shape[3])
    label_non_blank = np.array(label_non_blank)
    
    # break
    # 将图片数据转化为稀疏矩阵进行存储
    image_non_blank = sparse.csr_matrix(image_non_blank)
    sparse.save_npz(dir_2+name+'.npz', image_non_blank)
    np.save(dir_3+name+'.npy', label_non_blank)
