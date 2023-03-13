from scipy import sparse
import numpy as np
import random
from skimage import transform
from tqdm import tqdm
import matplotlib.pyplot as plt

shape_list_full = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94)]
shape_list_brain = [(2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]
name_list_full = ['E11.5', 'E13.5', 'E15.5']
name_list_brain = ['E18.5', 'P4', 'P14', 'P56']
dir_0 = './Data_raw/'
dir_1 = '_ish_data_density_csc.npz'
dir_2 = './Data_non_blank_T/image_'
dir_3 = './Data_non_blank_T/label_'

# for id in range(len(name_list_full)):
#     name = name_list_full[id]
#     shape = shape_list_full[id]
#     ish_arr = sparse.load_npz(dir_0 + name + dir_1)
#     data = ish_arr.toarray().reshape(shape)
#     data = transform.resize(data, (shape[0], shape[1], 75, 70))
#     # data_temp = []
#     # for gene in range(shape[0]):
#     #     data_temp.append(transform.resize(data[gene], (shape[1], 75, 70)))
#     # data = np.array(data_temp)
#     num_data = shape[0] * shape[1]
#     num_non_blank = 0
#     image_non_blank = []
#     label_non_blank = []
#     # 取出所有非空白图片以及对应的标签
#     for gene in tqdm(range(shape[0])):
#         for slice in range(shape[1]):
#             if(np.max(data[gene][slice]) != 0):
#                 num_non_blank = num_non_blank + 1
#                 image_non_blank.append(data[gene][slice])
#                 label_non_blank.append([gene, slice])
    
#     print(name+'包含{}张图片，其中空白图片有{}张，非空白图片有{}张'.format(num_data, num_data-num_non_blank, num_non_blank))
    
#     image_non_blank = np.array(image_non_blank).reshape(num_non_blank, 75*70)
#     label_non_blank = np.array(label_non_blank)

#     # 将图片数据转化为稀疏矩阵进行存储
#     image_non_blank = sparse.csr_matrix(image_non_blank)
#     sparse.save_npz(dir_2+name+'.npz', image_non_blank)
#     np.save(dir_3+name+'.npy', label_non_blank)


# for id in range(len(name_list_brain)):
#     name = name_list_brain[id]
#     shape = shape_list_brain[id]
#     ish_arr = sparse.load_npz(dir_0 + name + dir_1)
#     data = ish_arr.toarray().reshape(shape)
#     data = transform.resize(data, (shape[0], shape[1], 40, 68))
#     # data_temp = []
#     # for gene in range(shape[0]):
#     #     data_temp.append(transform.resize(data[gene], (shape[1], 40, 68)))
#     # data = np.array(data_temp)
#     num_data = shape[0] * shape[1]
#     num_non_blank = 0
#     image_non_blank = []
#     label_non_blank = []
#     # 取出所有非空白图片以及对应的标签
#     for gene in tqdm(range(shape[0])):
#         for slice in range(shape[1]):
#             if(np.max(data[gene][slice]) != 0):
#                 num_non_blank = num_non_blank + 1
#                 image_non_blank.append(data[gene][slice])
#                 label_non_blank.append([gene, slice])
    
#     print(name+'包含{}张图片，其中空白图片有{}张，非空白图片有{}张'.format(num_data, num_data-num_non_blank, num_non_blank))
    
#     image_non_blank = np.array(image_non_blank).reshape(num_non_blank, 40*68)
#     label_non_blank = np.array(label_non_blank)

#     # 将图片数据转化为稀疏矩阵进行存储
#     image_non_blank = sparse.csr_matrix(image_non_blank)
#     sparse.save_npz(dir_2+name+'.npz', image_non_blank)
#     np.save(dir_3+name+'.npy', label_non_blank)

if __name__ == '__main__':
    shape_non_blank_list = [59943, 60119, 75295, 39306, 46081, 42116, 62799]
    
    # for id in range(len(name_list_full)):
    #     id = id + 2
    #     name = name_list_full[id]
    #     shape = shape_list_full[id]
    #     print(name)
    #     dir_image = './Data_non_blank_T/image_' + name + '.npz'
    #     dir_label = './Data_non_blank_T/label_' + name + '.npy'

    #     ish_arr = sparse.load_npz(dir_image)
    #     data = ish_arr.toarray().reshape(shape_non_blank_list[id], 75, 70)
    #     # label = np.load(dir_label).reshape(shape_non_blank_list[id], 2)
    #     # print(data.shape, end=" ")
    #     # print(label.shape)

    #     for i in range(0, 300, 10):
    #         plt.imsave(str(i)+'.png', data[i])
    #     break

    # for id in range(len(name_list_brain)):
    #     name = name_list_brain[id]
    #     shape = shape_list_brain[id]
    #     print(name)

    #     dir_image = './Data_non_blank_T/image_' + name + '.npz'
    #     dir_label = './Data_non_blank_T/label_' + name + '.npy'

    #     ish_arr = sparse.load_npz(dir_image)
    #     data = ish_arr.toarray().reshape(shape_non_blank_list[id+3], 40, 68)
    #     # label = np.load(dir_label).reshape(shape_non_blank_list[id+3], 2)

    #     # print(data.shape, end=" ")
    #     # print(label.shape)

    #     for i in range(0, 300, 10):
    #         plt.imsave(str(i)+'.png', data[i])
    #     break

