from scipy import sparse
import numpy as np
import random
from skimage import transform
from tqdm import tqdm
import matplotlib.pyplot as plt

shape_list_full = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94)]
shape_list_brain = [(2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]
shape_full_non_blank_list = [59943, 60119, 75295]
shape_brain_non_blank_list = [39306, 46081, 42116, 62799]
name_list_full = ['E11.5', 'E13.5', 'E15.5']
name_list_brain = ['E18.5', 'P4', 'P14', 'P56']
time_list_full = [11.5, 13.5, 15.5]
time_list_brain = [18.5, 24, 34, 76]
name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
dir_0 = './Data_raw/'
dir_1 = '_ish_data_density_csc.npz'
dir_2 = './Data_time/image_'
dir_3 = './Data_time/label_'
dir_4 = './Data_time/gene_id_full.npy'
dir_5 = './Data_time/gene_id_brain.npy'
dir_6 = './Data_id/'

gene_id_full = np.load(dir_4)
gene_id_brain = np.load(dir_5)

image_non_blank_full = []
label_non_blank_full = []
for id in range(len(name_list_full)):
    name = name_list_full[id]
    shape = shape_list_full[id]
    gene_id = np.load(dir_6 + name + '_id.npy')

    ish_arr = sparse.load_npz(dir_0 + name + dir_1)
    data = ish_arr.toarray().reshape(shape)
    data = transform.resize(data, (shape[0], shape[1], 75, 70))
    
    num_data = shape[0] * shape[1]
    num_non_blank = 0
    num_gene = 0
    image_non_blank = []
    label_non_blank = []

    # 取出所有非空白图片以及对应的标签
    gene = 0
    for i in tqdm(range(shape[0])):
        if(gene_id[i] in gene_id_full):
            num_gene = num_gene + 1
        else:
            continue
        for slice in range(shape[1]):
            if(np.max(data[i][slice]) != 0):
                num_non_blank = num_non_blank + 1
                image_non_blank.append(data[i][slice])
                label_non_blank.append([time_list_full[id], gene, slice])
        gene = gene + 1
    
    print(name+'提取出{}个基因, {}张图片'.format(gene, num_non_blank))
    
    image_non_blank = np.array(image_non_blank).reshape(num_non_blank, 75*70)
    label_non_blank = np.array(label_non_blank)

    image_non_blank_full.append(image_non_blank)
    label_non_blank_full.append(label_non_blank)

image_non_blank_full = np.concatenate(image_non_blank_full)
label_non_blank_full = np.concatenate(label_non_blank_full)
print(image_non_blank_full.shape)
print(label_non_blank_full.shape)

# 将图片数据转化为稀疏矩阵进行存储
image_non_blank_full = sparse.csr_matrix(image_non_blank_full)
sparse.save_npz(dir_2+'full.npz', image_non_blank_full)
np.save(dir_3+'full.npy', label_non_blank_full)

image_non_blank_brain = []
label_non_blank_brain = []
for id in range(len(name_list_brain)):
    name = name_list_brain[id]
    shape = shape_list_brain[id]
    gene_id = np.load(dir_6 + name + '_id.npy')

    ish_arr = sparse.load_npz(dir_0 + name + dir_1)
    data = ish_arr.toarray().reshape(shape)
    data = transform.resize(data, (shape[0], shape[1], 40, 68))
    
    num_data = shape[0] * shape[1]
    num_non_blank = 0
    num_gene = 0
    image_non_blank = []
    label_non_blank = []
    # 取出所有非空白图片以及对应的标签
    gene = 0
    for i in tqdm(range(shape[0])):
        if(gene_id[i] in gene_id_brain):
            num_gene = num_gene + 1
        else:
            continue
        for slice in range(shape[1]):
            if(np.max(data[i][slice]) != 0):
                num_non_blank = num_non_blank + 1
                image_non_blank.append(data[i][slice])
                label_non_blank.append([time_list_brain[id], gene, slice])
        gene = gene + 1
    
    print(name+'提取出{}个基因, {}张图片'.format(gene, num_non_blank))
    
    image_non_blank = np.array(image_non_blank).reshape(num_non_blank, 40*68)
    label_non_blank = np.array(label_non_blank)

    image_non_blank_brain.append(image_non_blank)
    label_non_blank_brain.append(label_non_blank)

image_non_blank_brain = np.concatenate(image_non_blank_brain)
label_non_blank_brain = np.concatenate(label_non_blank_brain)
print(image_non_blank_brain.shape)
print(label_non_blank_brain.shape)

# 将图片数据转化为稀疏矩阵进行存储
image_non_blank_brain = sparse.csr_matrix(image_non_blank_brain)
sparse.save_npz(dir_2+'brain.npz', image_non_blank_brain)
np.save(dir_3+'brain.npy', label_non_blank_brain)

    

