from scipy import sparse
import numpy as np
import random
from tqdm import tqdm

shape_list_full = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94)]
shape_list_brain = [(2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]
name_list_full = ['E11.5', 'E13.5', 'E15.5']
name_list_brain = ['E18.5', 'P4', 'P14', 'P56']
name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
dir_0 = './Data_id/'
dir_2 = './Data_non_blank_T/image_'
dir_3 = './Data_non_blank_T/label_'
dir_4 = './Data_time/gene_id_full.npy'
dir_5 = './Data_time/gene_id_brain.npy'

gene_id_list = []
for id in range(0, 3, 1):
    name = name_list[id]
    print(name)
    gene_id = np.load(dir_0 + name + '_id.npy')
    gene_id_list.append(gene_id)

temp_set = set(gene_id_list[0])
for id in range(0, 2, 1):
    temp_set = temp_set.intersection(set(gene_id_list[id+1]))

temp_set = np.array(list(temp_set))
print(temp_set.shape[0])
np.save(dir_4, temp_set)


