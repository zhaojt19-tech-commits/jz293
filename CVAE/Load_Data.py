from scipy import sparse
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ish_arr = sparse.load_npz('./Sample/e13_ish_data_csc.npz')
    gid = 0  # between 0 and 2001
    slice_id = 31  # between 0 and 69
    print(ish_arr.shape)
    arr2 = ish_arr[gid, :].toarray().reshape((69, 109, 89)) # 取出基因0对应的所有切片图(69)
    img2 = arr2[slice_id, :, :]
    plt.imsave('test0.png', img2)  # 基因0对应的第slice_id号切片





