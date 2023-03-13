import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
if __name__ == '__main__':
    H_in = 109
    W_in = 89
    # 卷积参数
    kernel_size_in = [(5, 5), (5, 5), (5, 5)]
    stride_in = [(2, 2), (2, 2), (2, 2)]
    padding_in = [(2, 2), (2, 2), (2, 2)]
    # 反卷积参数
    kernel_size_out = [(3, 7), (3, 7), (3, 6)]
    stride_out = [(2, 1), (2, 2), (2, 2)]
    padding_out = [(1, 0), (1, 0), (1, 0)]
    output_padding = [(1, 0), (0, 1), (0, 1)]

    H_temp = H_in
    W_temp = W_in
    for i in range(len(kernel_size_in)):
        H_temp = int((H_temp + 2*padding_in[i][0]-kernel_size_in[i][0])/stride_in[i][0]) + 1
        W_temp = int((W_temp + 2 * padding_in[i][1] - kernel_size_in[i][1]) / stride_in[i][1]) + 1
    print('经过卷积以后，图片尺寸为：')
    print([H_temp, W_temp])

    for i in range(len(kernel_size_out)):
        H_temp = (H_temp-1)*stride_out[i][0]-2*padding_out[i][0]+kernel_size_out[i][0]+output_padding[i][0]
        W_temp = (W_temp - 1) * stride_out[i][1] - 2 * padding_out[i][1] + kernel_size_out[i][1] + output_padding[i][1]
    print('经过反卷积以后，图片尺寸为：')
    print([H_temp, W_temp])