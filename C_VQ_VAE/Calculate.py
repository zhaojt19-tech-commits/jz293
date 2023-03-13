import numpy as np

if __name__ == '__main__':
    conv_dims = [32, 64]
    image_H = 109
    image_W = 89
    num_convs = len(conv_dims)
    kernel_size = 3
    stride = 2
    padding = 1
    # 计算编码过程中的图像尺寸变化
    temp_H = image_H
    temp_W = image_W
    for i in range(num_convs):
        temp_H = int((temp_H + 2 * padding - kernel_size) / stride) + 1
        temp_W = int((temp_W + 2 * padding - kernel_size) / stride) + 1

    image_H_encode = temp_H
    image_W_encode = temp_W
    print("编码器编码以后的图像尺寸：")
    print('H={}'.format(image_H_encode))
    print('W={}'.format(image_W_encode))

    # 计算解码过程中的图像尺寸变化
    output_padding = 1

    temp_H = image_H_encode
    temp_W = image_W_encode
    for i in range(num_convs):
        temp_H = stride * (temp_H - 1) - 2 * padding + kernel_size + output_padding
        temp_W = stride * (temp_W - 1) - 2 * padding + kernel_size + output_padding
    
    stride = 1
    output_padding = 0
    kernel_size = 2
    padding = 2
    temp_H = stride * (temp_H - 1) - 2 * padding + kernel_size + output_padding
    temp_W = stride * (temp_W - 1) - 2 * padding + kernel_size + output_padding


    print("解码器倒数第二层的图像尺寸：")
    print('H={}'.format(temp_H))
    print('W={}'.format(temp_W))