import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from VQ_VAE_Module import *
from MyDataset_10K import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # TODO
    model_name = 'model_2022_05_27_01_11_15_Brain_auto_32_8_0'
    path_model = './Checkpoint/'
    batch_size = 32
    embedding_dim = 16
    num_embeddings = 128
    id = 0
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_train = Brain_Dataset(mode='train', id=id)
    dataset_test = Brain_Dataset(mode='test', id=id)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    model = VQVAE(in_dim=1, embedding_dim=embedding_dim, num_embeddings=num_embeddings)
    model.load_state_dict(torch.load(path_model + model_name))
    model = model.cuda()
    model.eval()
    sample_freq = 200    # 抽取重建图片的频率
    # 生成重构图片文件夹
    path = "./Pictures_recons"
    if not os.path.exists(path):
        os.mkdir(path)
    path = "./Pictures_recons/auto"
    if not os.path.exists(path):
        os.mkdir(path)
    path = "./Pictures_recons/manual"
    if not os.path.exists(path):
        os.mkdir(path)
    path_this = "./Pictures_recons/auto"
    # 重构测试
    recon_loss_sum = 0
    with torch.no_grad():
        for t, (images, labels) in enumerate(test_loader):
            input_size = labels.shape[0]
            images = images.cuda()
            encoding_indices, x_recon = model(images)
            # 计算MSE损失
            loss_fn = torch.nn.MSELoss(reduction = 'sum').cuda()
            recon_loss = loss_fn(x_recon, images).cpu()
            recon_loss_sum = recon_loss_sum + recon_loss
            if(t % 50 == 0):
                print(encoding_indices.cpu())
                recon = x_recon.cpu().numpy()
                images = images.cpu().numpy()
                for j in range(input_size):
                    plt.imsave(path_this+'/'+str(t*50+j)+'_'+str(1)+'.png', recon[j].squeeze(0))
                    plt.imsave(path_this+'/'+str(t*50+j)+'_'+str(0)+'.png', images[j].squeeze(0))
    recon_loss_average = recon_loss_sum / ((len(test_loader)-1)*batch_size+input_size)
    recon_loss_average = torch.sqrt(recon_loss_average)
    print('测试集上平均每张图片重构损失为{}'.format(recon_loss_average))
