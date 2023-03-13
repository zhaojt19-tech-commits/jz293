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
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import argparse
from utils import str2bool, save_samples, get_loaders
from tqdm import tqdm
from GatedPixelCNN_Model import PixelCNN
import time


if __name__ == '__main__':
    # 配置VQ-VAE部分
    model_name = 'model_2022_05_27_01_11_15_Brain_auto_32_8_0'
    path_model = './Checkpoint/'
    batch_size = 128
    embedding_dim = 16
    num_embeddings = 128
    id = 0
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_train = Brain_Dataset(mode='train', id=id)
    # dataset_test = Brain_Dataset(mode='test', id=id)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    model_VAE = VQVAE(in_dim=1, embedding_dim=embedding_dim, num_embeddings=num_embeddings)
    model_VAE.load_state_dict(torch.load(path_model + model_name))
    model_VAE = model_VAE.cuda()
    model_VAE.eval()
    
    # 配置Gated_PixelCNN部分
    # 设置输入参数
    parser = argparse.ArgumentParser(description='PixelCNN')

    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of epochs to train model for')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Number of images per mini-batch')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Dataset to train model on. Either mnist, fashionmnist or cifar.')

    parser.add_argument('--causal-ksize', type=int, default=7,
                        help='Kernel size of causal convolution')
    parser.add_argument('--hidden-ksize', type=int, default=7,
                        help='Kernel size of hidden layers convolutions')

    parser.add_argument('--color-levels', type=int, default=128,
                        help='Number of levels to quantisize value of each channel of each pixel into')

    parser.add_argument('--hidden-fmaps', type=int, default=30,
                        help='Number of feature maps in hidden layer (must be divisible by 3)')
    parser.add_argument('--out-hidden-fmaps', type=int, default=10,
                        help='Number of feature maps in outer hidden layer')
    parser.add_argument('--hidden-layers', type=int, default=6,
                        help='Number of layers of gated convolutions with mask of type "B"')

    parser.add_argument('--learning-rate', '--lr', type=float, default=0.0001,
                        help='Learning rate of optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay rate of optimizer')
    parser.add_argument('--max-norm', type=float, default=1.,
                        help='Max norm of the gradients after clipping')

    parser.add_argument('--epoch-samples', type=int, default=25,
                        help='Number of images to sample each epoch')

    parser.add_argument('--cuda', type=str2bool, default=True,
                        help='Flag indicating whether CUDA should be used')

    cfg = parser.parse_args()
    
    torch.manual_seed(42)
    
    model_PixelCNN = PixelCNN(cfg=cfg)
    model_PixelCNN_name = 'model_2022_06_26_07_23_51_Brain_128_20_0'
    path_model_PixelCNN = './Checkpoint_PixelCNN/'
    model_PixelCNN.load_state_dict(torch.load(path_model_PixelCNN + model_PixelCNN_name))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    model_PixelCNN.to(device)

    optimizer = optim.Adam(model_PixelCNN.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, cfg.learning_rate, 10*cfg.learning_rate, cycle_momentum=False)
    # 设置训练轮次
    epochs = cfg.epochs
    # 利用VQ-VAE生成压缩后的图片
    encoding_indices_list = []
    with torch.no_grad():
        for t, (images, labels) in enumerate(tqdm(train_loader)):
            input_size = labels.shape[0]
            images = images.cuda()
            encoding_indices, x_recon = model_VAE(images)
            encoding_indices = encoding_indices.unsqueeze(1)
            # print(encoding_indices.size())
            encoding_indices_list.append(encoding_indices)
    
    # encoding_indices_list = torch.tensor(encoding_indices_list).cuda()
    print('VQ-VAE部分结束')
    print_freq = 200
    # 开始训练PixelCNN
    for e in range(epochs):
        for t, (images, labels) in enumerate(tqdm(train_loader)):
            input_size = labels.shape[0]
            encoding_indices = encoding_indices_list[t].reshape(input_size, 28, 23, -1)
            encoding_indices = encoding_indices.permute(0, 3, 1, 2).contiguous().float()

            labels = labels.cuda()
            optimizer.zero_grad()

            outputs = model_PixelCNN(encoding_indices, labels)
            encoding_indices = encoding_indices.long()
            loss = F.cross_entropy(outputs, encoding_indices)
            
            loss.backward()
            # if (t + 1) % print_freq == 0 or (t + 1) == len(encoding_indices_list):
            #     print("\t [{}/{}]: loss {}".format(t+1, len(encoding_indices_list), loss.item()))
            # clip_grad_norm_(model_PixelCNN.parameters(), max_norm=cfg.max_norm)
            optimizer.step()
        print('After {} epoch, the final loss is {}'.format(e+1, loss))
        scheduler.step()
    
    path_model = './Checkpoint_PixelCNN/'
    name_dataset = '_Brain'
    batch_str = '_' + str(batch_size)
    epoch_str = '_' + str(epochs + 20)
    id_str = '_' + str(id)
    name_model = 'model_' + time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + name_dataset + batch_str + epoch_str + id_str
    torch.save(model_PixelCNN.state_dict(), path_model + name_model)
    print(name_model)
       