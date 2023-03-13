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
from GatedPixelCNN_Onehot_Model import PixelCNN
import time


if __name__ == '__main__':
    
    # 设置输入参数
    parser = argparse.ArgumentParser(description='PixelCNN_VAE')
    parser.add_argument('--is_continue', type=bool, default=False,
                        help='weather train from the old model')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train model for')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of images per mini-batch')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Dataset to train model on. Either mnist, fashionmnist or cifar.')
    parser.add_argument('--id', type=int, default=0,
                        help='id of the K-fold for test')
    parser.add_argument('--model_VAE_name', type=str, default='model_2022_05_27_01_11_15_Brain_auto_32_8_0',
                        help='name of the model of VAE')
    parser.add_argument('--model_PixelCNN_name', type=str, default='model_2022_06_28_21_17_10.t7',
                        help='name of the model of PixelCNN')
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

    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-4,
                        help='Learning rate of optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay rate of optimizer')
    parser.add_argument('--max-norm', type=float, default=1.,
                        help='Max norm of the gradients after clipping')

    parser.add_argument('--epoch-samples', type=int, default=25,
                        help='Number of images to sample each epoch')

    parser.add_argument('--cuda', type=str2bool, default=True,
                        help='Flag indicating whether CUDA should be used')
    
    parser.add_argument('--num_classes', type=list, default=[2001, 69],
                        help='the class number of each label')
    parser.add_argument('--H_latent', type=int, default=28,
                        help='the height of the latent map')                    
    parser.add_argument('--W_latent', type=int, default=23,
                        help='the width of the latent map')
    
    parser.add_argument('--embedding_dim', type=int, default=16,
                        help='the dimensions of the vector table')
    parser.add_argument('--num_embeddings', type=int, default=128,
                        help='the number of vectors in the vector table')

    parser.add_argument('--path_model_VAE', type=str, default='./Checkpoint/',
                        help='the folder of the checkpoint of VAE')
    parser.add_argument('--path_model_PixelCNN', type=str, default='./Checkpoint_PixelCNN/',
                        help='the folder of the checkpoint of PixelCNN')
    
    parser.add_argument('--name_dataset', type=str, default='Brain',
                        help='the name of dataset for model training')
    
    cfg = parser.parse_args()

    torch.manual_seed(42)

    # 配置VQ-VAE部分
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_train = Brain_Dataset(mode='train', id=cfg.id)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=False)
    model_VAE = VQVAE(in_dim=1, embedding_dim=cfg.embedding_dim, num_embeddings=cfg.num_embeddings)
    model_VAE.load_state_dict(torch.load(cfg.path_model_VAE + cfg.model_VAE_name))
    model_VAE = model_VAE.cuda()
    model_VAE.eval()

    
    model_PixelCNN = PixelCNN(cfg, cfg.num_classes, cfg.H_latent, cfg.W_latent)
    epochs_old = 0
    if(cfg.is_continue == True):
        checkpoint = torch.load(cfg.path_model_PixelCNN + cfg.model_PixelCNN_name)
        model_PixelCNN.load_state_dict(checkpoint['model'])
        epochs_old = checkpoint['epochs']
    else:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    model_PixelCNN.to(device)
    model_PixelCNN.train()

    optimizer = optim.Adam(model_PixelCNN.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, cfg.learning_rate, 10*cfg.learning_rate, cycle_momentum=False)

    # 利用VQ-VAE生成压缩后的图片
    encoding_indices_list = []
    with torch.no_grad():
        for t, (images, labels) in enumerate(tqdm(train_loader)):
            input_size = labels.shape[0]
            images = images.cuda()
            encoding_indices, x_recon = model_VAE(images)
            encoding_indices = encoding_indices.unsqueeze(1)
            encoding_indices_list.append(encoding_indices)
    
    print('VQ-VAE部分结束')
    loss_sum = 0        # 所有训练集图片总损失
    # 开始训练PixelCNN
    for e in range(cfg.epochs):
        for t, (images, labels) in enumerate(tqdm(train_loader)):
            input_size = labels.shape[0]
            encoding_indices = encoding_indices_list[t].reshape(input_size, 28, 23, -1)
            encoding_indices = encoding_indices.permute(0, 3, 1, 2).contiguous().float()
            label_double = labels.reshape(input_size, -1)
            label0 = label_double[:, 0].long()
            label1 = label_double[:, 1].long()
            index = np.linspace(0, input_size - 1, input_size).astype("long")
            one_hot_0 = torch.zeros(input_size, cfg.num_classes[0])
            one_hot_1 = torch.zeros(input_size, cfg.num_classes[1])
            one_hot_0[[index, label0]] = 1
            one_hot_1[[index, label1]] = 1
            one_hot_0 = one_hot_0.cuda()
            one_hot_1 = one_hot_1.cuda()
            optimizer.zero_grad()
            outputs = model_PixelCNN(encoding_indices, one_hot_0, one_hot_1)
            encoding_indices = encoding_indices.long()
            loss = F.cross_entropy(outputs, encoding_indices)
            loss_sum = loss_sum + loss.item()
            loss.backward()
            optimizer.step()
        loss_averate = loss_sum / ((len(train_loader)-1)*cfg.batch_size+input_size)
        print('After {} epoch, the average loss is {}'.format(e+1, loss))
        # scheduler.step()
    if(cfg.is_continue == True):
        name_model = cfg.model_PixelCNN_name
    else:
        name_model = 'model_' + time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.t7'
    state = {
        'model': model_PixelCNN.state_dict(),
        'id': cfg.id, 
        'batch_size': cfg.batch_size, 
        'epochs': cfg.epochs + epochs_old, 
        'loss_last': loss.item(), 
        'name_dataset': cfg.name_dataset
    }
    torch.save(state, cfg.path_model_PixelCNN + name_model)
    print(name_model)
       