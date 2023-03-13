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
from GatedPixelCNN_Model import PixelCNN
import argparse
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from utils import str2bool, save_samples, get_loaders
from tqdm import tqdm

if __name__ == '__main__':
    # TODO
    model_VAE_name = 'model_2022_05_27_01_11_15_Brain_auto_32_8_0'
    path_model_VAE = './Checkpoint/'
    model_PixelCNN_name = 'model_2022_06_26_08_28_37_Brain_128_40_0'
    path_model_PixelCNN = './Checkpoint_PixelCNN/'
    batch_size = 256
    embedding_dim = 16
    num_embeddings = 128
    id = 0
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # dataset_train = Brain_Dataset(mode='train', id=id)
    dataset_test = Brain_Dataset(mode='test', id=id)
    # train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    model_VAE = VQVAE(in_dim=1, embedding_dim=embedding_dim, num_embeddings=num_embeddings)
    model_VAE.load_state_dict(torch.load(path_model_VAE + model_VAE_name))
    model_VAE = model_VAE.cuda()
    model_VAE.eval()
    sample_freq = 50    # 抽取重建图片的频率
    # 生成重构图片文件夹
    path = "./Pictures_generate"
    if not os.path.exists(path):
        os.mkdir(path)
    path = "./Pictures_generate/id0"
    if not os.path.exists(path):
        os.mkdir(path)
    path_this = "./Pictures_generate/id0"
    # TODO
    mode_generate = 0

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
    model_PixelCNN.load_state_dict(torch.load(path_model_PixelCNN + model_PixelCNN_name))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    model_PixelCNN.to(device)
    
    generate_loss_sum = 0
    # 直接抽样生成
    if(mode_generate==0):
        with torch.no_grad():
            for t, (images, labels) in enumerate(tqdm(test_loader)):
                input_size = labels.shape[0]
                labels = labels.cuda()
                images = images.cuda()
                shape = (1, 28, 23)
                samples = model_PixelCNN.sample(shape=shape, count=input_size, label=labels, device='cuda')
                break
                encoding_indices = samples.permute(0, 2, 3, 1).contiguous().ravel().long()
                shape = [input_size, 28, 23, -1]
                images_generate = model_VAE.generate(encoding_indices, shape)
                # 计算MSE损失
                loss_fn = torch.nn.MSELoss(reduction = 'sum').cuda()
                generate_loss = loss_fn(images_generate, images).cpu()
                generate_loss_sum = generate_loss_sum + generate_loss
                print("t={} generate loss:{}".format(t, generate_loss))
                if(t % sample_freq == 0):
                    images_generate = images_generate.cpu().numpy()
                    images = images.cpu().numpy()
                    for j in range(input_size):
                        plt.imsave(path_this+'/'+str(t*sample_freq+j)+'_'+str(1)+'.png', images_generate[j].squeeze(0))
                        plt.imsave(path_this+'/'+str(t*sample_freq+j)+'_'+str(0)+'.png', images[j].squeeze(0))
        generate_loss_average = generate_loss_sum / ((len(test_loader)-1)*batch_size+input_size)
        generate_loss_average = torch.sqrt(generate_loss_average)
        print('测试集上平均每张图片重构损失为{}'.format(generate_loss_average))


    # 利用周围图片先验信息
    feature_train = dataset_test.feature_train
    label_train = dataset_test.label_train
    size_test = dataset_test.__len__()
    if(mode_generate==1):
        with torch.no_grad():
            for t, (images, labels) in enumerate(tqdm(test_loader)):
                input_size = labels.shape[0]
                label_train = np.array(label_train)
                index_train = []
                labels = np.array(labels)
                labels0 = labels[:, 0]
                labels1 = labels[:, 1]
                label_train0 = label_train[:, 0]
                label_train1 = label_train[:, 1]
                for i in range(len(labels0)):
                    temp = np.argwhere(label_train0 == labels0[i])
                    min = 1000
                    index_temp = 0
                    for j in temp:
                        min_temp = (labels1[i] - label_train1[j])**2
                        if(min_temp < min):
                            min = min_temp
                            index_temp = j
                    index_train.append(index_temp)
                index_train = np.array(index_train)
                feature_seed = torch.tensor(feature_train[index_train].reshape(input_size, 1, 109, 89)).cuda()
                encoding_indices_neighbor, x_recon = model_VAE(feature_seed)
                encoding_indices_neighbor = encoding_indices_neighbor.reshape(input_size, 28, 23, -1)
                encoding_indices_neighbor = encoding_indices_neighbor.permute(0, 3, 1, 2).contiguous().float()
                labels = torch.tensor(labels)
                labels = labels.cuda()
                images = images.cuda()
                shape = (1, 28, 23)
                samples = model_PixelCNN.sample(shape=shape, count=input_size, label=labels, images=encoding_indices_neighbor, device='cuda')
                encoding_indices = samples.permute(0, 2, 3, 1).contiguous().ravel().long()
                shape = [input_size, 28, 23, -1]
                images_generate = model_VAE.generate(encoding_indices, shape)
                # 计算MSE损失
                loss_fn = torch.nn.MSELoss(reduction = 'sum').cuda()
                generate_loss = loss_fn(images_generate, images).cpu()
                generate_loss_sum = generate_loss_sum + generate_loss
                print("t={} generate loss:{}".format(t, generate_loss))
                if(t % sample_freq == 0):
                    images_generate = images_generate.cpu().numpy()
                    images = images.cpu().numpy()
                    for j in range(input_size):
                        plt.imsave(path_this+'/'+str(t*sample_freq+j)+'_'+str(1)+'.png', images_generate[j].squeeze(0))
                        plt.imsave(path_this+'/'+str(t*sample_freq+j)+'_'+str(0)+'.png', images[j].squeeze(0))
        generate_loss_average = generate_loss_sum / ((len(test_loader)-1)*batch_size+input_size)
        generate_loss_average = torch.sqrt(generate_loss_average)
        print('测试集上平均每张图片重构损失为{}'.format(generate_loss_average))

