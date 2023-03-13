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
import time

if __name__ == '__main__':
    batch_size = 32
    embedding_dim = 16
    num_embeddings = 128
    id = 0
    select_id = 1
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 定义训练集数据集
    dataset_train = Brain_Dataset(mode='train', id=id)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    model = VQVAE(in_dim=1, embedding_dim=embedding_dim, num_embeddings=num_embeddings)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train VQ-VAE
    epochs = 100
    print_freq = 200
    model.train()
    recon_loss_average_last = 1e5
    epoch_auto = 0
    for epoch in range(epochs):
        epoch_auto = epoch + 1
        recon_loss_sum = 0
        print("Start training epoch {}".format(epoch+1, ))
        for i, (images, labels) in enumerate(train_loader):
            input_size = images.shape[0]
            images = images.cuda()
            loss_sum, e_q_loss, recon_loss, x_recon = model(images)
            recon_loss_sum = recon_loss_sum + recon_loss
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader):
                print("\t [{}/{}]: loss_sum {}, reconstruct loss {}, e_q_loss {}".format(i+1, len(train_loader), loss_sum.item(), recon_loss.item(), e_q_loss.item()))
        recon_loss_average = recon_loss_sum / ((len(train_loader)-1) * batch_size + input_size)
        recon_loss_average = torch.sqrt(recon_loss_average)
        print('After {} epoch, the average reconstruct loss is {}'.format(epoch+1, recon_loss_average))
        # 是否以重构损失上升为准结束训练
        if(select_id == 0):
            continue
        else:
            # 如果重构损失上升，则停止训练
            if(recon_loss_average <= 90):
                # recon_loss_average_last = recon_loss_average
                break
            else:
                # break
                pass
    select_mode = ['_manual', '_auto']
    path_model = './Checkpoint/'
    name_dataset = '_Brain'
    name_mode = select_mode[select_id]
    batch_str = '_' + str(batch_size)
    epoch_str = '_' + str(epoch_auto)
    id_str = '_' + str(id)
    name_model = 'model_' + time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + name_dataset + name_mode + batch_str + epoch_str + id_str
    torch.save(model.state_dict(), path_model + name_model)
    print(name_model)

