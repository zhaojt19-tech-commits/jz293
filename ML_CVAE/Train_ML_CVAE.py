import os
import sys
sys.path.append("..")
split_id = 2
if(split_id == 2):
    from Data.Dataset_2 import *
if (split_id == 3):
    from Data.Dataset_3 import *
if(split_id == 4):
    from Data.Dataset_4 import *
if(split_id == 5):
    from Data.Dataset_5 import *
import numpy as np
import torch
import torch.optim as optim
import argparse
from tqdm import tqdm
from ML_CVAE_model import *


name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]

temp_list = [0, 1, 2, 3, 4, 5, 6]
id_list = [0, 1, 2, 3, 4]
proportion_list = [0.005, 0.05, 0.1, 0.15, 0.2, 0.25]

def initialize_random_seed(seed=0):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

name_model_list = []
for name_id in temp_list:
    print('开始在'+name_list[name_id]+'上训练：')
    for id in id_list:
        name = name_list[name_id]
        shape = shape_list[name_id]
        
        # 训练参数
        squeeze = True
        device = 'cuda:3'
        seed = id
        if(split_id == 4 or split_id == 5):
            proportion = proportion_list[id]
        else:
            proportion = 0.1
        initialize_random_seed(seed)

        # 设置输入参数
        parser = argparse.ArgumentParser(description='ML_CVAE')
        parser.add_argument('--is_continue', type=bool, default=False,
                            help='weather train from the old model')
        # 设置训练epoch数
        parser.add_argument('--epochs', type=int, default=20,
                            help='Number of epochs to train model for')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Number of images per mini-batch')
        parser.add_argument('--lr', type=float, default=1e-4,
                            help='Learning rate of optimizer')

        parser.add_argument('--model_name', type=str, default=' ',
                            help='name of the model of VAE')
        parser.add_argument('--path_model', type=str, default='./Checkpoint/',
                            help='the folder of the checkpoint of VAE')
        
        # 模型定义参数
        parser.add_argument('--img_size', type=List[int], default=[shape[2], shape[3]])
        parser.add_argument('--in_channels', type=int, default=1)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--class_sizes', type=List[int], default=[shape[0], 1])
        parser.add_argument('--latent_dim', type=int, default=128)
        parser.add_argument('--hidden_dims', type=List[int], default=[32, 64, 128, 256])
        parser.add_argument('--kernel_size', type=int, default=3)
        parser.add_argument('--stride', type=int, default=2)
        parser.add_argument('--padding', type=int, default=1)
        parser.add_argument('--output_padding', type=int, default=0)
        parser.add_argument('--decomp', type=str, default='CP')
        parser.add_argument('--k', type=int, default=64)
        parser.add_argument('--beta', type=float, default=1)
        parser.add_argument('--distribution', type=str, default='gaussian')
        parser.add_argument('--use_ode2cvae_arch', type=bool, default=True)
        # parser.add_argument('--dict_keep', type=dict, default={'dataset':'drug'})    # TODO 这部分可以改成自己的逻辑
        
        name_dataset = name
        image_H = shape[2]
        image_W = shape[3]
        # 实例化参数对象
        cfg = parser.parse_args()

        # 定义DataLoader
        dataset_train = Brain_Dataset(mode='train', name_id=name_id, seed=seed, proportion=proportion, squeeze=squeeze)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True)

        picture_num = dataset_train.__len__()
        
        # 定义模型
        model = SquareConvolutionalGeorgopoulosMLCVAE(name_id, device, cfg.img_size, cfg.in_channels, cfg.num_classes, cfg.class_sizes, 
        cfg.latent_dim, cfg.hidden_dims, cfg.kernel_size, cfg.stride, cfg.padding, cfg.output_padding, cfg.decomp,
        cfg.k, cfg.beta, cfg.distribution, cfg.use_ode2cvae_arch)


        epochs_old = 0
        if(cfg.is_continue == True):
            checkpoint = torch.load(cfg.path_model + cfg.model_name)
            model.load_state_dict(checkpoint['model'])
            epochs_old = checkpoint['epochs']
        else:
            pass

        model.to(device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

        loss_sum = 0        # 所有训练集图片总损失
        Reconstruction_Loss_Img_sum = 0
        Reconstruction_Loss_Classes_sum = 0
        KLD_sum = 0
        MSE_sum = 0
        # 开始训练模型
        for e in range(cfg.epochs):
            loss_sum = 0
            Reconstruction_Loss_Img_sum = 0
            Reconstruction_Loss_Classes_sum = 0
            KLD_sum = 0
            MSE_sum = 0
            for t, (images, labels) in enumerate(tqdm(train_loader)):
                input_size = labels.shape[0]
                label_double = labels.reshape(input_size, -1)
                label0 = label_double[:, 0].long()
                # label1 = label_double[:, 1].long()
                index = np.linspace(0, input_size - 1, input_size).astype("long")
                one_hot_0 = torch.zeros(input_size, cfg.class_sizes[0])
                # one_hot_1 = torch.zeros(input_size, cfg.class_sizes[1])
                one_hot_0[[index, label0]] = 1
                # one_hot_1[[index, label1]] = 1
                one_hot_0 = one_hot_0.to(device)
                # one_hot_1 = one_hot_1.to(device)
                labels = labels.to(device)
                images = images.reshape(input_size, 1, shape[2], shape[3]).to(device)
                optimizer.zero_grad()
                # forward函数输出
                output_list = model(images, [one_hot_0, labels[:, 1].reshape(-1, 1).float()])
                # 计算损失
                loss_dic = model.loss_function(*output_list)
                # 得到各部分损失
                loss = loss_dic['loss']
                Reconstruction_Loss_Img = loss_dic['Reconstruction_Loss_Img']
                Reconstruction_Loss_Classes = loss_dic['Reconstruction_Loss_Classes']
                KLD = loss_dic['KLD']
                loss_sum = loss_sum + loss.item()*input_size
                Reconstruction_Loss_Img_sum = Reconstruction_Loss_Img_sum + Reconstruction_Loss_Img.item()*input_size
                Reconstruction_Loss_Classes_sum = Reconstruction_Loss_Classes_sum + Reconstruction_Loss_Classes.item()*input_size
                KLD_sum = KLD_sum + KLD.item()*input_size
                # 计算梯度并优化
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    xhat = output_list[0]
                    loss_MSE = model.MSE_ori(images, xhat)
                    MSE_sum = MSE_sum + loss_MSE.item()*input_size
    
            loss_average = loss_sum / picture_num
            Reconstruction_Loss_Img_average = Reconstruction_Loss_Img_sum / (picture_num*shape[2]*shape[3])
            Reconstruction_Loss_Classes_average = Reconstruction_Loss_Classes_sum / picture_num
            KLD_average = KLD_sum / picture_num
            MSE_average = MSE_sum / picture_num
            print(name_dataset + ':')
            print('After {} epoch, the average loss is {}, the average image MSE loss is {}, the average label MSE loss is {}, the average KLD is {}'.format(e+1+epochs_old, 
            loss_average, MSE_average, Reconstruction_Loss_Classes_average, KLD_average))
            # scheduler.step()
        if(cfg.is_continue == True):
            name_model = cfg.model_name
        else:
            if(cfg.decomp == 'CP'):
                name_model = 'model_' + name_dataset + '_' + str(split_id) + '_' + str(seed) + '_CP' + '.t7'
            else:
                name_model = 'model_' + name_dataset + '_' + str(split_id) + '_' + str(seed)  + '.t7'
        state = {
            'model': model.state_dict(),
            'id': seed, 
            'name_id': name_id, 
            'batch_size': cfg.batch_size, 
            'epochs': cfg.epochs + epochs_old, 
            'loss_last': loss.item(), 
            'name_dataset': name_dataset
        }
        torch.save(state, cfg.path_model + name_model)
        name_model_list.append(name_model)

print(name_model_list)
       