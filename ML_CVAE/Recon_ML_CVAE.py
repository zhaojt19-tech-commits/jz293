import os
import sys
sys.path.append("..")
split_id = 4
if(split_id == 2):
    from Data.Dataset_2 import *
if (split_id == 3):
    from Data.Dataset_3 import *
if(split_id == 4):
    from Data.Dataset_4 import *
from Data.SSIM import *
import numpy as np
import torch
import argparse
from tqdm import tqdm
from ML_CVAE_model import *
import matplotlib.pyplot as plt
import csv
import shutil

name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]

name_model_list = []

temp_list = [0, 1, 2, 3, 4, 5, 6]
id_list = [0, 1, 2, 3, 4, 5]
proportion_list = [0.005, 0.05, 0.1, 0.15, 0.2, 0.25]

def initialize_random_seed(seed=0):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

path_data = './data_recon_CP.csv'
if os.path.exists(path_data):
    # os.remove(path_data)
    f_data =  open(path_data, 'a', encoding='UTF8', newline='')
    writer_data = csv.writer(f_data)
else:
    header = ['name_dataset', 'split_id', 'seed', 'proportion', 'MSE', 'SSIM', 'PSNR']
    f_data =  open(path_data, 'w', encoding='UTF8', newline='')
    writer_data = csv.writer(f_data)
    writer_data.writerow(header)

path_average = './average_recon_CP.csv'
if os.path.exists(path_average):
    # os.remove(path_average)
    f_average =  open(path_average, 'a', encoding='UTF8', newline='')
    writer_average = csv.writer(f_average)
else:
    header = ['name_dataset', 'split_id', 'proportion', 'MSE', 'SSIM', 'PSNR']
    f_average =  open(path_average, 'w', encoding='UTF8', newline='')
    writer_average = csv.writer(f_average)
    writer_average.writerow(header)

for name_id in temp_list:
    MSE_Sum = 0
    SSIM_Sum = 0
    PSNR_Sum = 0
    for id in id_list:
        name = name_list[name_id]
        shape = shape_list[name_id]

        model_name = 'model_' + name + '_' + str(split_id) + '_' + str(id) + '_CP' +'.t7' 
        # model_name = 'model_' + name + '_' + str(split_id) + '_' + str(id) +'.t7'
        squeeze = True
        device = 'cuda:2'
        if(split_id == 4 or split_id == 5):
            proportion = proportion_list[id]
        else:
            proportion = 0.1
        seed = id
        initialize_random_seed(seed)
        name_dataset = name
        image_H = shape[2]
        image_W = shape[3]
        # 设置输入参数
        parser = argparse.ArgumentParser(description='ML_CVAE')
        parser.add_argument('--is_continue', type=bool, default=False,
                            help='weather train from the old model')
        parser.add_argument('--epochs', type=int, default=1,
                            help='Number of epochs to train model for')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Number of images per mini-batch')
        parser.add_argument('--lr', type=float, default=1e-4,
                            help='Learning rate of optimizer')                    
        parser.add_argument('--id', type=int, default=0,
                            help='id of the K-fold for test')
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
        parser.add_argument('--beta', type=float, default=0)
        parser.add_argument('--distribution', type=str, default='gaussian')
        parser.add_argument('--use_ode2cvae_arch', type=bool, default=True)
        # parser.add_argument('--dict_keep', type=dict, default={'dataset':'drug'})    # TODO 这部分可以改成自己的逻辑
        

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

        checkpoint = torch.load(cfg.path_model + model_name)
        model.load_state_dict(checkpoint['model'])
        epochs_old = checkpoint['epochs']

        model.to(device)
        model.eval()

        # 重构训练集图片文件夹
        path_img = './Pictures_recon/' + name_dataset + '/' + str(split_id) + '/' +str(id)
        if not os.path.exists(path_img):
            os.makedirs(path_img)
        else:
            shutil.rmtree(path_img)
            os.makedirs(path_img)

        if not os.path.exists('./CSV_recon/'+ name_dataset+ '/'):
            os.makedirs('./CSV_recon/'+ name_dataset+ '/')
        path_csv = './CSV_recon/' + name_dataset + '/' + str(split_id) + '_' + str(id) + '.csv'
        if os.path.exists(path_csv):
            os.remove(path_csv)
        header = ['gene_id', 'slice_id', 'MSE', 'SSIM', 'PSNR']
        f_csv =  open(path_csv, 'w', encoding='UTF8', newline='')
        writer = csv.writer(f_csv)
        writer.writerow(header)
        
        # 抽取的切片id
        id_low = int((shape[1] / 2) - 5)
        id_high = int((shape[1] / 2) + 5)
        sample_id = [i for i in range(id_low, id_high, 1)]
        # 抽取重建图片的频率
        sample_freq = 50    

        loss_sum = 0        # 所有训练集图片总损失
        Reconstruction_Loss_Img_sum = 0
        Reconstruction_Loss_Classes_sum = 0
        KLD_sum = 0
        MSE_sum = 0
        PSNR_sum = 0
        SSIM_sum = 0
        # 开始训练模型
        with torch.no_grad():
            for t, (images, labels) in enumerate(tqdm(train_loader)):
                input_size = labels.shape[0]
                label_slice = labels[:, 1].ravel()
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
                images = images.to(device)
                labels = labels.to(device)
                # forward函数输出
                output_list = model(images, [one_hot_0, labels[:, 1].reshape(-1, 1).float()])
                # # 计算损失
                # loss_dic = model.loss_function(*output_list)
                # 得到各部分损失
                # loss = loss_dic['loss']
                # Reconstruction_Loss_Img = loss_dic['Reconstruction_Loss_Img']
                # Reconstruction_Loss_Classes = loss_dic['Reconstruction_Loss_Classes']
                # KLD = loss_dic['KLD']
                # loss_sum = loss_sum + loss.item()*input_size
                # Reconstruction_Loss_Img_sum = Reconstruction_Loss_Img_sum + Reconstruction_Loss_Img.item()*input_size
                # Reconstruction_Loss_Classes_sum = Reconstruction_Loss_Classes_sum + Reconstruction_Loss_Classes.item()*input_size
                # KLD_sum = KLD_sum + KLD.item()*input_size

                sample_list = []
                for i in range(label_slice.shape[0]):
                    if label_slice[i] in sample_id:
                        sample_list.append(i)

                xhat = output_list[0]
                loss_MSE = model.MSE_ori(images, xhat)
                MSE_sum = MSE_sum + loss_MSE.item()*input_size
                
                # 计算图片的分别MSE
                loss_MSE_sample = model.MSE_sample(images, xhat)

                # 计算图像平均PSNR
                PSNR_temp_sum = 0
                for g in range(input_size):
                    psnr = 10 * torch.log10(1 * 1 / loss_MSE_sample[g])
                    PSNR_temp_sum = PSNR_temp_sum + psnr
                PSNR_sum = PSNR_sum + PSNR_temp_sum.item()
                
                # 计算图像平均SSIM
                ssim = ssim_calculate(images, xhat, window_size = 7)
                SSIM_sum = SSIM_sum + ssim.item() * input_size
                
                img_ori_sample = images[sample_list]
                img_pre_sample = xhat[sample_list]
                labels_sample = labels[sample_list]
                loss_MSE_sample = loss_MSE_sample[sample_list]

                if(t % sample_freq == 0):
                    # 将MSE写入.csv文件
                    for m in range(loss_MSE_sample.shape[0]):
                        row = [labels_sample[m][0].item(), labels_sample[m][1].item(), loss_MSE_sample[m].item()]
                        writer.writerow(row)

                    img_ori_sample = img_ori_sample.cpu().numpy()
                    img_pre_sample = img_pre_sample.cpu().numpy()
                    for j in range(labels_sample.shape[0]):
                        plt.imsave(path_img+'/'+str(labels_sample[j][0].item())+'_'+str(labels_sample[j][1].item())+'_pre'+'.png', img_pre_sample[j].squeeze(0))
                        plt.imsave(path_img+'/'+str(labels_sample[j][0].item())+'_'+str(labels_sample[j][1].item())+'_ori'+'.png', img_ori_sample[j].squeeze(0))
                
                    
        # loss_average = loss_sum / picture_num
        # Reconstruction_Loss_Img_average = Reconstruction_Loss_Img_sum / (picture_num*shape[2]*shape[3])
        # Reconstruction_Loss_Classes_average = Reconstruction_Loss_Classes_sum / picture_num
        # KLD_average = KLD_sum / picture_num
        MSE_average = MSE_sum / picture_num 
        PSNR_average = PSNR_sum / picture_num
        SSIM_average = SSIM_sum / picture_num
        print(name_dataset, end =' ')
        print(str(seed)+':')
        print('After {} epoch, the average Pixel MSE loss is {}, PSNR is {}, SSIM is {}'.format(epochs_old, MSE_average, PSNR_average, SSIM_average))
        f_csv.close()

        row_data = [name_dataset, split_id, seed, proportion, MSE_average, SSIM_average, PSNR_average]
        writer_data.writerow(row_data)
    
        MSE_Sum = MSE_Sum + MSE_average
        SSIM_Sum = SSIM_Sum + SSIM_average
        PSNR_Sum = PSNR_Sum + PSNR_average

    MSE_Average = MSE_Sum / 5.0
    SSIM_Average = SSIM_Sum / 5.0
    PSNR_Average = PSNR_Sum / 5.0
    row_average = [name_dataset, split_id, proportion, MSE_Average, SSIM_Average, PSNR_Average]
    # writer_average.writerow(row_average)
        
f_data.close()
f_average.close()
    
       