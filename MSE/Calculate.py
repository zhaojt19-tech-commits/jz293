import sys
sys.path.append("..")
from Data.Dataset_2 import Brain_Dataset as Dataset_2
from Data.Dataset_3 import Brain_Dataset as Dataset_3
from Data.Dataset_4 import Brain_Dataset as Dataset_4
from Data.SSIM import *
import os
from tqdm import tqdm
import random
import csv
import shutil
import json

split_list = [2, 3, 4]
name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]
temp_list = [0, 1, 2, 3, 4, 5, 6]
id_list = [0, 1, 2, 3, 4]
id_list_4 = [0, 1, 2, 3, 4, 5]
proportion_list = [0.005, 0.05, 0.1, 0.15, 0.2, 0.25]
distance_list = [1, 2, 3, 4, 5, 6, 7, 8]

name_model_list = []

def initialize_random_seed(seed=0):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def MSE_ori(x, recon):
        loss_fn = torch.nn.MSELoss(reduction = 'mean')
        recon_loss = loss_fn(recon, x)

        return recon_loss
    
def MSE_sample(x, recon):
    loss_fn = torch.nn.MSELoss(reduction = 'none')
    recon_loss = loss_fn(recon, x)

    input_size = x.shape[0]
    recon_loss = recon_loss.reshape(input_size, -1)
    recon_loss = recon_loss.mean(axis=1, keepdim=False)

    return recon_loss

# path_data = './data_pred.csv'
# if os.path.exists(path_data):
#     # os.remove(path_data)
#     f_data =  open(path_data, 'a', encoding='UTF8', newline='')
#     writer_data = csv.writer(f_data)
# else:
#     header = ['name_dataset', 'split_id', 'seed', 'proportion', 'MSE', 'SSIM', 'PSNR', 'distance']
#     f_data =  open(path_data, 'w', encoding='UTF8', newline='')
#     writer_data = csv.writer(f_data)
#     writer_data.writerow(header)

# path_average = './average_pred.csv'
# if os.path.exists(path_average):
#     # os.remove(path_average)
#     f_average =  open(path_average, 'a', encoding='UTF8', newline='')
#     writer_average = csv.writer(f_average)
# else:
#     header = ['name_dataset', 'split_id', 'proportion', 'MSE', 'SSIM', 'PSNR', 'distance']
#     f_average =  open(path_average, 'w', encoding='UTF8', newline='')
#     writer_average = csv.writer(f_average)
#     writer_average.writerow(header)

# 定义存储字典
fig = {}
fig['fig2'] = {'MSE':{}, 'SSIM':{}, 'PSNR':{}}
fig['fig3'] = {'MSE':{}, 'SSIM':{}, 'PSNR':{}}
fig['fig4'] = {'MSE':{'E11.5':{}, 'E13.5':{}, 'E15.5':{}, 'E18.5':{}, 'P4':{}, 'P14':{}, 'P56':{}}, 
               'SSIM':{'E11.5':{}, 'E13.5':{}, 'E15.5':{}, 'E18.5':{}, 'P4':{}, 'P14':{}, 'P56':{}}, 
               'PSNR':{'E11.5':{}, 'E13.5':{}, 'E15.5':{}, 'E18.5':{}, 'P4':{}, 'P14':{}, 'P56':{}}
              }
for name in name_list:
    for prop in proportion_list:
                fig['fig4']['MSE'][name][prop] = []
                fig['fig4']['SSIM'][name][prop] = []
                fig['fig4']['PSNR'][name][prop] = []
                
for split_id in split_list:
    for name_id in temp_list:
        name = name_list[name_id]
        shape = shape_list[name_id]

        MSE_Average_list = []
        SSIM_Average_list = []
        PSNR_Average_list = []

        for distance in distance_list:
            MSE_Sum = 0
            SSIM_Sum = 0
            PSNR_Sum = 0
            if(split_id == 4):
                num_id = 6
            else:
                num_id = 5
            
            for id in range(num_id):
                
                # 训练参数
                batch_size = 128
                seed = id
                squeeze = True
                device = 'cuda:0'
                if(split_id == 4 or split_id == 5):
                    proportion = proportion_list[id]
                else:
                    proportion = 0.1
                initialize_random_seed(seed)

                num_classes = [shape[0], shape[1]]
                name_dataset = name
                
                if(split_id == 2):
                    dataset_test = Dataset_2(mode='test', name_id=name_id, seed=seed, proportion=proportion, squeeze=squeeze)
                if(split_id == 3):
                    dataset_test = Dataset_3(mode='test', name_id=name_id, seed=seed, proportion=proportion, squeeze=squeeze)
                if(split_id == 4):
                    dataset_test = Dataset_4(mode='test', name_id=name_id, seed=seed, proportion=proportion, squeeze=squeeze)
                picture_num = dataset_test.__len__()
                test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
                
                # 取出训练集数据备用
                feature_train = dataset_test.feature_train
                labels_train = np.array(dataset_test.label_train)
                labels_train_0 = labels_train[:, 0]
                labels_train_1 = labels_train[:, 1]

                MSE_sum = 0       
                PSNR_sum = 0
                SSIM_sum = 0
                with torch.no_grad():
                    for t, (images, labels) in enumerate(tqdm(test_loader)):
                        input_size = labels.shape[0]

                        # 找到临近切片
                        seed_index = []
                        labels = np.array(labels)
                        labels_test_0 = labels[:, 0]
                        labels_test_1 = labels[:, 1]
                        for i in range(input_size):
                            temp_index = np.argwhere(labels_train_0 == labels_test_0[i]).flatten()
                            temp_dis = (labels_train_1[temp_index] - labels_test_1[i])**2
                            sort_id = np.argsort(temp_dis)
                            sort_dis = np.sort(temp_dis)
                            j = 1
                            k = 0
                            while(j < distance and sort_dis.shape[0] > 1):
                                if(k + 1 >= sort_dis.shape[0]):
                                    break
                                k = k + 1
                                if(sort_dis[k] > sort_dis[k-1]):
                                    j = j + 1
                            index_select = temp_index[sort_id[k]]
                            seed_index.append(index_select)
                            
                        seed_index = np.array(seed_index).astype(np.int32)

                        # print(labels_test_1)
                        # print(labels_train_1[seed_index])

                        feature_seed = feature_train[seed_index].reshape(input_size, 1, shape[2], shape[3])
                        feature_seed = torch.tensor(feature_seed).to(device)

                        labels = torch.tensor(labels)

                        images = images.to(device)
                        loss_MSE = MSE_ori(x=images.reshape(input_size, 1, shape[2], shape[3]), recon=feature_seed.reshape(input_size, 1, shape[2], shape[3]))
                        MSE_sum = MSE_sum + loss_MSE.item()*input_size

                        loss_MSE_sample = MSE_sample(x=images.reshape(input_size, 1, shape[2], shape[3]), recon=feature_seed.reshape(input_size, 1, shape[2], shape[3]))

                        # 计算图像平均PSNR
                        PSNR_temp_sum = 0
                        for g in range(input_size):
                            psnr = 10 * torch.log10(1 * 1 / loss_MSE_sample[g])
                            PSNR_temp_sum = PSNR_temp_sum + psnr
                        PSNR_sum = PSNR_sum + PSNR_temp_sum.item()
                        
                        # 计算图像平均SSIM
                        ssim = ssim_calculate(images.reshape(input_size, 1, shape[2], shape[3]), feature_seed.reshape(input_size, 1, shape[2], shape[3]), window_size = 7)
                        SSIM_sum = SSIM_sum + ssim.item() * input_size
                                    
                # 输出分析结果
                MSE_average = MSE_sum / picture_num
                PSNR_average = PSNR_sum / picture_num
                SSIM_average = SSIM_sum / picture_num
                if(split_id == 4):
                    fig['fig4']['MSE'][name][proportion].append(MSE_average)
                    fig['fig4']['SSIM'][name][proportion].append(SSIM_average)
                    fig['fig4']['PSNR'][name][proportion].append(PSNR_average)

                print(name_dataset, end =' ')
                print(str(seed)+':')
                print('When the distance is {}, the average Pixel MSE loss is {}, PSNR is {}, SSIM is {}'.format(distance, MSE_average, PSNR_average, SSIM_average))

                # row_data = [name_dataset, split_id, seed, proportion, MSE_average, SSIM_average, PSNR_average, distance]
                # writer_data.writerow(row_data)
                
                MSE_Sum = MSE_Sum + MSE_average
                SSIM_Sum = SSIM_Sum + SSIM_average
                PSNR_Sum = PSNR_Sum + PSNR_average

            MSE_Average = MSE_Sum / 5.0
            SSIM_Average = SSIM_Sum / 5.0
            PSNR_Average = PSNR_Sum / 5.0

            MSE_Average_list.append(MSE_Average)
            SSIM_Average_list.append(SSIM_Average)
            PSNR_Average_list.append(PSNR_Average)
            # row_average = [name_dataset, split_id, proportion, MSE_Average, SSIM_Average, PSNR_Average, distance]
            # if(split_id == 2 or split_id == 3):
            #     writer_average.writerow(row_average)

        if(split_id == 2):
            fig['fig2']['MSE'][name] = MSE_Average_list
            fig['fig2']['SSIM'][name] = SSIM_Average_list
            fig['fig2']['PSNR'][name] = PSNR_Average_list
        if(split_id == 3):
            fig['fig3']['MSE'][name] = MSE_Average_list
            fig['fig3']['SSIM'][name] = SSIM_Average_list
            fig['fig3']['PSNR'][name] = PSNR_Average_list
        # row_zeros = [0] * 7
        # for i in range(3):
        #     writer_data.writerow(row_zeros)
        # for i in range(3):
        #     writer_average.writerow(row_zeros)

# f_data.close()
# f_average.close()
with open('data_seed.json', 'w') as fp:
    json.dump(fig, fp)

