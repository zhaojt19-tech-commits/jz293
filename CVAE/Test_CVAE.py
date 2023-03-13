import matplotlib.pyplot as plt
import sys
sys.path.append("..")
split_id = 4
if(split_id == 2):
    from Data.Dataset_2 import *
if (split_id == 3):
    from Data.Dataset_3 import *
if(split_id == 4):
    from Data.Dataset_4 import *
if(split_id == 5):
    from Data.Dataset_5 import *
from Data.SSIM import *
import os
from tqdm import tqdm
from CVAE_label_module import *
import csv
import shutil

name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]
temp_list = [0, 1, 2, 3, 4, 5, 6]
id_list = [0, 1, 2, 3, 4, 5]
proportion_list = [0.005, 0.05, 0.1, 0.15, 0.2, 0.25]
name_model_list = []

def initialize_random_seed(seed=0):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

path_data = './data_pred.csv'
if os.path.exists(path_data):
    # os.remove(path_data)
    f_data =  open(path_data, 'a', encoding='UTF8', newline='')
    writer_data = csv.writer(f_data)
else:
    header = ['name_dataset', 'split_id', 'seed', 'proportion', 'MSE', 'SSIM', 'PSNR']
    f_data =  open(path_data, 'w', encoding='UTF8', newline='')
    writer_data = csv.writer(f_data)
    writer_data.writerow(header)

path_average = './average_pred.csv'
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
        # 模型保存
        path_model = './Checkpoint/'
        model_name = 'model_' + name + '_' + str(split_id) + '_' + str(id)  + '.t7' 
        name_dataset = name

        # 模型参数
        num_classes = [shape[0], shape[1]]
        image_H = shape[2]
        image_W = shape[3]
        conv_dims = [32, 64, 128, 256]
        latent_dim = 128
        embedding_dim = 64

        # 训练参数
        if(split_id == 4 or split_id == 5):
            proportion = proportion_list[id]
        else:
            proportion = 0.1
        seed = id
        batch_size = 128
        squeeze = True
        device = 'cuda:2'
        beta = 0
        mode_generate = 1  # 0：同基因临近切片 1：直接随机采样生成
        initialize_random_seed(seed)
        
        # 定义模型
        model = CVAE_label(image_H, image_W, 1, conv_dims, latent_dim, num_classes, name_id, embedding_dim)

        # 加载训练好的模型参数
        checkpoint = torch.load(path_model + model_name)
        model.load_state_dict(checkpoint['model'])
        epochs_old = checkpoint['epochs']
        model = model.to(device)
        model.eval()

        # 抽取预测图片的频率
        sample_freq = 50    

       # 预测测试集图片文件夹
        path_img = './Pictures_pred/' + name_dataset + '/' + str(split_id) + '/' +str(id)
        if not os.path.exists(path_img):
            os.makedirs(path_img)
        else:
            shutil.rmtree(path_img)
            os.makedirs(path_img)

        if not os.path.exists('./CSV_pred/'+ name_dataset+ '/'):
            os.makedirs('./CSV_pred/'+ name_dataset+ '/')
        path_csv = './CSV_pred/' + name_dataset + '/' + str(split_id) + '_' + str(id) + '.csv'
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

        dataset_test = Brain_Dataset(mode='test', transform=None, name_id=name_id, seed=seed, proportion=proportion, squeeze=squeeze)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

        # 取出训练集数据备用
        feature_train = dataset_test.feature_train
        label_train = dataset_test.label_train
        picture_num = dataset_test.__len__()

        loss_sum = 0
        MSE_sum = 0
        PSNR_sum = 0
        SSIM_sum = 0
        # pixel_max_list = []
        # pixel_min_list = []
        with torch.no_grad():
            for t, (images, labels) in enumerate(tqdm(test_loader)):
                input_size = labels.shape[0]

                images = images.to(device)

                label_slice = labels[:, 1].ravel()

                # 相同基因临近切片
                if (mode_generate == 0):
                    # 找到临近切片
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
                            min_temp = (labels1[i] - label_train1[j]) ** 2
                            if (min_temp < min):
                                min = min_temp
                                index_temp = j
                        index_train.append(index_temp)
                    index_train = np.array(index_train)

                    feature_seed = feature_train[index_train].reshape(input_size, 1, image_H, image_W)
                    feature_seed = torch.tensor(feature_seed).to(device)

                    labels = torch.tensor(labels)

                    labels = labels.to(device)
                    xhat, mu, log_var = model(feature_seed, one_hot_0, one_hot_1, labels, image_H, image_W, input_size)
                    
                    loss, kl_loss, recon_loss = model.compute_loss(images, xhat, mu, log_var, beta)
                    loss_sum = loss_sum + loss.cpu()*input_size
                
                # 随机采样生成
                if (mode_generate == 1):
                    labels = labels.to(device)
                    z = torch.randn(input_size, latent_dim).to(device)
                    xhat = model.generate(z, labels)

                sample_list = []
                for i in range(label_slice.shape[0]):
                    if label_slice[i] in sample_id:
                        sample_list.append(i) 

                # pixel_max_list.append(torch.max(xhat).item())
                # pixel_min_list.append(torch.min(xhat).item())

                # max = max.ravel().to(device)
                # images = (images.reshape(input_size, -1) * max.reshape(-1, 1)).reshape(input_size, 1, image_H, image_W)
                # xhat = (xhat.reshape(input_size, -1) * max.reshape(-1, 1)).reshape(input_size, 1, image_H, image_W)

                loss_MSE = model.MSE_ori(images, xhat)
                MSE_sum = MSE_sum + loss_MSE.item()*input_size

                # 计算图片的分别MSE
                loss_MSE_sample = model.MSE_sample(images, xhat)

                # 计算图像平均PSNR
                PSNR_temp_sum = 0
                for g in range(input_size):
                    # psnr = 10 * torch.log10(max[g] * max[g] / loss_MSE_sample[g])
                    psnr = 10 * torch.log10(1 * 1 / loss_MSE_sample[g])
                    PSNR_temp_sum = PSNR_temp_sum + psnr
                PSNR_sum = PSNR_sum + PSNR_temp_sum.item()
                
                # 计算图像平均SSIM
                ssim = ssim_calculate(images, xhat, window_size = 7)
                SSIM_sum = SSIM_sum + ssim.item() * input_size

                # 采样图片的数值信息
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
            

        # 输出分析结果
        MSE_average = MSE_sum / (picture_num) 
        PSNR_average = PSNR_sum / picture_num
        SSIM_average = SSIM_sum / picture_num
        print(name_dataset, end =' ')
        print(str(seed)+':')
        print('After {} epoch, the average Pixel MSE loss is {}, PSNR is {}, SSIM is {}'.format(epochs_old, MSE_average, PSNR_average, SSIM_average))
        f_csv.close()

        row_data = [name_dataset, split_id, seed, proportion, MSE_average, SSIM_average, PSNR_average]
        writer_data.writerow(row_data)

        # pixel_max_list.sort()
        # pixel_min_list.sort()
        # print(pixel_max_list)
        # print(pixel_min_list)
        
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



