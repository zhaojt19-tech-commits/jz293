import sys
sys.path.append("..")
split_id = 2
if(split_id == 2):
    from Data.Dataset_transformer_2 import *
if (split_id == 3):
    from Data.Dataset_transformer_3 import *
if(split_id == 4):
    from Data.Dataset_transformer_4 import *
if(split_id == 5):
    from Data.Dataset_transformer_5 import *
from Data.SSIM import *
import matplotlib.pyplot as plt
from CVAE_Transformer_Module import *
from tqdm import tqdm
import csv
import shutil

name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]
temp_list = [0, 1, 2, 3, 4, 5, 6]
id_list = [0, 1, 2, 3, 4]
proportion_list = [0.005, 0.05, 0.1, 0.15, 0.2, 0.25]
name_model_list = []

def initialize_random_seed(seed=0):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

path_data = './data_pred_t.csv'
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

        model_name = 'model_' + name + '_' + str(split_id) + '_' + str(id) + '.t7'
        path_model = '../CVAE_Transformer_Batch/Checkpoint/'
        batch_size = 1
        squeeze = True
        seed = id
        if(split_id == 4 or split_id == 5):
            proportion = proportion_list[id]
        else:
            proportion = 0.1
        initialize_random_seed(seed)

        name_dataset = name
        # CVAE隐层维数
        latent_dim = 512
        # 切片id的embedding向量维数
        temporal_dim = 512
        # transformer类别embedding向量维数
        cat_dims = [32]
        # muti-head的数量
        num_heads = 16
        num_ref_points = 21
        tr_dim = 512
        adversarial_hidden_dims_encoder = None
        latent_hidden_dims_encoder = None
        # CVAE类别embedding向量维数
        cvae_yemb_dims = [32]
        # TODO 待考虑是否学习（仔细看代码）
        learn_temporal_emb = False
        # TODO 待考虑是否使用mask
        mask = None
        # 损失函数中KLD损失的系数
        beta = 0
        # 标签重构损失
        adv_cvae_weight=0
        # 标签重构损失
        adv_tr_weight=0
        mask= None
        minT = 10
        maxT = 50
        device = 'cuda:1'
        num_cont = None
        other_temporal_dims = None
        other_minT = None
        other_maxT = None
        cont_model = 'sinusoidal'
        rec_loss = 'mse'
        temporal_uncertainty = False

        dataset_test = Brain_Dataset(mode='test', name_id=name_id, seed=seed, proportion=proportion, squeeze=squeeze)
        picture_num = dataset_test.get_picture_num()
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        # 定义模型
        model = CatConTransformer(input_size=[shape[2], shape[3]], input_channels=1, output_channels=1, num_classes=1, class_sizes=[shape[0]],
        latent_dim=latent_dim, temporal_dim=temporal_dim, cat_dims=cat_dims, num_heads=num_heads, num_ref_points=num_ref_points,
        tr_dim=tr_dim, adversarial_hidden_dims_encoder=adversarial_hidden_dims_encoder, latent_hidden_dims_encoder=latent_hidden_dims_encoder,
        cvae_yemb_dims=cvae_yemb_dims, learn_temporal_emb=learn_temporal_emb, minT=minT, maxT=maxT, device=device, num_cont=num_cont,
        other_temporal_dims=other_temporal_dims, other_minT=other_minT, other_maxT=other_maxT, cont_model=cont_model, rec_loss=rec_loss,
        temporal_uncertainty=temporal_uncertainty, name_id=name_id)
        
        # 加载训练好的模型参数
        checkpoint = torch.load(path_model + model_name)
        model.load_state_dict(checkpoint['model'])
        epochs_old = checkpoint['epochs']
        model = model.to(device)
        model.eval()

        # 抽取重建图片的频率
        sample_freq = 50

        # 预测测试集图片文件夹
        path_img = './Pictures_pred_max/' + name_dataset + '/' + str(split_id) + '/' +str(id)
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
        
        # 取出训练集数据备用
        x_train = dataset_test.data_train
        labels_train = dataset_test.y_train
        ts_train = dataset_test.t_train
        gene_list = dataset_test.gene_list

        # 记录各个基因每个切片的平均MSE损失
        recon_loss_list = []
        # 预测测试集图片
        MSE_sum = 0       # 所有测试集图片总MSE损失
        PSNR_sum = 0
        SSIM_sum = 0
        # pixel_max_list = []
        # pixel_min_list = []
        with torch.no_grad():
            for t, (images, labels, ts) in enumerate(tqdm(test_loader)):
                T = labels.ravel().shape[0]
                input_size = T
                
                # 取出一个标签
                y = labels.ravel()[0]
                images_new = images.reshape(T, 1, shape[2], shape[3]).to(device)
                
                # 用于采样
                t_slice = ts.ravel()
                y_slice = labels.ravel()

                labels_new = labels.reshape(1, T, 1).to(device)
                ts_new = ts.reshape(1, T).to(device)
                # 将基因id转化为下标id
                y = gene_list[y.item()]

                images_old = torch.tensor(x_train[y].reshape(-1, 1, shape[2], shape[3])).to(device)
                labels_old = torch.tensor(labels_train[y].reshape(1, -1, 1)).to(device)
                ts_old = torch.tensor(ts_train[y].reshape(1, -1)).to(device)
                
                # forward函数输出
                xhat, mu, logvar, adv_cvae, adv_tr = model.generate(xs=images_old, old_ts=ts_old, new_ts=ts_new, old_ys=labels_old, new_ys=labels_new,
                batch_size=T, old_mask=None, old_other_ts=None, new_other_ts=None,
                compute_adv=False, k=1, get_zdec=False, old_uncertainties=None, new_uncertainties=None)

                # pixel_max_list.append(torch.max(xhat).item())
                # pixel_min_list.append(torch.min(xhat).item())
                
                # # 恢复
                # max = max.ravel().to(device)
                # images_new = (images_new.reshape(T, -1) * max.reshape(T, -1)).reshape(T, 1, shape[2], shape[3])
                # xhat = (xhat.reshape(T, -1) * max.reshape(T, -1)).reshape(T, 1, shape[2], shape[3])
    
                # 计算损失
                loss_dic = model.loss_fn(x=images_new, ys=labels_new, xhat=xhat, mu=mu, logvar=logvar, adv_cvae=adv_cvae, adv_tr=adv_tr, 
                beta=beta, adv_cvae_weight=adv_cvae_weight, adv_tr_weight=adv_tr_weight)
                loss_MSE_sample = model.MSE_sample(images_new, xhat)
                # 得到各部分损失
                MSE_loss = loss_dic['MSE']
                MSE_sum = MSE_sum + MSE_loss.item()*T

                # print(input_size)
                # print(loss_MSE_sample.shape)
                # print(max.shape)

                # 计算图像平均PSNR
                PSNR_temp_sum = 0
                for g in range(input_size):
                    # psnr = 10 * torch.log10(max[g] * max[g] / loss_MSE_sample[g])
                    psnr = 10 * torch.log10(1 * 1 / loss_MSE_sample[g])
                    PSNR_temp_sum = PSNR_temp_sum + psnr
                PSNR_sum = PSNR_sum + PSNR_temp_sum.item()
                
                # 计算图像平均SSIM
                ssim = ssim_calculate(images_new, xhat, window_size = 7)
                SSIM_sum = SSIM_sum + ssim.item() * input_size

                sample_list = []
                for i in range(t_slice.shape[0]):
                    if t_slice[i] in sample_id:
                        sample_list.append(i)

                img_ori_sample = images_new[sample_list]
                img_pre_sample = xhat[sample_list]
                loss_MSE_sample = loss_MSE_sample[sample_list]
                y_sample = y_slice[sample_list]
                t_sample = t_slice[sample_list]

                # 抽取部分图片查看预测效果
                if (t % sample_freq == 0):
                    # 将MSE写入.csv文件
                    for m in range(loss_MSE_sample.shape[0]):
                        row = [y_sample[m].item(), t_sample[m].item(), loss_MSE_sample[m].item()]
                        writer.writerow(row)
                    img_ori_sample = img_ori_sample.cpu().numpy()
                    img_pre_sample = img_pre_sample.cpu().numpy()
                    for j in range(loss_MSE_sample.shape[0]):
                        plt.imsave(path_img + '/' + str(y_sample[j].item()) + '_' + str(
                            t_sample[j].item()) + '_pre' + '.png', img_pre_sample[j].squeeze(0))
                        plt.imsave(path_img + '/' + str(y_sample[j].item()) + '_' + str(
                            t_sample[j].item()) + '_ori' + '.png', img_ori_sample[j].squeeze(0))

        MSE_average = MSE_sum / picture_num 
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
        # print(set(pixel_max_list))
        # print(set(pixel_min_list))
        
        MSE_Sum = MSE_Sum + MSE_average
        SSIM_Sum = SSIM_Sum + SSIM_average
        PSNR_Sum = PSNR_Sum + PSNR_average

    MSE_Average = MSE_Sum / 5.0
    SSIM_Average = SSIM_Sum / 5.0
    PSNR_Average = PSNR_Sum / 5.0
    row_average = [name_dataset, split_id, proportion, MSE_Average, SSIM_Average, PSNR_Average]
    writer_average.writerow(row_average)
        
f_data.close()
f_average.close()


