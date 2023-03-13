import sys
sys.path.append("..")
from Dataset_transformer_2 import Brain_Dataset as Dataset_2
from Dataset_transformer_3 import Brain_Dataset as Dataset_3
from Dataset_transformer_4 import Brain_Dataset as Dataset_4
from Data.SSIM import *
import matplotlib.pyplot as plt
from CVAE_Transformer_Module import *
from tqdm import tqdm
import csv
import shutil
import json
import random

split_list = [3]
name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]
temp_list = [2]
id_list = [0, 1, 2, 3, 4]
id_list_4 = [0, 1, 2, 3, 4, 5]
proportion_list = [0.005, 0.05, 0.1, 0.15, 0.2, 0.25]
distance_list = [1, 2, 3, 4, 5, 6]
name_model_list = []

def initialize_random_seed(seed=0):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_seed_image(xs, num_non_blank_old, num_non_blank_new, ts_old, ts_new, distance):
    N, num_slice, _, H, W = xs.shape

    # print(num_non_blank_new)
    # print(num_non_blank_old)

    # print(ts_old.shape)
    # print(ts_new.shape)
    
    img_select_list = []
    for i in range(N):
        ts_new_i = ts_new[i][0:num_non_blank_new[i]].reshape(num_non_blank_new[i], 1)
        ts_old_i = ts_old[i][0:num_non_blank_old[i]].reshape(1, num_non_blank_old[i])

        ts_matrix_i = torch.repeat_interleave(ts_new_i, num_non_blank_old[i], dim=1)

        # print(ts_new_i)
        # print(ts_old_i)
        # print(ts_matrix_i)

        dis_matrix_i = (ts_matrix_i - ts_old_i) ** 2

        sorted, indices = torch.sort(dis_matrix_i, dim=1, descending=False)
        
        # print(indices)

        id_select = []
        for n in range(num_non_blank_new[i]):
            j = 1
            k = 0
            while(j < distance and sorted.shape[1] > 1):
                if(k + 1 >= sorted.shape[1]):
                    break
                k = k + 1
                if(sorted[n][k] > sorted[n][k-1]):
                    j = j + 1
            index_select = indices[n][k].item()
            id_select.append(index_select)

        id_select = torch.tensor(id_select)

        # print(id_select)

        img_select_non_blank = xs[i][id_select].reshape(id_select.shape[0], 1, H, W).cpu().numpy()
        img_blank = np.zeros((num_slice-id_select.shape[0], H*W)).reshape(num_slice-id_select.shape[0], 1, H, W)
        img_select_list.append(np.concatenate((img_select_non_blank, img_blank), axis=0))

    img_select_list = torch.tensor(np.array(img_select_list)).float()

    # print(img_select_list.shape)

    return img_select_list
    
# # 定义存储字典
# fig = {}
# fig['fig2'] = {'MSE':{}, 'SSIM':{}, 'PSNR':{}}
# fig['fig3'] = {'MSE':{}, 'SSIM':{}, 'PSNR':{}}
# fig['fig4'] = {'MSE':{'E11.5':{}, 'E13.5':{}, 'E15.5':{}, 'E18.5':{}, 'P4':{}, 'P14':{}, 'P56':{}}, 
#                'SSIM':{'E11.5':{}, 'E13.5':{}, 'E15.5':{}, 'E18.5':{}, 'P4':{}, 'P14':{}, 'P56':{}}, 
#                'PSNR':{'E11.5':{}, 'E13.5':{}, 'E15.5':{}, 'E18.5':{}, 'P4':{}, 'P14':{}, 'P56':{}}
#               }
# for name in name_list:
#     for prop in proportion_list:
#                 fig['fig4']['MSE'][name][prop] = []
#                 fig['fig4']['SSIM'][name][prop] = []
#                 fig['fig4']['PSNR'][name][prop] = [] 

with open('./data_transformer.json', 'r') as fp:
    fig = json.load(fp)
fp.close()

# for name in name_list:
#     fig['fig3']['MSE'][name] = []
#     fig['fig3']['SSIM'][name] = []
#     fig['fig3']['PSNR'][name] = [] 

# path_data = './data_pred.csv'
# if os.path.exists(path_data):
#     # os.remove(path_data)
#     f_data =  open(path_data, 'a', encoding='UTF8', newline='')
#     writer_data = csv.writer(f_data)
# else:
#     header = ['name_dataset', 'split_id', 'seed', 'proportion', 'MSE', 'SSIM', 'PSNR']
#     f_data =  open(path_data, 'w', encoding='UTF8', newline='')
#     writer_data = csv.writer(f_data)
#     writer_data.writerow(header)

# path_average = './average_pred.csv'
# if os.path.exists(path_average):
#     # os.remove(path_average)
#     f_average =  open(path_average, 'a', encoding='UTF8', newline='')
#     writer_average = csv.writer(f_average)
# else:
#     header = ['name_dataset', 'split_id', 'proportion', 'MSE', 'SSIM', 'PSNR']
#     f_average =  open(path_average, 'w', encoding='UTF8', newline='')
#     writer_average = csv.writer(f_average)
#     writer_average.writerow(header)


for split_id in split_list:
    # split_id = 4
    for name_id in temp_list:
        name = name_list[name_id]
        shape = shape_list[name_id]
        num_slice = shape[1]

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
            
            # for id in range(num_id):
            for id in id_list:

                model_name = 'model_' + name + '_' + str(split_id) + '_' + str(id) +'.t7'
                path_model = './Checkpoint/'
                batch_size = 4
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
                cat_dims = [128]
                # muti-head的数量
                num_heads = 16
                num_ref_points = 21
                tr_dim = 512
                adversarial_hidden_dims_encoder = None
                latent_hidden_dims_encoder = None
                # CVAE类别embedding向量维数
                cvae_yemb_dims = [128]
                # TODO 待考虑是否学习（仔细看代码）
                learn_temporal_emb = True
                # # TODO 待考虑是否使用mask
                # mask = None
                # 损失函数中KLD损失的系数
                beta = 0
                # 标签重构损失
                adv_cvae_weight=0
                # 标签重构损失
                adv_tr_weight=0
                mask= None
                minT = 10
                maxT = 50
                device = 'cuda:0'
                num_cont = None
                other_temporal_dims = None
                other_minT = None
                other_maxT = None
                cont_model = 'sinusoidal'
                rec_loss = 'mse'
                temporal_uncertainty = False
                
                if(split_id == 2):
                    dataset_test = Dataset_2(mode='test', name_id=name_id, seed=seed, proportion=proportion, squeeze=squeeze, num_slice=num_slice)
                if(split_id == 3):
                    dataset_test = Dataset_3(mode='test', name_id=name_id, seed=seed, proportion=proportion, squeeze=squeeze, num_slice=num_slice)
                if(split_id == 4):
                    dataset_test = Dataset_4(mode='test', name_id=name_id, seed=seed, proportion=proportion, squeeze=squeeze, num_slice=num_slice)
                
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

                # # 抽取重建图片的频率
                # sample_freq = 50

                # # 预测测试集图片文件夹
                # path_img = './Pictures_pred_max/' + name_dataset + '/' + str(split_id) + '/' +str(id)
                # if not os.path.exists(path_img):
                #     os.makedirs(path_img)
                # else:
                #     shutil.rmtree(path_img)
                #     os.makedirs(path_img)

                # if not os.path.exists('./CSV_pred/'+ name_dataset+ '/'):
                #     os.makedirs('./CSV_pred/'+ name_dataset+ '/')
                # path_csv = './CSV_pred/' + name_dataset + '/' + str(split_id) + '_' + str(id) + '.csv'
                # if os.path.exists(path_csv):
                #     os.remove(path_csv)
                # header = ['gene_id', 'slice_id', 'MSE', 'SSIM', 'PSNR']
                # f_csv =  open(path_csv, 'w', encoding='UTF8', newline='')
                # writer = csv.writer(f_csv)
                # writer.writerow(header)

                # # 抽取的切片id
                # id_low = int((shape[1] / 2) - 5)
                # id_high = int((shape[1] / 2) + 5)
                # sample_id = [i for i in range(id_low, id_high, 1)]
                
                # 取出训练集数据备用
                x_train = dataset_test.data_train
                labels_train = dataset_test.y_train
                ts_train = dataset_test.t_train
                mask_train = dataset_test.mask_train
                gene_list = dataset_test.gene_list
                num_non_blank_train = dataset_test.num_non_blank_train

                # 记录各个基因每个切片的平均MSE损失
                recon_loss_list = []
                # 预测测试集图片
                MSE_sum = 0       # 所有测试集图片总MSE损失
                PSNR_sum = 0
                SSIM_sum = 0
                # pixel_max_list = []
                # pixel_min_list = []
                with torch.no_grad():
                    for t, (images, labels, ts, mask, num_non_blank) in enumerate(tqdm(test_loader)):
                        num_all = num_non_blank.sum()

                        N = images.shape[0]
                        images_new = images.reshape(N, num_slice, 1, shape[2], shape[3]).float().to(device)
                        labels_new = labels.reshape(N, num_slice, 1).to(device)
                        ts_new = ts.reshape(N, num_slice).to(device)
                        mask_new = mask.reshape(N, num_slice).to(device)
                        
                        y_query = labels_new.reshape(N, -1)[:, 0].ravel().cpu().numpy()
                        
                        # # 用于采样
                        # t_slice = ts.ravel()
                        # y_slice = labels.ravel()

                        # 将基因id转化为下标id
                        y = []
                        for i in y_query:
                            y.append(gene_list[i])
                        y = np.array(y)

                        images_old = torch.tensor(x_train[y].reshape(N, num_slice, 1, shape[2], shape[3])).float().to(device)
                        labels_old = torch.tensor(labels_train[y].reshape(N, num_slice, 1)).to(device)
                        ts_old = torch.tensor(ts_train[y].reshape(N, num_slice)).to(device)
                        mask_old = torch.tensor(mask_train[y].reshape(N, num_slice)).to(device)
                        num_non_blank_old = torch.tensor(num_non_blank_train[y].ravel()).to(device)
                        
                        # 得到生成种子
                        img_seed = get_seed_image(images_old, num_non_blank_old, num_non_blank, ts_old, ts_new, distance).to(device)
                        
                        # forward函数输出
                        xhat, mu, logvar, adv_cvae, adv_tr = model.generate(xs=images_old, xs_seed=img_seed, old_ts=ts_old, new_ts=ts_new, old_ys=labels_old, new_ys=labels_new,
                        batch_size=N, old_mask=mask_old, old_other_ts=None, new_other_ts=None,
                        compute_adv=False, k=5, get_zdec=False, old_uncertainties=None, new_uncertainties=None)
                        # pixel_max_list.append(torch.max(xhat).item())
                        # pixel_min_list.append(torch.min(xhat).item())
                        
                        # # 恢复
                        # max = max.ravel().to(device)
                        # images_new = (images_new.reshape(T, -1) * max.reshape(T, -1)).reshape(T, 1, shape[2], shape[3])
                        # xhat = (xhat.reshape(T, -1) * max.reshape(T, -1)).reshape(T, 1, shape[2], shape[3])
            
                        # 计算损失
                        loss_dic = model.loss_fn(x=images_new, ys=labels_new, xhat=xhat, mask=mask_new, num_all=num_all, mu=mu, logvar=logvar, adv_cvae=adv_cvae, adv_tr=adv_tr, beta=beta, adv_cvae_weight=adv_cvae_weight, adv_tr_weight=adv_tr_weight)
                        loss_MSE_sample = model.MSE_sample(images_new, xhat, num_non_blank.ravel())
                        # 得到各部分损失
                        MSE_loss = loss_dic['loss']
                        MSE_sum = MSE_sum + MSE_loss.item()*num_all.item()

                        # print(input_size)
                        # print(loss_MSE_sample.shape)
                        # print(max.shape)

                        # 计算图像平均PSNR
                        PSNR_temp_sum = 0
                        for g in range(num_all):
                            psnr = 10 * torch.log10(1 * 1 / loss_MSE_sample[g])
                            PSNR_temp_sum = PSNR_temp_sum + psnr
                        PSNR_sum = PSNR_sum + PSNR_temp_sum.item()
                        
                        # 计算图像平均SSIM
                        images = images_new.reshape(N, num_slice, -1)
                        xhat = xhat.reshape(N, num_slice, -1)
                        num_non_blank = num_non_blank.ravel()

                        images_non_blank = []
                        for n in range(N):
                            images_non_blank.append(images[n][0: num_non_blank[n]])
                        images_non_blank = torch.cat(images_non_blank).reshape(num_all, 1, shape[2], shape[3])

                        xhat_non_blank = []
                        for n in range(N):
                            xhat_non_blank.append(xhat[n][0: num_non_blank[n]])
                        xhat_non_blank = torch.cat(xhat_non_blank).reshape(num_all, 1, shape[2], shape[3])

                        ssim = ssim_calculate(images_non_blank, xhat_non_blank, window_size = 7)
                        SSIM_sum = SSIM_sum + ssim.item() * num_all.item()

                        # sample_list = []
                        # for i in range(t_slice.shape[0]):
                        #     if t_slice[i] in sample_id:
                        #         sample_list.append(i)

                        # img_ori_sample = images_new[sample_list]
                        # img_pre_sample = xhat[sample_list]
                        # loss_MSE_sample = loss_MSE_sample[sample_list]
                        # y_sample = y_slice[sample_list]
                        # t_sample = t_slice[sample_list]

                        # print(sample_list)
                        # print(img_pre_sample.shape)
                        # print(img_ori_sample.shape)
                        

                        # # 抽取部分图片查看预测效果
                        # if (t % sample_freq == 0):
                        #     # 将MSE写入.csv文件
                        #     for m in range(loss_MSE_sample.shape[0]):
                        #         row = [y_sample[m].item(), t_sample[m].item(), loss_MSE_sample[m].item()]
                        #         writer.writerow(row)
                        #     img_ori_sample = img_ori_sample.cpu().numpy()
                        #     img_pre_sample = img_pre_sample.cpu().numpy()
                        #     for j in range(loss_MSE_sample.shape[0]):
                        #         plt.imsave(path_img + '/' + str(y_sample[j].item()) + '_' + str(
                        #             t_sample[j].item()) + '_pre' + '.png', img_pre_sample[j].squeeze(0))
                        #         plt.imsave(path_img + '/' + str(y_sample[j].item()) + '_' + str(
                        #             t_sample[j].item()) + '_ori' + '.png', img_ori_sample[j].squeeze(0))

                MSE_average = MSE_sum / picture_num 
                PSNR_average = PSNR_sum / picture_num
                SSIM_average = SSIM_sum / picture_num

                if(split_id == 4):
                    fig['fig4']['MSE'][name][str(proportion)].append(MSE_average)
                    fig['fig4']['SSIM'][name][str(proportion)].append(SSIM_average)
                    fig['fig4']['PSNR'][name][str(proportion)].append(PSNR_average)
                with open('data_transformer.json', 'w') as fp:
                    json.dump(fig, fp)
                fp.close()

                print(name_dataset, end =' ')
                print(str(seed)+':')
                print('After {} epoch, when the fig is {}, the distance is {}, the average Pixel MSE loss is {}, PSNR is {}, SSIM is {}'.format(epochs_old, split_id, distance, MSE_average, PSNR_average, SSIM_average))
                # f_csv.close()

                # row_data = [name_dataset, split_id, seed, proportion, MSE_average, SSIM_average, PSNR_average]
                # writer_data.writerow(row_data)
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
            if(split_id == 3):
                fig['fig3']['MSE'][name].append(MSE_Average)
                fig['fig3']['SSIM'][name].append(SSIM_Average)
                fig['fig3']['PSNR'][name].append(PSNR_Average)

            with open('data_transformer.json', 'w') as fp:
                json.dump(fig, fp)
            fp.close()

            # MSE_Average_list.append(MSE_Average)
            # SSIM_Average_list.append(SSIM_Average)
            # PSNR_Average_list.append(PSNR_Average)
            # row_average = [name_dataset, split_id, proportion, MSE_Average, SSIM_Average, PSNR_Average]
            # print(MSE_Average)
            # writer_average.writerow(row_average)
        # if(split_id == 2):
        #     fig['fig2']['MSE'][name] = MSE_Average_list
        #     fig['fig2']['SSIM'][name] = SSIM_Average_list
        #     fig['fig2']['PSNR'][name] = PSNR_Average_list
        # if(split_id == 3):
        #     fig['fig3']['MSE'][name] = MSE_Average_list
        #     fig['fig3']['SSIM'][name] = SSIM_Average_list
        #     fig['fig3']['PSNR'][name] = PSNR_Average_list
        
# f_data.close()
# f_average.close()
with open('data_transformer.json', 'w') as fp:
    json.dump(fig, fp)
fp.close()
