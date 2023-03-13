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
from tqdm import tqdm
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from CPA_Module import *

def initialize_random_seed(seed=0):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]

temp_list = [0, 1, 2, 3, 4, 5, 6]
id_list = [0, 1, 2, 3, 4, 5]
proportion_list = [0.005, 0.05, 0.1, 0.15, 0.2, 0.25]

name_model_list = []
for name_id in temp_list:
    print('开始在'+name_list[name_id]+'上训练：')
    for id in id_list:
        name = name_list[name_id]
        shape = shape_list[name_id]

        batch_size = 128
        epoch_num = 30
        seed = id
        lr = 1e-4
        is_continue = False
        device = 'cuda:0'
        if(split_id == 4 or split_id == 5):
            proportion = proportion_list[id]
        else:
            proportion = 0.1
        squeeze = True
        initialize_random_seed(seed)

        # 模型保存
        path_model = './Checkpoint/'
        model_name = ''
        name_dataset = name
        
        num_classes = [shape[0], shape[1]]
        
        # 定义DataLoader
        dataset_train = Brain_Dataset(mode='train', name_id=name_id, seed=seed, proportion=proportion, squeeze=squeeze)
        picture_num = dataset_train.__len__()
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        
        # 定义模型
        model = ComPert(
                image_H=shape[2], 
                image_W=shape[3], 
                input_channels=1,
                output_channels=1, 
                device=device,
                seed=0,
                patience=10,
                doser_type='logsigm',
                hparams="", 
                name_id=name_id, 
                shape=shape)


        epochs_old = 0
        if(is_continue == True):
            checkpoint = torch.load(path_model + model_name)
            model.load_state_dict(checkpoint['model'])
            epochs_old = checkpoint['epochs']
        else:
            pass

        model.to(device)
        model.train()
        
        # 所有训练集图片总MSE损失
        MSE_sum = 0       
        # 开始训练模型
        for e in range(epoch_num):
            MSE_sum = 0       # 所有训练集图片总MSE损失
            loss_list = []
            loss_adversary_list = []
            loss_recon_list = []
            loss_adv_drugs_list = []
            loss_adv_cell_types_list = []
            penalty_adv_drugs_list = []
            penalty_adv_cell_types_list = []
            
            for t, (images, labels) in enumerate(tqdm(train_loader)):
                input_size = labels.shape[0]
                # 计算损失, 得到恢复后的压缩图片
                loss_dic, xhat = model.update(images, labels, input_size, num_classes)

                # 得到各部分损失
                loss = loss_dic['loss']
                loss_adversary = loss_dic['loss_adversary']
                loss_recon = loss_dic["loss_reconstruction"]
                loss_adv_drugs = loss_dic['loss_adv_drugs']
                # loss_adv_cell_types = loss_dic['loss_adv_cell_types']
                penalty_adv_drugs = loss_dic['penalty_adv_drugs']
                # penalty_adv_cell_types = loss_dic['penalty_adv_cell_types']

                # 将非零损失存入list
                if loss:
                    loss_list.append(loss)
                if loss_adversary:
                    loss_adversary_list.append(loss_adversary)
                if loss_recon:
                    loss_recon_list.append(loss_recon)
                if loss_adv_drugs:
                    loss_adv_drugs_list.append(loss_adv_drugs)
                # if loss_adv_cell_types:
                #     loss_adv_cell_types_list.append(loss_adv_cell_types)
                if penalty_adv_drugs:
                    penalty_adv_drugs_list.append(penalty_adv_drugs)
                # if penalty_adv_cell_types:
                #     penalty_adv_cell_types_list.append(penalty_adv_cell_types)

                with torch.no_grad():
                    images = images.to(device)
                    loss_MSE = model.MSE_ori(images.reshape(input_size, 1, shape[2], shape[3]), xhat.reshape(input_size, 1, shape[2], shape[3]))
                    MSE_sum = MSE_sum + loss_MSE.item()*input_size

            MSE_average = MSE_sum / picture_num 

            # 开始计算各部分损失的平均值
            loss_list = np.array(loss_list)
            loss_adversary_list = np.array(loss_adversary_list)
            loss_recon_list = np.array(loss_recon_list)
            loss_adv_drugs_list = np.array(loss_adv_drugs_list)
            # loss_adv_cell_types_list = np.array(loss_adv_cell_types_list)
            penalty_adv_drugs_list = np.array(penalty_adv_drugs_list)
            # penalty_adv_cell_types_list = np.array(penalty_adv_cell_types_list)

            loss_average = np.mean(loss_list)
            loss_adversary_average = np.mean(loss_adversary_list)
            loss_recon_average = np.mean(loss_recon_list)
            loss_adv_drugs_average = np.mean(loss_adv_drugs_list)
            # loss_adv_cell_types_average = np.mean(loss_adv_cell_types_list)
            penalty_adv_drugs_average = np.mean(penalty_adv_drugs_list)
            # penalty_adv_cell_types_average = np.mean(penalty_adv_cell_types_list)

            print('After {} epoch, the average image MSE loss is {}'.format(e+1+epochs_old, MSE_average))
            print('loss_average = {}, loss_adversary_average = {}, loss_recon_average = {}, loss_adv_drugs_average = {}, penalty_adv_drugs_average = {}'.format(
                loss_average, 
                loss_adversary_average,
                loss_recon_average,
                loss_adv_drugs_average,
                penalty_adv_drugs_average
                )
            )

        if(is_continue == True):
            name_model = model_name
        else:
            name_model = 'model_' + name_dataset + '_' + str(split_id) + '_' + str(seed) + '_t' + '.t7'
        state = {
            'model': model.state_dict(),
            'id': seed, 
            'batch_size': batch_size, 
            'epochs': epoch_num + epochs_old, 
            'loss_last': loss, 
            'name_dataset': name_dataset
        }
        torch.save(state, path_model + name_model)
        name_model_list.append(name_model)

    

print(name_model_list)