import sys
sys.path.append("..")
split_id = 2
if(split_id == 2):
    from Dataset_transformer_2 import *
if (split_id == 3):
    from Dataset_transformer_3 import *
if(split_id == 4):
    from Dataset_transformer_4 import *
if(split_id == 5):
    from Data.Dataset_transformer_5 import *
from CVAE_Transformer_Module import *
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

def initialize_random_seed(seed=0):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]
temp_list = [1, 2, 3, 4, 5, 6]
id_list = [0, 1, 2, 3, 4]
proportion_list = [0.005, 0.05, 0.1, 0.15, 0.2, 0.25]
name_model_list = []

for name_id in temp_list:
    print('开始在'+name_list[name_id]+'上训练：')
    for id in id_list:
        name = name_list[name_id]
        shape = shape_list[name_id]
        num_slice = shape[1]
        
        # 训练参数
        batch_size = 8
        epoch_num = 50
        lr = 1e-4
        is_continue = False
        squeeze = True
        if(split_id == 4 or split_id == 5):
            proportion = proportion_list[id]
        else:
            proportion = 0.1
        seed = id
        device = 'cuda:0'

        # 模型保存
        path_model = './Checkpoint/'
        model_name = ''
        name_dataset = name
        initialize_random_seed(seed)
        
        # 模型参数
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
        # [128, 64, 32]
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
        num_cont = None
        other_temporal_dims = None
        other_minT = None
        other_maxT = None
        cont_model = 'sinusoidal'
        # cont_model = 'sigmoidal'
        rec_loss = 'mse'
        temporal_uncertainty = False

        # 定义DataLoader
        dataset_train = Brain_Dataset(mode='train', transform=None, name_id=name_id, seed=seed, proportion=proportion, squeeze=squeeze, num_slice=num_slice)
        picture_num = dataset_train.get_picture_num()
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        # dataset_transfer = Dataset_Transfer(mode='train', transform=None, name_id=name_id, seed=seed, proportion=proportion, squeeze=squeeze, num_slice=num_slice)
        # target_num = dataset_transfer.get_picture_num()
        # transfer_loader = torch.utils.data.DataLoader(dataset_transfer, batch_size=batch_size, shuffle=True)

        # 定义模型
        model = CatConTransformer(input_size=[shape[2], shape[3]], input_channels=1, output_channels=1, num_classes=1, class_sizes=[shape[0]],
        latent_dim=latent_dim, temporal_dim=temporal_dim, cat_dims=cat_dims, num_heads=num_heads, num_ref_points=num_ref_points,
        tr_dim=tr_dim, adversarial_hidden_dims_encoder=adversarial_hidden_dims_encoder, latent_hidden_dims_encoder=latent_hidden_dims_encoder,
        cvae_yemb_dims=cvae_yemb_dims, learn_temporal_emb=learn_temporal_emb, minT=minT, maxT=maxT, device=device, num_cont=num_cont,
        other_temporal_dims=other_temporal_dims, other_minT=other_minT, other_maxT=other_maxT, cont_model=cont_model, rec_loss=rec_loss,
        temporal_uncertainty=temporal_uncertainty, name_id=name_id)


        epochs_old = 0
        if(is_continue == True):
            checkpoint = torch.load(path_model + model_name)
            model.load_state_dict(checkpoint['model'])
            epochs_old = checkpoint['epochs']
        else:
            pass

        model.to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

        loss_sum = 0        # 所有训练集图片总损失
        MSE_sum = 0       # 所有训练集图片总MSE损失
        cvae_adv_sum = 0
        latent_adv_sum = 0
        # 开始训练模型
        print('此时seed = '+str(seed))
        for e in range(epoch_num):
            # loss_sum = 0        # 所有训练集图片总损失
            MSE_sum = 0       # 所有训练集图片总MSE损失
            # KLD_sum = 0         # 所有训练集图片总KLD损失
            for t, (images, labels, ts, mask, num_non_blank) in enumerate(tqdm(train_loader)):
                # y = labels[0][0].item()
                num_all = num_non_blank.sum()
                N = images.shape[0]
                images = images.reshape(N, num_slice, 1, shape[2], shape[3]).float().to(device)
                labels = labels.reshape(N, num_slice, 1).to(device)
                ts = ts.reshape(N, num_slice).to(device)
                mask = mask.reshape(N, num_slice).to(device)
                optimizer.zero_grad()
                # forward函数输出
                xhat, mu, logvar, adv_cvae, adv_tr = model(xs=images, ts=ts, ys=labels, batch_size=N, mask=mask, is_label=False)
                # 计算损失
                loss_dic = model.loss_fn(x=images, ys=labels, xhat=xhat, mask=mask, num_all=num_all, mu=mu, logvar=logvar, adv_cvae=adv_cvae,
                adv_tr=adv_tr, beta=beta, adv_cvae_weight=adv_cvae_weight, adv_tr_weight=adv_tr_weight)
                # 得到各部分损失
                # loss = loss_dic['loss']
                MSE_loss = loss_dic['loss']
                # loss_sum = loss_sum + loss.item()*T
                MSE_sum = MSE_sum + MSE_loss.item()*num_all

                MSE_loss.backward()
                optimizer.step()
                
                # print(scheduler.get_last_lr())
                # scheduler.step()
                

            # loss_average = loss_sum / picture_num
            MSE_average = (MSE_sum / picture_num)
            print('After {} epoch, the average image MSE loss is {}'.format(e+1+epochs_old, MSE_average))

            # MSE_sum = 0       # 所有训练集图片总MSE损失

            # for t, (images_target, labels_target, ts_target, mask_target, num_non_blank_target, images_source, labels_source, ts_source, mask_source, num_non_blank_source) in enumerate(tqdm(transfer_loader)):
            #     num_all = num_non_blank_target.sum()

            #     N = images_target.shape[0]
            #     images_new = images_target.reshape(N, num_slice, 1, shape[2], shape[3]).float().to(device)
            #     labels_new = labels_target.reshape(N, num_slice, 1).to(device)
            #     ts_new = ts_target.reshape(N, num_slice).to(device)
            #     mask_new = mask_target.reshape(N, num_slice).to(device)

            #     images_old = images_source.reshape(N, num_slice, 1, shape[2], shape[3]).float().to(device)
            #     labels_old = labels_source.reshape(N, num_slice, 1).to(device)
            #     ts_old = ts_source.reshape(N, num_slice).to(device)
            #     mask_old = mask_source.reshape(N, num_slice).to(device)
                
            #     # forward函数输出
            #     xhat, mu, logvar, adv_cvae, adv_tr = model.generate(xs=images_old, old_ts=ts_old, new_ts=ts_new, old_ys=labels_old, new_ys=labels_new,
            #     batch_size=N, old_mask=mask_old, old_other_ts=None, new_other_ts=None,
            #     compute_adv=False, k=1, get_zdec=False, old_uncertainties=None, new_uncertainties=None)
            #     # pixel_max_list.append(torch.max(xhat).item())
            #     # pixel_min_list.append(torch.min(xhat).item())
                
            #     # # 恢复
            #     # max = max.ravel().to(device)
            #     # images_new = (images_new.reshape(T, -1) * max.reshape(T, -1)).reshape(T, 1, shape[2], shape[3])
            #     # xhat = (xhat.reshape(T, -1) * max.reshape(T, -1)).reshape(T, 1, shape[2], shape[3])
    
            #     # 计算损失
            #     loss_dic = model.loss_fn(x=images_new, ys=labels_new, xhat=xhat, mask=mask_new, num_all=num_all, mu=mu, logvar=logvar, adv_cvae=adv_cvae, adv_tr=adv_tr, beta=beta, adv_cvae_weight=adv_cvae_weight, adv_tr_weight=adv_tr_weight)
            #     loss_MSE_sample = model.MSE_sample(images_new, xhat, num_non_blank_target)
            #     # 得到各部分损失
            #     MSE_loss = loss_dic['loss']
            #     MSE_sum = MSE_sum + MSE_loss.item()*num_all.item()

            #     optimizer.zero_grad()
            #     MSE_loss.backward()
            #     optimizer.step()

            # # loss_average = loss_sum / picture_num
            # MSE_average = (MSE_sum / target_num)
            # print('After {} epoch, the average image MSE loss on transfer is {}'.format(e+1+epochs_old, MSE_average))

        if(is_continue == True):
            name_model = model_name
        else:
            name_model = 'model_' + name_dataset + '_' + str(split_id) + '_' + str(seed)  + '.t7'

        state = {
            'model': model.state_dict(),
            'id': id,
            'batch_size': batch_size,
            'epochs': epoch_num + epochs_old,
            'loss_last': 0,
            'name_dataset': name_dataset,
            'Mask': mask,
            'learn_temporal_emb': learn_temporal_emb
        }
        torch.save(state, path_model + name_model)
        name_model_list.append(name_model)
    

print(name_model_list)