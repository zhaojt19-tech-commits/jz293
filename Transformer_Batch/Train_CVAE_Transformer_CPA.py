import sys
sys.path.append("..")
split_id = 4
if(split_id == 2):
    from Dataset_transformer_2 import *
if (split_id == 3):
    from Dataset_transformer_3 import *
if(split_id == 4):
    from Dataset_4 import *
if(split_id == 5):
    from Data.Dataset_transformer_5 import *
from CVAE_Transformer_Module import *
from tqdm import tqdm

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
        num_slice = shape[1]
        
        # 训练参数
        batch_size = 128
        epoch_num = 30
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
        learn_temporal_emb = False
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
        dataset_train = Brain_Dataset(mode='train', transform=None, name_id=name_id, seed=seed, proportion=proportion, squeeze=squeeze)
        picture_num = dataset_train.__len__()
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
            for t, (images, labels) in enumerate(tqdm(train_loader)):
                input_size = images.shape[0]
                num_all = input_size
                N = input_size
                images = images.reshape(N, 1, 1, shape[2], shape[3]).float().to(device)
                labels_double = labels.reshape(N, 2).to(device)
                labels = labels_double[:, 0].reshape(N, 1, 1)
                ts = labels_double[:, 1].reshape(N, 1)
                mask = torch.ones(input_size).reshape(input_size, 1).to(device)
                num_non_blank = [1] * input_size
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

            # loss_average = loss_sum / picture_num
            MSE_average = (MSE_sum / picture_num)
            print('After {} epoch, the average image MSE loss is {}'.format(e+1+epochs_old, MSE_average))

        if(is_continue == True):
            name_model = model_name
        else:
            name_model = 'model_' + name_dataset + '_' + str(split_id) + '_' + str(seed)  + '_c' +'.t7'

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