import sys
sys.path.append("..")
from Data.Dataset_transformer_T import *
from CVAE_Transformer_Module import *
from tqdm import tqdm
import torchvision.transforms as transforms

def initialize_random_seed(seed=0):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

name_list = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
part_list = ['full', 'brain']
shape_time_list = [(75, 70), (40, 68)]
num_gene_list = [2079, 1985]
shape_list = [(2099, 40, 75, 70), (2097, 69, 109, 89), (2088, 65, 132, 94), (2045, 40, 43, 67), (2073, 50, 43, 77), (2071, 50, 40, 68), (2079, 58, 41, 67)]
temp_list = [0, 1, 2, 3, 4, 5, 6]
id_list = [3]
proportion_list = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
name_model_list = []
split_id = 2


for name_id in range(2):
    print('开始在'+part_list[name_id]+'上训练：')
    for id in id_list:
        name = part_list[name_id]
        shape = shape_time_list[name_id]
        num_gene = num_gene_list[name_id]
        
        # 训练参数
        batch_size = 1
        epoch_num = 30
        lr = 1e-4
        is_continue = False
        squeeze = True
        proportion = 0.1
        seed = id
        device = 'cuda:1'
        initialize_random_seed(seed)

        # 模型保存
        path_model = './Checkpoint/'
        model_name = ''
        name_dataset = name
        
        # 模型参数
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
        # [128, 64, 32]
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
        num_cont = 2
        other_temporal_dims = [256]
        if(name_id == 0):
            other_minT = [10]
            other_maxT = [20]
        else:
            other_minT = [10]
            other_maxT = [80]
        cont_model = 'sinusoidal'
        # cont_model = 'sigmoidal'
        rec_loss = 'mse'
        temporal_uncertainty = False

        # 定义DataLoader
        dataset_train = Brain_Dataset(mode='train', transform=None, part=name, seed=seed, proportion=proportion, squeeze=squeeze)
        picture_num = dataset_train.get_picture_num()
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        # 定义模型
        model = CatConTransformer(input_size=[shape[0], shape[1]], input_channels=1, output_channels=1, num_classes=1, class_sizes=[num_gene],
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
            loss_sum = 0        # 所有训练集图片总损失
            MSE_sum = 0       # 所有训练集图片总MSE损失
            KLD_sum = 0         # 所有训练集图片总KLD损失
            for t, (images, labels, ts, time) in enumerate(tqdm(train_loader)):
                # y = labels[0][0].item()
                T = labels.ravel().shape[0]
                images = images.reshape(T, 1, shape[0], shape[1]).to(device)
                labels = labels.reshape(1, T, 1).to(device)
                ts = ts.reshape(1, T).to(device)
                time = time.reshape(1, T).to(device)
                optimizer.zero_grad()
                # forward函数输出
                xhat, mu, logvar, adv_cvae, adv_tr = model(xs=images, ts=ts, ys=labels, batch_size=T, mask=mask, is_label=False, other_ts=[time])
                # 计算损失
                loss_dic = model.loss_fn(x=images, ys=labels, xhat=xhat, mu=mu, logvar=logvar, adv_cvae=adv_cvae,
                adv_tr=adv_tr, beta=beta, adv_cvae_weight=adv_cvae_weight, adv_tr_weight=adv_tr_weight)
                # 得到各部分损失
                loss = loss_dic['loss']
                MSE_loss = loss_dic['MSE']
                MSE_sum = MSE_sum + MSE_loss.item()*T
                # cvae_adv_loss = loss_dic['cvae_adversary']
                # latent_adv_loss = loss_dic['latent_adversary']

                loss_sum = loss_sum + loss.item()*T
                # cvae_adv_sum = cvae_adv_sum + cvae_adv_loss.item()*T
                # latent_adv_sum = latent_adv_sum + latent_adv_loss.item()*T

                loss.backward()
                optimizer.step()

            loss_average = loss_sum / picture_num
            # cvae_adv_average = cvae_adv_sum / picture_num
            # latent_adv_average = latent_adv_sum / picture_num
            MSE_average = (MSE_sum / picture_num) 
            print('After {} epoch, the average loss is {}, the average image MSE loss is {}'.format(e+1+epochs_old,
            loss_average, MSE_average))

        if(is_continue == True):
            name_model = model_name
        else:
            name_model = 'model_' + name_dataset + '_' + str(split_id) + '_' + str(seed) + '.t7'

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