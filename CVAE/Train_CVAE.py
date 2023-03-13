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
from CVAE_label_module import *
import time
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

        # 模型保存
        path_model = './Checkpoint/'
        model_name = ''
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
        epoch_num = 20
        is_continue = False
        squeeze = True
        device = 'cuda:2'
        beta = 1
        initialize_random_seed(seed)


        dataset_train = Brain_Dataset(mode='train', transform=None, name_id=name_id, seed=seed, proportion=proportion, squeeze=squeeze)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        # 得到训练集图片数量
        picture_num = dataset_train.__len__()

        model = CVAE_label(image_H, image_W, 1, conv_dims, latent_dim, num_classes, name_id, embedding_dim)

        epochs_old = 0
        if(is_continue == True):
            checkpoint = torch.load(path_model + model_name)
            model.load_state_dict(checkpoint['model'])
            epochs_old = checkpoint['epochs']
        else:
            pass
        # 将模型复制到GPU上
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        print('此时seed = '+str(seed))
        for epoch in range(epoch_num):
            MSE_sum = 0
            kl_sum = 0
            loss_sum = 0
            for i, (images, labels) in enumerate(tqdm(train_loader)):
                # 将标签转化为独热码传入
                input_size = labels.shape[0]
                images = images.reshape(input_size, 1, image_H, image_W).to(device)
                labels = labels.to(device)
                
                xhat, mu, log_var = model(images, labels, image_H, image_W, input_size)
                loss, loss_kl, loss_MSE = model.compute_loss(images, xhat.reshape(input_size, 1, image_H, image_W), mu, log_var, beta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                MSE_sum = MSE_sum + loss_MSE.item()*input_size
                loss_sum = loss_sum + loss.item()*input_size
                kl_sum = kl_sum + loss_kl.item()*input_size

            # 计算一个epoch后平均一张图片的ems
            MSE_average = MSE_sum / picture_num
            kl_average = kl_sum / picture_num
            loss_average = loss_sum / picture_num
            print('After {} epoch, the average loss is {}, the average MSE loss is {}, the average kl loss is {}'.format(epoch+1+epochs_old, loss_average, MSE_average, kl_average))

        if(is_continue == True):
            name_model = model_name
        else:
            name_model = 'model_' + name_dataset + '_' + str(split_id) + '_' + str(seed) +'.t7'
        state = {
            'model': model.state_dict(),
            'id': id, 
            'batch_size': batch_size, 
            'epochs': epoch_num + epochs_old, 
            'loss_last': loss, 
            'name_dataset': name_dataset
        }
        torch.save(state, path_model + name_model)
        name_model_list.append(name_model)

# 打印训练好的模型名称
print(name_model_list)
    
