import torch
import torch.nn as nn

class Encoder(nn.Module):
    """The encoder for CVAE"""

    def __init__(self, image_H, image_W, input_dim, conv_dims, latent_dim, num_classes):
        super().__init__()

        # # 处理label
        # self.fc_class_0 = nn.Linear(num_classes[0], image_H * image_W)
        # self.fc_class_1 = nn.Linear(num_classes[1], image_H * image_W)
        # input_dim = input_dim + 2

        # 系列卷积
        padding = 1
        kernel_size = 3
        stride = 2
        convs = []
        prev_dim = input_dim
        for conv_dim in conv_dims:
            convs.append(nn.Sequential(
                nn.Conv2d(prev_dim, conv_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU()
            ))
            prev_dim = conv_dim
        self.convs = nn.Sequential(*convs)

        # 计算卷积后的图像大小
        temp_H = image_H
        temp_W = image_W
        for i in conv_dims:
            temp_H = int((temp_H + 2 * padding - kernel_size) / stride) + 1
            temp_W = int((temp_W+ 2 * padding - kernel_size) / stride) + 1

        prev_dim = temp_H * temp_W * conv_dims[-1]

        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

    def forward(self, x, image_H, image_W, batch_size): # label0: batchz_size * 2001, label1: batch_size * 69
        # 标签拼接
        # embed_class_0 = self.fc_class_0(label0).reshape(batch_size, 1, image_H, image_W)
        # embed_class_1 = self.fc_class_1(label1).reshape(batch_size, 1, image_H, image_W)
        # x = torch.cat([x, embed_class_0], dim = 1)
        # x = torch.cat([x, embed_class_1], dim = 1)

        x = self.convs(x)
        x = torch.flatten(x, start_dim=1)
        
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    """The decoder for CVAE"""

    def __init__(self, latent_dim, image_H, image_W, conv_dims, output_dim, num_classes, name_id, embedding_dim):
        super().__init__()

        # 考虑标签的加入
        latent_dim = latent_dim + 1 + embedding_dim

        # 计算卷积后的图像大小
        padding = 1
        kernel_size = 3
        stride = 2
        temp_H = image_H
        temp_W = image_W
        for i in conv_dims:
            temp_H = int((temp_H + 2 * padding - kernel_size) / stride) + 1
            temp_W = int((temp_W + 2 * padding - kernel_size) / stride) + 1

        fc_dim = temp_H * temp_W * conv_dims[-1]
        self.embedding = nn.Embedding(num_classes[0], embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, fc_dim),
            nn.ReLU()
        )
        self.conv_H = temp_H
        self.conv_W = temp_W
        de_convs = []
        prev_dim = conv_dims[-1]
        conv_dims = conv_dims[::-1]
        for conv_dim in conv_dims[1:]:
            de_convs.append(nn.Sequential(
                nn.ConvTranspose2d(prev_dim, conv_dim, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1),
                nn.ReLU()
            ))
            prev_dim = conv_dim
        self.de_convs = nn.Sequential(*de_convs)
        
        # 参数字典：
        para_dict = {
            0: [[3, 2], 2, [3, 5], 0], 
            1: [[3, 3], 2, [2, 4], 0], 
            2: [[2, 2], 2, [6, 1], 0], 
            3: [[3, 3], 2, [3, 7], 0], 
            4: [[3, 3], 2, [3, 2], 0], 
            5: [[2, 2], 2, [4, 6], 0], 
            6: [[3, 3], 2, [4, 7], 0], 
        }
        # 设置最后一层的参数

        self.pred_layer = nn.Sequential(
            nn.ConvTranspose2d(prev_dim, output_dim, kernel_size=para_dict[name_id][0], stride=para_dict[name_id][1], padding=para_dict[name_id][2], output_padding=para_dict[name_id][3])
        )

    def forward(self, x, labels): # label0: batchz_size * 2001, label1: batch_size * 69
        # 加入标签
        labels_emb = self.embedding(labels[:, 0])
        # print(labels[:, 1])
        # print(x)
        # print(labels_emb)
        x = torch.cat([x, labels_emb, labels[:, 1].unsqueeze(1)], dim = 1)
        x = self.fc(x)
        x = x.reshape(x.size(0), -1, self.conv_H, self.conv_W)
        x = self.de_convs(x)
        x = self.pred_layer(x)
        return x


class CVAE_label(nn.Module):
    """CVAE"""

    def __init__(self, image_H, image_W, input_dim, conv_dims, latent_dim, num_classes, name_id, embedding_dim):
        super().__init__()

        self.encoder = Encoder(image_H, image_W, input_dim, conv_dims, latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, image_H, image_W, conv_dims, input_dim, num_classes, name_id, embedding_dim)
        self.H = image_H
        self.W = image_W
    def sample_z(self, mu, log_var):
        """sample z by reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels, image_H, image_W, batch_size):
        mu, log_var = self.encoder(x, image_H, image_W, batch_size)
        z = self.sample_z(mu, log_var)
        recon = self.decoder(z, labels)

        # print([torch.min(recon).item(), torch.max(recon).item()])
        # print([torch.min(x).item(), torch.max(x).item()])

        # # 数据归一化
        # recon = recon.reshape(batch_size, -1)
        # max_list = torch.max(recon, 1, keepdim=True).values
        # min_list = torch.min(recon, 1, keepdim=True).values
        # recon = (recon - min_list) / (max_list - min_list)
        # recon = recon.reshape(batch_size, 1, self.H, self.W)
        
        return recon, mu, log_var

    def generate(self, z, labels):
        batch_size = z.shape[0]
        recon = self.decoder(z, labels)
        
        # # 数据归一化
        # recon = recon.reshape(batch_size, -1)
        # max_list = torch.max(recon, 1, keepdim=True).values
        # min_list = torch.min(recon, 1, keepdim=True).values
        # recon = (recon - min_list) / (max_list - min_list)
        # recon = recon.reshape(batch_size, 1, self.H, self.W)

        return recon

    def compute_loss(self, x, recon, mu, log_var, beta):
        """compute loss of VAE"""

        loss_fn = torch.nn.MSELoss(reduction = 'mean')
        # KL loss
        kl_loss = (0.5 * (log_var.exp() + mu ** 2 - 1 - log_var)).sum(1).mean()
        
        recon_loss = loss_fn(recon, x)
        loss  = recon_loss + kl_loss*beta

        return loss, kl_loss, recon_loss
        
    def MSE_ori(self, x, recon):
        loss_fn = torch.nn.MSELoss(reduction = 'mean')
        recon_loss = loss_fn(recon, x)

        return recon_loss
    
    def MSE_sample(self, x, recon):
        loss_fn = torch.nn.MSELoss(reduction = 'none')
        recon_loss = loss_fn(recon, x)

        input_size = x.shape[0]
        recon_loss = recon_loss.reshape(input_size, -1)
        recon_loss = recon_loss.mean(axis=1, keepdim=False)

        return recon_loss
