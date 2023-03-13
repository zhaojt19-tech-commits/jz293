import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):

    def __init__(self, image_H, image_W, input_channels, latent_dim, class_sizes):
        super().__init__()

        self.H = image_H
        self.W = image_W
        self.class_sizes = class_sizes
        self.hidden_dims_enc = [32, 64, 128, 256]
        # self.hidden_dims_enc = [16, 32, 64]
        self.kernel_enc = [3, 3, 3, 3]
        self.stride_enc = [2, 2, 2, 2]
        self.padding_enc = [1, 1, 1, 1]

        # # 将独热码形式的label映射到和图片相同大小
        self.fc_class_0 = nn.Linear(class_sizes[0], image_H * image_W)

        input_channels = input_channels + len(class_sizes)

        # 卷积部分
        modules = []
        for i in range(len(self.hidden_dims_enc)):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, out_channels=self.hidden_dims_enc[i],
                              kernel_size=self.kernel_enc[i],
                              stride=self.stride_enc[i], padding=self.padding_enc[i]),
                    # nn.BatchNorm2d(self.hidden_dims_enc[i]),
                    nn.ReLU()
                    # nn.Tanh()
                    )
            )
            input_channels = self.hidden_dims_enc[i]

        self.conv = nn.Sequential(*modules)

        # 计算卷积1后的图像大小
        self.out_width = image_W
        self.out_height = image_H
        for i in range(len(self.hidden_dims_enc)):
            self.out_width = int(np.floor(
                (self.out_width - self.kernel_enc[i] + 2 * self.padding_enc[i])
                / self.stride_enc[i])) + 1
            self.out_height = int(np.floor(
                (self.out_height - self.kernel_enc[i] + 2 * self.padding_enc[i])
                / self.stride_enc[i])) + 1
        self.fc_dim = self.hidden_dims_enc[-1] * self.out_height * self.out_width

        self.fc_mu = nn.Linear(self.fc_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.fc_dim, latent_dim)

    # 这里默认使用batch_size = 1, T根据切片数量选择
    def forward(self, x, label0, batch_size, device='cuda:0'):
        # 将标签转为独热码形式
        label_double = label0.reshape(batch_size, -1)
        label0 = label_double[:, 0].long()
        index = np.linspace(0, batch_size - 1, batch_size).astype("long")
        one_hot_0 = torch.zeros(batch_size, self.class_sizes[0]).to(device)
        one_hot_0[[index, label0]] = 1
        # print(label0)
        # print(one_hot_0)

        # 标签拼接
        embed_class_0 = self.fc_class_0(one_hot_0).reshape(batch_size, 1, self.H, self.W)
        # print(embed_class_0)
        # print(x)
        x = torch.cat([x, embed_class_0], dim = 1)
        # print(x)
        # print(x.shape)
        out = self.conv(x)
        out = torch.flatten(out, start_dim=1)
        # print(out)
        mu = self.fc_mu(out)
        log_var = self.fc_log_var(out)
        return mu, log_var

    # 返回经过编码器卷积处理后的图像大小
    def get_shape(self):
        return self.out_height,  self.out_width

class Decoder(nn.Module):

    def __init__(self, latent_dim, label_emb_dim, H, W, output_channels, class_sizes, device: str = 'cuda:0', name_id=1):
        super().__init__()
        self.class_sizes = class_sizes
        self.H = H
        self.W = W
        self.hidden_dims_enc = [32, 64, 128, 256]
        # self.hidden_dims_enc = [16, 32, 64]
        self.hidden_dims_dec = [128, 64, 32]
        # self.hidden_dims_dec = [128, 64, 32]
        self.kernel_dec = [(3, 3), (3, 3), (3, 3)]
        self.stride_dec = [(2, 2), (2, 2), (2, 2)]
        self.padding_dec = [(1, 1), (1, 1), (1, 1)]
        self.output_padding_dec = [(1, 1), (1, 1), (1, 1)]

        # 考虑标签的加入（通过embadding表的形式）
        self.label_emb = nn.Embedding(self.class_sizes[0], label_emb_dim[0]).to(device)
        # 引入标签导致的隐层维数变化
        latent_dim = latent_dim + label_emb_dim[0]

        # 将隐层映射到图像大小
        self.fc = nn.Linear(latent_dim, H*W*self.hidden_dims_enc[-1])

        # 卷积部分
        modules = []
        prev = self.hidden_dims_enc[-1]
        for i in range(len(self.hidden_dims_dec)):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(
                    prev, self.hidden_dims_dec[i],
                    kernel_size=self.kernel_dec[i],
                    stride=self.stride_dec[i],
                    padding=self.padding_dec[i],
                    output_padding=self.output_padding_dec[i]),
                # nn.BatchNorm2d(self.hidden_dims_dec[i]),
                nn.ReLU()
                # nn.Tanh()
                )
            )
            prev = self.hidden_dims_dec[i]

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

        # 最终层用来调整图片大小和通道数
        modules.append(nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims_dec[-1], output_channels,
                               kernel_size=para_dict[name_id][0], stride=para_dict[name_id][1], padding=para_dict[name_id][2], output_padding=para_dict[name_id][3])
            # nn.Sigmoid()
            )
        )
        self.conv = nn.Sequential(*modules)

    def forward(self, x, label0, batch_size): # 这里的label为正常标签
        # 加入标签
        label0 = label0.ravel()
        x = x.squeeze()
        label_emb = self.label_emb(label0)

        if(label_emb.shape[0] == 1):
            x = x.reshape(1, -1)
        # print('*****************')
        # print(x.shape)
        # print(label_emb.shape)
        x = torch.cat([x, label_emb], dim = 1)
        # x = x + label_emb
        x = self.fc(x).reshape(batch_size, self.hidden_dims_enc[-1], self.H, self.W)
        recon = self.conv(x)
        return recon


class CVAE(nn.Module):

    def __init__(self, image_H, image_W, input_channels, output_channels, latent_dim, label_emb_dim, class_sizes, device: str = 'cuda:0', name_id=1):
        super().__init__()
        self.device = device
        self.encoder = Encoder(image_H, image_W, input_channels, latent_dim, class_sizes)
        H, W = self.encoder.get_shape()
        self.decoder = Decoder(latent_dim, label_emb_dim, H, W, output_channels, class_sizes, device = device, name_id=name_id)
    def encode(self, x, label0, batch_size):
        mu, log_var = self.encoder(x, label0, batch_size, self.device)
        return mu, log_var

    def decode(self, z, label0, batch_size):
         recon = self.decoder(z, label0, batch_size)
         return recon
         
    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, label0, batch_size):
        mu, log_var = self.encoder(x, label0, batch_size, self.device)
        z = self.sample_z(mu, log_var)
        recon = self.decoder(z, label0, batch_size)
        return recon, mu, log_var

    def generate(self, z, label0, batch_size):
        recon = self.decoder(z, label0, batch_size)
        return recon

    def compute_loss(self, x, recon, mu, log_var, kl_weight):
        loss_fn = torch.nn.MSELoss(reduction = 'mean').to(self.device)
        # KL loss
        kl_loss = (0.5 * (log_var.exp() + mu ** 2 - 1 - log_var)).sum(1).mean()
        # recon_loss = F.binary_cross_entropy(recon, x, reduction="none").sum([1, 2, 3]).mean()
        recon_loss = loss_fn(recon, x)
        
        
        loss  = recon_loss + kl_weight * kl_loss

        # print([kl_weight, recon_loss, kl_loss, loss])

        return {'loss': loss,
                'MSE': recon_loss,
                'KLD': kl_loss}

    def MSE_sample(self, x, recon):
        loss_fn = torch.nn.MSELoss(reduction='none')
        recon_loss = loss_fn(recon, x)

        input_size = x.shape[0]
        recon_loss = recon_loss.reshape(input_size, -1)
        recon_loss = recon_loss.mean(axis=1, keepdim=False)

        return recon_loss