import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):

    def __init__(self, image_H, image_W, input_channels, latent_dim):
        super().__init__()

        self.H = image_H
        self.W = image_W
        self.hidden_dims_enc = [32, 64, 128, 256]

        self.kernel_enc = [3, 3, 3, 3]
        self.stride_enc = [2, 2, 2, 2]
        self.padding_enc = [1, 1, 1, 1]


        # 卷积部分
        modules = []
        for i in range(len(self.hidden_dims_enc)):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, out_channels=self.hidden_dims_enc[i],
                              kernel_size=self.kernel_enc[i],
                              stride=self.stride_enc[i], padding=self.padding_enc[i]),
                    #nn.BatchNorm2d(self.hidden_dims_enc[i]),
                    nn.ReLU())
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
    def forward(self, x, batch_size, device='cuda:0'):
        # print(x.shape)
        out = self.conv(x)
        out = torch.flatten(out, start_dim=1)
        mu = self.fc_mu(out)
        log_var = self.fc_log_var(out)
        return mu, log_var

    # 返回经过编码器卷积处理后的图像大小
    def get_shape(self):
        return self.out_height,  self.out_width

class Decoder(nn.Module):

    def __init__(self, latent_dim, H, W, output_channels, device: str = 'cuda:0', name_id=1):
        super().__init__()
        self.H = H
        self.W = W
        self.hidden_dims_enc = [32, 64, 128, 256]
        self.hidden_dims_dec = [128, 64, 32]
        self.kernel_dec = [(3, 3), (3, 3), (3, 3)]
        self.stride_dec = [(2, 2), (2, 2), (2, 2)]
        self.padding_dec = [(1, 1), (1, 1), (1, 1)]
        self.output_padding_dec = [(1, 1), (1, 1), (1, 1)]


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
                nn.ReLU())
            )
            prev = self.hidden_dims_dec[i]

        # 最终层用来调整图片大小和通道数
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
            nn.ConvTranspose2d(self.hidden_dims_dec[-1], output_channels, kernel_size=para_dict[name_id][0], stride=para_dict[name_id][1], padding=para_dict[name_id][2], output_padding=para_dict[name_id][3])
        )
        modules.append(self.pred_layer)

        self.conv = nn.Sequential(*modules)

    def forward(self, x, batch_size): 
        # 加入标签
        x = x.squeeze()
        x = self.fc(x).reshape(batch_size, self.hidden_dims_enc[-1], self.H, self.W)
        recon = self.conv(x)
        return recon


class VAE(nn.Module):

    def __init__(self, image_H, image_W, input_channels, output_channels, latent_dim, device: str='cuda:0', name_id=1):
        super().__init__()
        self.device = device
        self.encoder = Encoder(image_H, image_W, input_channels, latent_dim)
        H, W = self.encoder.get_shape()
        self.decoder = Decoder(latent_dim, H, W, output_channels, device=device, name_id=name_id)
    def encode(self, x, batch_size):
        mu, log_var = self.encoder(x, batch_size, self.device)
        return mu, log_var

    def decode(self, z, batch_size):
         recon = self.decoder(z, batch_size)
         return recon
         
    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, batch_size):
        mu, log_var = self.encoder(x, batch_size, self.device)
        z = self.sample_z(mu, log_var)
        recon = self.decoder(z, batch_size)
        return recon, mu, log_var

    def generate(self, z, batch_size):
        recon = self.decoder(z, batch_size)
        return recon

    def compute_loss(self, x, recon, mu, log_var, kl_weight):
        loss_fn = torch.nn.MSELoss(reduction = 'mean')
        # # KL loss
        # kl_loss = (0.5 * (log_var.exp() + mu ** 2 - 1 - log_var)).sum(1).mean()
        # # recon_loss = F.binary_cross_entropy(recon, x, reduction="none").sum([1, 2, 3]).mean()
        recon_loss = loss_fn(recon, x)
        # loss  = recon_loss + kl_weight * kl_loss

        return recon_loss
    
    