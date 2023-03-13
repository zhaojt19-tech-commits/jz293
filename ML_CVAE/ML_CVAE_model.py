"""
Implementation of Multilinear Latent Conditioning model from
Georgopoulos et. al. paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from torch import Tensor
import numpy as np
import tensorly as tl
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

from BaseVAE import BaseVAE
from model_building_blocks import *
# from models.priors.base_prior import BasePrior
# from models.priors.std_normal_prior import StandardNormalPrior

tl.set_backend('pytorch')


class GeorgopoulosMLCVAE(BaseVAE):
    def __init__(self,
                 builder: Builder,
                 latent_dim: int = 512,
                 img_size: List[int] = [109, 89],
                 in_channels: int = 3,
                 num_classes: int = 2,
                 class_sizes: List[int] = [10, 10],
                 hidden_dims: List[int] = [128, 128 * 2, 128 * 4],
                 decomp: str = 'Tucker',  # decomposition method, either Tucker or CP
                 k: int = 64,  # rank for decomposition
                 beta: float = 1.0,  # beta to use for KL divergence
                 distribution: str = 'gaussian',
                 device: torch.device = None,
                 **kwargs) -> None:
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_sizes = class_sizes
        self.hidden_dims = hidden_dims
        self.decomp = decomp
        self.k = k
        self.beta = beta
        self.distribution = distribution
        # self.device = device
        self.pz = Normal(torch.zeros(1, self.latent_dim).to(device),
                         torch.ones(1, self.latent_dim).to(device))

        # Use builder to construct encoder / decoder
        self.encoder = builder.get_encoder()
        self.encoder_output_dim = builder.get_encoder_output_dim(img_size[0], img_size[1])
        self.fc_mu = nn.Linear(np.prod(self.encoder_output_dim), latent_dim)
        self.fc_var = nn.Linear(np.prod(self.encoder_output_dim), latent_dim)
        self.decoder_input = nn.Linear(latent_dim, np.prod(self.encoder_output_dim))
        self.decoder = builder.get_decoder()
        self.final_layer = builder.get_final_layer()

        # Label inference
        self.label_inf = nn.ModuleList([
            nn.Linear(latent_dim, class_sizes[i], bias=True) for i in range(num_classes)])

        # Mean network
        assert num_classes >= 2, 'Must have at least two labels!'
        if self.decomp == 'Tucker':
            self.U3 = nn.Linear(class_sizes[1], class_sizes[1], bias=False)
            self.U2 = nn.Linear(class_sizes[0], class_sizes[0], bias=False)
            self.U1 = nn.Linear(self.k, latent_dim, bias=False)
            self.G = nn.Linear(class_sizes[0] * class_sizes[1], self.k, bias=False)
        else:
            assert self.decomp == 'CP'
            self.U3 = nn.Linear(class_sizes[1], self.k, bias=False)
            self.U2 = nn.Linear(class_sizes[0], self.k, bias=False)
            self.U1 = nn.Linear(self.k, latent_dim, bias=False)

        if num_classes == 3:  # add 4th order decomposition
            if self.decomp == 'Tucker':
                self.V4 = nn.Linear(class_sizes[2], class_sizes[2], bias=False)
                self.V3 = nn.Linear(class_sizes[1], class_sizes[1], bias=False)
                self.V2 = nn.Linear(class_sizes[0], class_sizes[0], bias=False)
                self.V1 = nn.Linear(self.k, latent_dim, bias=False)
                self.H = nn.Linear(class_sizes[0] * class_sizes[1] * class_sizes[2], self.k, bias=False)
            else:
                assert self.decomp == 'CP'
                self.V4 = nn.Linear(class_sizes[2], self.k, bias=False)
                self.V3 = nn.Linear(class_sizes[1], self.k, bias=False)
                self.V2 = nn.Linear(class_sizes[0], self.k, bias=False)
                self.V1 = nn.Linear(self.k, latent_dim, bias=False)

        self.W1 = nn.Linear(class_sizes[0], latent_dim, bias=False)
        self.W2 = nn.Linear(class_sizes[1], latent_dim, bias=False)
        
    # weight_init
    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()

    def compute_prior(self, cats: List[Tensor]) -> Tensor:
        assert len(cats) >= 2, 'Must have at least two labels!'
        # print(cats[1].shape)
        
        if self.decomp == 'Tucker':
            l2_m = self.U3(cats[1])
            l1_m = self.U2(cats[0])
            kr_m = tl.tenalg.khatri_rao([l2_m.t(), l1_m.t()]).t()
            g_m = self.G(kr_m)
            mean = self.U1(g_m)
        else:
            assert self.decomp == 'CP'
            l2_m = self.U3(cats[1])
            l1_m = self.U2(cats[0])
            kr_m = l2_m * l1_m
            mean = self.U1(kr_m)

        m1 = self.W1(cats[0])
        m2 = self.W2(cats[1])

        if len(cats) == 2:
            return mean + m1 + m2

        # Otherwise, do the third order as well.
        assert len(cats) == 3, 'If not two classes, must be three classes.'
        if self.decomp == 'Tucker':
            l3_m = self.V4(cats[2])
            l2_m = self.V3(cats[1])
            l1_m = self.V2(cats[0])
            kr_m = tl.tenalg.khatri_rao([l3_m.t(), l2_m.t(), l1_m.t()]).t()
            g_m = self.H(kr_m)
            mean3 = self.V1(g_m)
        else:
            assert self.decomp == 'CP'
            l3_m = self.V4(cats[2])
            l2_m = self.V3(cats[1])
            l1_m = self.V2(cats[0])
            kr_m = l3_m * l2_m * l1_m
            mean3 = self.V1(kr_m)

        return mean3 + mean + m1 + m2

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> List[Tensor]:
        input_size = z.shape[0]
        result_img = self.decoder_input(z)
        result_img = self.decoder(result_img)
        result_img = self.final_layer(result_img)
        if self.distribution == 'Bernulli':
            result_img = torch.sigmoid(result_img)
        else:
            result_img = torch.tanh(result_img)

        # # 数据归一化
        # result_img = result_img.reshape(input_size, -1)
        # max_list = torch.max(result_img, 1, keepdim=True).values
        # min_list = torch.min(result_img, 1, keepdim=True).values
        # result_img = (result_img - min_list) / (max_list - min_list)
        # result_img = result_img.reshape(input_size, 1, self.img_size[0], self.img_size[1])
        
        result_classes = [self.label_inf[i](z) for i in range(self.num_classes)]
        return [result_img, *result_classes]

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, cats: List[Tensor], **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        mu_prior = self.compute_prior(cats)
        z = self.reparameterize(mu, log_var) + mu_prior        
        reconstructions = self.decode(z)
        return [*reconstructions, input, *cats, mu, log_var, mu_prior]

    def get_reconstruction_error(self, x_hat: Tensor, x: Tensor) -> Tensor:
        # TODO(addiewc): Can also play with binary cross entropy!
        return F.mse_loss(x_hat, x, reduction='sum')

    def get_class_reconstruction_error(self,
                                       cats_hat: List[Tensor],
                                       cats: List[Tensor]) -> List[Tensor]:
        # TODO(addiewc): Can also play with nn.CrossEntropyLoss
        return [F.cross_entropy(cats_hat[i], torch.argmax(cats[i], dim=1), reduction='sum')
                for i in range(self.num_classes)]

    def loss_function(self, *args) -> dict:
        recons_img = args[0]
        recons_cats = args[1:1 + self.num_classes]
        input = args[1 + self.num_classes]
        cats = args[2 + self.num_classes: 2 + 2 * self.num_classes]
        mu = args[2 + 2 * self.num_classes]
        log_var = args[3 + 2 * self.num_classes]
        mu_prior = args[4 + 2 * self.num_classes]

        N = input.shape[0]
        # use MLC VAE loss function!
        log_px = F.mse_loss(recons_img, input, reduction='none')
        log_px = log_px.view(N, -1).mean(1, keepdim=True)

        qz = Normal(mu, (0.5 * log_var).exp())
        kl_qp = kl_divergence(qz, self.pz)
        kl_qp = kl_qp.view(N, -1).sum(1, keepdim=True)

        c0 = torch.argmax(cats[0], dim=-1)
        c1 = torch.argmax(cats[1], dim=-1)
        if len(cats) == 2:
            lab_loss = nn.CrossEntropyLoss(reduction='none')(recons_cats[0], c0) \
                       + nn.CrossEntropyLoss(reduction='none')(recons_cats[1], c1)
        else:
            assert len(cats) == 3
            lab_loss = nn.CrossEntropyLoss(reduction='none')(recons_cats[0], c0) \
                       + nn.CrossEntropyLoss(reduction='none')(recons_cats[1], c1) \
                       + nn.MSELoss(reduction='none')(recons_cats[2], cats[2])  # this is our continuous variable

        lab_loss = lab_loss.view(N, -1).sum(1, keepdim=True)
        elbo = -log_px - self.beta * kl_qp
        loss = (-elbo).mean() + self.beta * lab_loss.mean()

        return {'loss': loss, 'Reconstruction_Loss_Img': log_px.mean(),
                'Reconstruction_Loss_Classes': lab_loss.mean(), 'KLD': kl_qp.mean()}
    # 生成未知标签图片时使用
    def sample(self, num_samples: int, current_device: int, cats: List[Tensor],
               **kwargs) -> Tensor:
        M_cats = self.compute_prior(cats)
        z = M_cats + torch.randn(num_samples, self.latent_dim).to(current_device)
        samples = self.decode(z)
        return samples[0]
    # 重构测试时使用，输入的标签应该与图像一致
    def generate(self, x: Tensor, cats_old: List[Tensor], cats_new: List[Tensor],
                 **kwargs) -> Tensor:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var) + self.compute_prior(cats_new)

        reconstructions = self.decode(z)
        return reconstructions[0], reconstructions[1:]

    # 检查代码逻辑性时使用，返回隐层各变量
    def get_latent_representation(self, x: Tensor, cats: List[Tensor], **kwargs) -> Tensor:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var) + self.compute_prior(cats)
        return z, mu, log_var


class SquareConvolutionalGeorgopoulosMLCVAE(GeorgopoulosMLCVAE):
    def __init__(self,
                 name_id = 1,
                 device = 'cuda:1', 
                 img_size: List[int] = [109, 89],
                 in_channels: int = 3,
                 num_classes: int = 2,
                 class_sizes: List[int] = [10, 10],
                 latent_dim: int = 512,
                 hidden_dims: List[int] = None,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 output_padding: int = 0,
                 decomp: str = 'Tucker',
                 k: int = 64,
                 beta: float = 1.0,
                 distribution: str = 'gaussian',
                 use_ode2cvae_arch: bool = True,
                 **kwargs) -> None:
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        if not use_ode2cvae_arch:
            builder = SquareConvolutionalBuilder(
                in_channels=in_channels, hidden_dims=hidden_dims,
                kernel_size=kernel_size, stride=stride, padding=padding,
                output_padding=output_padding, out_channels=in_channels,
            )
        else:
            builder = ODE2VAEBuilderFromArch(name_id=name_id, 
                in_channels=in_channels, out_channels=in_channels,
                num_classes=0  # don't include in encoder
            )

        super().__init__(builder, latent_dim, img_size, in_channels,
                         num_classes, class_sizes, hidden_dims, decomp=decomp, k=k, device=device, **kwargs)

    def decode(self, z: Tensor) -> List[Tensor]:
        result_img = self.decoder_input(z)
        result_img = result_img.reshape(-1, *self.encoder_output_dim)
        result_img = self.decoder(result_img)
        result_img = self.final_layer(result_img)
        result_classes = [self.label_inf[i](z) for i in range(self.num_classes)]
        return [result_img, *result_classes]
    
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



# class SquareLinearGeorgopoulosMLCVAE(GeorgopoulosMLCVAE):
#     def __init__(self,
#                  img_size: List[int] = [109, 89],
#                  in_channels: int = 3,
#                  num_classes: int = 2,
#                  class_sizes: List[int] = [10, 10],
#                  latent_dim: int = 512,
#                  hidden_dims: List[int] = None,
#                  decomp: str = 'Tucker',
#                  k: int = 64,
#                  beta: float = 1.0,
#                  distribution: str = 'gaussian',
#                  **kwargs) -> None:
#         if hidden_dims is None:
#             hidden_dims = [512, 256]
#         builder = LinearBuilder(
#             input_size=in_channels * img_size[0] * img_size[1], hidden_dims=hidden_dims)

#         super().__init__(builder, latent_dim, img_size, in_channels, num_classes,
#                          class_sizes, hidden_dims, decomp=decomp, k=k, **kwargs)

#     def encode(self, input: Tensor) -> List[Tensor]:
#         result = torch.flatten(input, start_dim=1)
#         return super().encode(result)

#     def loss_function(self, *args, **kwargs) -> dict:
#         image = args[1 + self.num_classes]
#         return super().loss_function(
#             *args[:1 + self.num_classes],
#             image.view(-1, self.in_channels * self.img_size[0] * img_size[1]),
#             *args[2 + self.num_classes:], **kwargs)


# class SequenceLinearGeorgopoulosMLCVAE(GeorgopoulosMLCVAE):
#     def __init__(self,
#                  seq_length: int = 64,
#                  num_classes: int = 2,
#                  class_sizes: List[int] = [10, 10],
#                  latent_dim: int = 512,
#                  hidden_dims: List[int] = None,
#                  decomp: str = 'Tucker',
#                  k: int = 64,
#                  beta: float = 1.0,
#                  distribution: str = 'gaussian',
#                  **kwargs) -> None:
#         if hidden_dims is None:
#             hidden_dims = [512, 256]
#         builder = LinearBuilder(input_size=seq_length, hidden_dims=hidden_dims)
#         super().__init__(builder, latent_dim, seq_length, in_channels=None, num_classes=num_classes,
#                          class_sizes=class_sizes, hidden_dims=hidden_dims, decomp=decomp, k=k, **kwargs)

