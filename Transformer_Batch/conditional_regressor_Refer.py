import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import *

class cVAE_Sequence(nn.Module):
    def __init__(self,
                 input_size: int,
                 latent_dim: int,
                 num_classes: int,
                 class_sizes: List[int],
                 hidden_dims: List[int],
                 device: str = 'cuda:0',
                 kaiming_init: bool = False) -> None:
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.class_sizes = class_sizes
        self.latent_dim = latent_dim
        self.device = device
        self.hidden_dims = hidden_dims  # need this for ODE interaction!

        prev_d = input_size + int(np.sum(class_sizes))
        enc_layers = []
        for i in range(len(hidden_dims)):
            enc_layers.append(nn.Sequential(nn.Linear(prev_d, hidden_dims[i]), nn.ReLU()))
            if kaiming_init:
                nn.init.kaiming_normal_(enc_layers[-1][0].weight,
                                        mode='fan_in')  # use kaiming initialization for linear + relu
            prev_d = hidden_dims[i]
        self.encoder = nn.Sequential(*enc_layers)
        self.mu_fc = nn.Linear(prev_d, latent_dim)
        self.var_fc = nn.Linear(prev_d, latent_dim)

        prev_d = latent_dim + int(np.sum(class_sizes))
        dec_layers = []
        for i in range(len(hidden_dims) - 1, -1, -1):  # go backwards through hidden dims
            dec_layers.append(nn.Sequential(nn.Linear(prev_d, hidden_dims[i]), nn.ReLU()))
            if kaiming_init:
                nn.init.kaiming_normal_(dec_layers[-1][0].weight, mode='fan_in')
            prev_d = hidden_dims[i]
        dec_layers.append(nn.Sequential(nn.Linear(prev_d, input_size)))  # , nn.ReLU()))  # output is 1-d
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: Tensor) -> List[Tensor]:
        x = self.encoder(x)
        return self.mu_fc(x), self.var_fc(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def reparameterize(self, mu: Tensor, log_var: Tensor, **kwargs) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        if 'return_eps' in kwargs and kwargs['return_eps']:
            return eps * std + mu, eps
        return eps * std + mu

    def forward(self, y: Tensor, cats: List[Tensor], **kwargs) -> List[Tensor]:
        if self.num_classes > 0:
            x = torch.cat([y, *cats], dim=1)
        else:
            x = y
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var, **kwargs)
        if self.num_classes > 0:
            z = torch.cat([z, *cats], dim=1)
        pred_y = self.decode(z)

        return pred_y, y, mu, log_var

    def loss_fn(self, *args, **kwargs) -> dict:
        pred_y = args[0]
        y = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['beta']

        if 'rec_loss' in kwargs:
            rec_loss_type = kwargs['rec_loss']
        else:
            rec_loss_type = 'mse'
        assert rec_loss_type in {'mse', 'bce'}, "Unrecognized loss {}".format(rec_loss_type)

        if rec_loss_type == 'mse':
            recons_y_loss = F.mse_loss(pred_y, y, reduction='mean')
        else:
            M = pred_y.shape[-1]
            # clamp pred_y to between [0, 1]
            pred_y = torch.clamp(pred_y, min=0, max=1)
            recons_y_loss = F.binary_cross_entropy(pred_y.view(-1, M), y.view(-1, M), reduction='mean')

        kld_loss = torch.mean(-0.5 * torch.sum(
            1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_y_loss + kld_weight * kld_loss
        return {'loss': loss, 'MSE': recons_y_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, cats: List[Tensor], **kwargs) -> Tensor:
        z = torch.rand(size=(num_samples, self.latent_dim))
        z = torch.cat([z, *cats])
        return self.decode(z)

    def generate(self, y: Tensor, old_cats: List[Tensor], new_cats: List[Tensor], **kwargs) -> Tensor:
        if self.num_classes > 0:
            x = torch.cat([y, *old_cats], dim=1)
        else:
            x = y
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        if self.num_classes > 0:
            z = torch.cat([z, *new_cats], dim=1)
        return self.decode(z)

    def get_latent_representation(self, y: Tensor, cats: List[Tensor], **kwargs) -> Tensor:
        x = torch.cat([y, *cats], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z

class cVariationalRegressor(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 num_classes: int,
                 class_sizes: List[int],
                 hidden_dims_enc: List[int],
                 hidden_dims_dec: List[int],
                 device: str = 'cuda:0') -> None:
        super().__init__()

        self.num_classes = num_classes
        self.class_sizes = class_sizes
        self.latent_dim = latent_dim
        self.device = device
        self.hidden_dims_dec = hidden_dims_dec  # need this for ODE interaction!

        prev_d = 1 + np.sum(class_sizes)
        enc_layers = []
        for i in range(len(hidden_dims_enc)):
            enc_layers.append(nn.Sequential(nn.Linear(prev_d, hidden_dims_enc[i]), nn.ReLU()))
            prev_d = hidden_dims_enc[i]
        self.encoder = nn.Sequential(*enc_layers)
        self.mu_fc = nn.Linear(prev_d, latent_dim)
        self.var_fc = nn.Linear(prev_d, latent_dim)

        prev_d = latent_dim + np.sum(class_sizes)
        dec_layers = []
        for i in range(len(hidden_dims_dec)):
            dec_layers.append(nn.Sequential(nn.Linear(prev_d, hidden_dims_dec[i]), nn.ReLU()))
            prev_d = hidden_dims_dec[i]
        dec_layers.append(nn.Sequential(nn.Linear(prev_d, 1), nn.ReLU()))  # output is 1-d
        self.regressor = nn.Sequential(*dec_layers)

        for m in self.encoder:
            self.init_weights(m)
        self.init_weights(self.mu_fc)
        self.init_weights(self.var_fc)
        for m in self.regressor:
            self.init_weights(m)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def encode(self, x: Tensor) -> List[Tensor]:
        x = self.encoder(x)
        return self.mu_fc(x), self.var_fc(x)

    def regress(self, x: Tensor) -> Tensor:
        return self.regressor(x)

    def reparameterize(self, mu: Tensor, log_var: Tensor, **kwargs) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        if 'return_eps' in kwargs and kwargs['return_eps']:
            return eps * std + mu, eps
        return eps * std + mu

    def forward(self, y: Tensor, cats: List[Tensor], **kwargs) -> List[Tensor]:
        x = torch.cat([y, *cats], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var, **kwargs)
        z = torch.cat([z, *cats], dim=1)
        pred_y = self.regress(z)

        return pred_y, y, mu, log_var

    def loss_fn(self, *args, **kwargs) -> dict:
        pred_y = args[0]
        y = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['beta']

        recons_y_loss = F.mse_loss(pred_y, y, reduction='sum')
        kld_loss = torch.mean(-0.5 * torch.sum(
            1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_y_loss + kld_weight * kld_loss
        return {'loss': loss, 'Prediction Loss': recons_y_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, cats: List[Tensor], **kwargs) -> Tensor:
        z = torch.rand(size=(num_samples, self.latent_dim))
        z = torch.cat([z, *cats])
        return self.regress(z)

    def generate(self, y: Tensor, old_cats: List[Tensor], new_cats: List[Tensor], **kwargs) -> Tensor:
        x = torch.cat([y, *old_cats], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, *new_cats], dim=1)
        return self.regress(z)

    def get_latent_representation(self, y: Tensor, cats: List[Tensor], **kwargs) -> Tensor:
        x = torch.cat([y, *cats], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z


