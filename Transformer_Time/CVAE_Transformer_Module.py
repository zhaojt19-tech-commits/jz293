import torch
from torch import nn, Tensor
import numpy as np
from typing import *
import sys
import os

sys.path.append(os.path.join(sys.path[0], '../../../'))
from CVAE_Module import *
from Transformer_Module import *
from Predict_Label import *

DO_KAIMING_INIT = False

# learn_temporal_emb=False
# cont_model='sinusoidal'

class CatConTransformer(nn.Module):
    def __init__(self, input_size: List[int], input_channels: int, output_channels: int, num_classes: int, class_sizes: List[int], latent_dim: int, temporal_dim: int,
                 cat_dims: List[int], num_heads: int, num_ref_points: int, tr_dim: int,
                 adversarial_hidden_dims_encoder: List[int], latent_hidden_dims_encoder: List[int],
                 cvae_yemb_dims: List[int], learn_temporal_emb: bool, minT: int, maxT: int, device: str,
                 num_cont: int = None, other_temporal_dims: List[int] = None, other_minT: List[int] = None,
                 other_maxT: List[int] = None, cont_model: str = 'sinusoidal', rec_loss: str = 'mse',
                 temporal_uncertainty: bool = False, name_id=1) -> None:
        super().__init__()
        # 输入图像大小
        self.image_H = input_size[0]
        self.image_W = input_size[1]
        # 数据集编号
        self.name_id = name_id
        # 标签数
        self.K = num_classes
        # CVAE隐层维数
        self.ld = latent_dim
        # transformer_encoder输出的维数
        self.tr_dim = tr_dim
        # 重构损失计算方式
        self.rec_loss = rec_loss
        assert len(class_sizes) == self.K, 'Expected {} class sizes, but got {}'.format(self.K, len(class_sizes))
        assert len(cvae_yemb_dims) == self.K, 'Expected {} class sizes, but got {}'.format(self.K, len(cvae_yemb_dims))
        assert len(cat_dims) == self.K, 'Expected {} categorical embedding sizes, but got {}'.format(
            self.K, len(cat_dims))
        self.class_sizes = class_sizes    # 每个标签对应的类别数
        self.device = device
        # 可能没用
        self.num_cont = num_cont
        if num_cont is not None:
            assert len(other_temporal_dims) == num_cont - 1
            assert len(other_minT) == num_cont - 1
            assert len(other_maxT) == num_cont - 1

        self.kq_dim = temporal_dim + np.sum(cat_dims)

        # 定义CVAE
        self.cvae_enc = CVAE(self.image_H, self.image_W, input_channels, output_channels, latent_dim, cvae_yemb_dims, class_sizes, device, self.name_id)

        # 定义transformer_encoder
        self.transformer_enc = CatConEncoder(
            num_classes, class_sizes, latent_dim, tr_dim, temporal_dim, cat_dims, num_heads, num_ref_points, minT, maxT,
            device, learn_emb=learn_temporal_emb, num_cont=num_cont, other_temporal_dims=other_temporal_dims,
            other_minT=other_minT, other_maxT=other_maxT, cont_model=cont_model,
            temporal_uncertainty=temporal_uncertainty)
        # 定义transformer_decoder
        self.transformer_dec = CatConDecoder(
            num_classes, class_sizes, latent_dim, tr_dim, temporal_dim, cat_dims, num_heads, num_ref_points, minT, maxT,
            device, learn_emb=learn_temporal_emb, num_cont=num_cont, other_temporal_dims=other_temporal_dims,
            other_minT=other_minT, other_maxT=other_maxT, cont_model=cont_model,
            temporal_uncertainty=temporal_uncertainty)
        if  adversarial_hidden_dims_encoder is not None:
            # 定义label_predict模块（输入CVAE编码器输出）
            self.cvae_adv = CatConAdversary(
                self.ld, self.K, self.class_sizes, adversarial_hidden_dims_encoder, self.device).to(self.device)
            # 定义label_predict模块（输入transformer编码器输出）
            self.latent_adv = CatConAdversary(
                self.tr_dim, self.K, self.class_sizes, latent_hidden_dims_encoder, self.device).to(self.device)

    def forward(self, xs: Tensor,  # T x 1 * H * W
                ts: Tensor,  # N x T
                ys: List[Tensor],  # N x T x 1
                batch_size: int,
                mask: Tensor = None,  # N x T x M
                is_label = False, 
                other_ts: List[Tensor] = None,
                uncertainties: Tensor = None  # N x T
                ) -> None:
        N = 1
        T = xs.shape[0]

        if self.num_cont is not None and (self.num_cont > 1 or other_ts is not None):
            assert len(other_ts) == self.num_cont - 1

        mu, logvar = self.cvae_enc.encode(xs, ys, batch_size)
        z_cvae = self.cvae_enc.sample_z(mu, logvar).view(N, T, self.ld)  # N x T x ld
        if(is_label):
            # now, make sure we have removed categorical information
            adv_cvae = self.cvae_adv(z_cvae)
        else:
            adv_cvae = None

        # next, use transformer to produce generalized embedding
        tr_embs = self.transformer_enc(z_cvae, ys, ts, mask, other_ts, uncertainties)  # N x T x tr_dim

        if(is_label):
            # again, make sure that we haven't introduced categorical information
            adv_tr = self.latent_adv(tr_embs)
        else:
            adv_tr = None

        # decode at ts, ys
        z_dec = self.transformer_dec(tr_embs, ys, ts, other_ts, uncertainties)  # N x T x ld

        # xhat = self.cvae_enc.decode(z_dec, ys, batch_size)
        # print(xhat.shape)
        xhat = self.cvae_enc.decode(z_dec, ys, batch_size).view(T, 1, self.image_H, self.image_W)

        return xhat, mu, logvar, adv_cvae, adv_tr

    def loss_fn(self, x, ys, xhat, mu, logvar, adv_cvae, adv_tr, beta=0, adv_cvae_weight=0, adv_tr_weight=0):
        cvae_losses = self.cvae_enc.compute_loss(x=x, recon=xhat, mu=mu, log_var=logvar, kl_weight=beta)

        if adv_cvae_weight > 0:
            adv_cvae_loss = self.cvae_adv.loss_fn(adv_cvae, [ys.ravel()])
            adv_tr_loss = self.latent_adv.loss_fn(adv_tr, [ys.ravel()])
        else:
            adv_cvae_loss = torch.tensor([0]).to(self.device)
            adv_tr_loss = torch.tensor([0]).to(self.device)

        total_loss = cvae_losses['loss'] - adv_cvae_weight * adv_cvae_loss - adv_tr_weight * adv_tr_loss
        loss_dict = {
            'loss': total_loss,
            'cvae_loss': cvae_losses['loss'],
            'MSE': cvae_losses['MSE'],
            'KLD': cvae_losses['KLD'],
            'cvae_adversary': adv_cvae_loss,
            'latent_adversary': adv_tr_loss}
        return loss_dict

    def generate(self, xs, old_ts, new_ts, old_ys, new_ys, batch_size, old_mask, old_other_ts=None, new_other_ts=None,
                 compute_adv=False, k=1, get_zdec=False, old_uncertainties=None, new_uncertainties=None):
        N = 1
        T = xs.shape[0]
        T_new = new_ts.shape[-1]

        # again, embed values; no longer need to consider the adversary!
        mu, logvar = self.cvae_enc.encode(xs, old_ys, batch_size=T)  # NT x M+sum(cvae_yemb_dims)
        z_cvaes = []
        for _ in range(k):
            z_cvae = self.cvae_enc.sample_z(mu, logvar).view(N, T, self.ld)  # N x T x ld
            z_cvaes.append(z_cvae)
        z_cvae = torch.mean(torch.stack(z_cvaes, dim=0), dim=0)

        if compute_adv:
            adv_cvae = self.cvae_adv(z_cvae)
        else:
            adv_cvae = None

        # next, use transformer to produce generalized embedding
        tr_embs = self.transformer_enc(z_cvae, old_ys, old_ts, old_mask, additional_ts=old_other_ts,
                                       uncertainties=old_uncertainties)  # N x T x tr_dim

        if compute_adv:
            adv_tr = self.latent_adv(tr_embs)
        else:
            adv_tr = None

        # decode at new ts, new ys
        z_dec = self.transformer_dec(tr_embs, new_ys, new_ts, additional_ts=new_other_ts,
                                     uncertainties=new_uncertainties)  # N x T x ld

        xgen = self.cvae_enc.decode(z_dec, new_ys, batch_size).view(T_new, 1, self.image_H, self.image_W)

        if compute_adv:
            return xgen, mu, logvar, adv_cvae, adv_tr

        if get_zdec:
            return xgen, z_dec

        return xgen, mu, logvar, adv_cvae, adv_tr
    
    def MSE_sample(self, x, recon):
        loss_fn = torch.nn.MSELoss(reduction = 'none')
        recon_loss = loss_fn(recon, x)

        input_size = x.shape[0]
        recon_loss = recon_loss.reshape(input_size, -1)
        recon_loss = recon_loss.mean(axis=1, keepdim=False)

        return recon_loss