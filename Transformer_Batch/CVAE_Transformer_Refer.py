import torch
from torch import nn, Tensor
import numpy as np
from typing import *
import sys
import os

sys.path.append(os.path.join(sys.path[0], '../../../'))
from models.conditional_regressor import cVAE_Sequence


DO_KAIMING_INIT = False

class CatConTransformer(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, class_sizes: List[int], latent_dim: int, temporal_dim: int,
                 cat_dims: List[int], value_hidden_dims: List[int], num_heads: int, num_ref_points: int, tr_dim: int,
                 adversarial_hidden_dims_encoder: List[List[int]], latent_hidden_dims_encoder: List[List[int]],
                 cvae_yemb_dims: List[int], learn_temporal_emb: bool, minT: int, maxT: int, device: str,
                 num_cont: int = None, other_temporal_dims: List[int] = None, other_minT: List[int] = None,
                 other_maxT: List[int] = None, cont_model: str = 'sinusoidal', rec_loss: str='mse',
                 temporal_uncertainty: bool = False) -> None:
        super().__init__()
        self.M = input_dim
        self.K = num_classes
        self.ld = latent_dim
        self.tr_dim = tr_dim
        self.rec_loss = rec_loss
        assert len(class_sizes) == self.K, 'Expected {} class sizes, but got {}'.format(self.K, len(class_sizes))
        assert len(cvae_yemb_dims) == self.K, 'Expected {} class sizes, but got {}'.format(self.K, len(cvae_yemb_dims))
        assert len(cat_dims) == self.K, 'Expected {} categorical embedding sizes, but got {}'.format(
            self.K, len(cat_dims))
        self.class_sizes = class_sizes
        self.device = device
        self.num_cont = num_cont
        if num_cont is not None:
            assert len(other_temporal_dims) == num_cont - 1
            assert len(other_minT) == num_cont - 1
            assert len(other_maxT) == num_cont - 1

        self.kq_dim = temporal_dim + np.sum(cat_dims)

        self.cvae_emb = nn.ModuleList([
            nn.Embedding(self.class_sizes[k], cvae_yemb_dims[k]).to(self.device) for k in range(self.K)])
        self.cvae_enc = cVAE_Sequence(
            input_size=self.M, latent_dim=self.ld, num_classes=self.K, class_sizes=cvae_yemb_dims,
            hidden_dims=value_hidden_dims, device=self.device, kaiming_init=DO_KAIMING_INIT).to(self.device)
        self.cvae_adv = CatConAdversary(
            self.ld, self.K, self.class_sizes, adversarial_hidden_dims_encoder, self.device).to(self.device)
        self.latent_adv = CatConAdversary(
            self.tr_dim, self.K, self.class_sizes, latent_hidden_dims_encoder, self.device).to(self.device)
        # self.decoder_adv = CatConAdversary(
        #     self.kq_dim, self.K, self.class_sizes, adversarial_hidden_dims_decoder, self.device).to(self.device)

        self.transformer_enc = CatConEncoder(
            num_classes, class_sizes, latent_dim, tr_dim, temporal_dim, cat_dims, num_heads, num_ref_points, minT, maxT,
            device, learn_emb=learn_temporal_emb, num_cont=num_cont, other_temporal_dims=other_temporal_dims,
            other_minT=other_minT, other_maxT=other_maxT, cont_model=cont_model,
            temporal_uncertainty=temporal_uncertainty)
        self.transformer_dec = CatConDecoder(
            num_classes, class_sizes, latent_dim, tr_dim, temporal_dim, cat_dims, num_heads, num_ref_points, minT, maxT,
            device, learn_emb=learn_temporal_emb, num_cont=num_cont, other_temporal_dims=other_temporal_dims,
            other_minT=other_minT, other_maxT=other_maxT, cont_model=cont_model,
            temporal_uncertainty=temporal_uncertainty)

    def forward(self, xs: Tensor,  # N x T x M
                ts: Tensor,  # N x T
                ys: List[Tensor],  # Len K, ys[i] has shape N x T x 1
                mask: Tensor,  # N x T x M
                other_ts: List[Tensor] = None,
                uncertainties: Tensor = None  # N x T
                ) -> None:
        N, T, M = xs.shape

        if self.num_cont is not None and (self.num_cont > 1 or other_ts is not None):
            assert len(other_ts) == self.num_cont - 1

        # first, get value embeddings
        cvae_ys = [self.cvae_emb[k](ys[k]) for k in range(self.K)]
        mu, logvar = self.cvae_enc.encode(torch.cat([xs, *cvae_ys], dim=-1).view(N*T, -1))  # NT x M+sum(cvae_yemb_dims)
        z_cvae = self.cvae_enc.reparameterize(mu, logvar).view(N, T, self.ld)  # N x T x ld

        # now, make sure we have removed categorical information
        adv_cvae = self.cvae_adv(z_cvae)

        # next, use transformer to produce generalized embedding
        tr_embs = self.transformer_enc(z_cvae, ys, ts, mask, other_ts, uncertainties)  # N x T x tr_dim

        # again, make sure that we haven't introduced categorical information
        adv_tr = self.latent_adv(tr_embs)

        # decode at ts, ys
        z_dec = self.transformer_dec(tr_embs, ys, ts, other_ts, uncertainties)  # N x T x ld

        xhat = self.cvae_enc.decode(torch.cat([z_dec, *cvae_ys], dim=-1).view(N*T, -1)).view(N, T, M)

        return xhat, mu, logvar, adv_cvae, adv_tr

    def loss_fn(self, x, ys, xhat, mu, logvar, adv_cvae, adv_tr, beta=1.0, adv_cvae_weight=1.0, adv_tr_weight=1.0):
        cvae_losses = self.cvae_enc.loss_fn(xhat, x, mu, logvar, beta=beta, rec_loss=self.rec_loss)

        if self.K > 0:
            adv_cvae_loss = self.cvae_adv.loss_fn(adv_cvae, ys)
            adv_tr_loss = self.latent_adv.loss_fn(adv_tr, ys)
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

    def generate(self, xs, old_ts, new_ts, old_ys, new_ys, old_mask, old_other_ts=None, new_other_ts=None,
                 compute_adv=False, k=1, get_zdec=False, old_uncertainties=None, new_uncertainties=None):
        N, T, M = xs.shape
        T_new = new_ts.shape[-1]

        # again, embed values; no longer need to consider the adversary!
        old_cvae_ys = [self.cvae_emb[k](old_ys[k]) for k in range(self.K)]
        mu, logvar = self.cvae_enc.encode(
            torch.cat([xs, *old_cvae_ys], dim=-1).view(N * T, -1))  # NT x M+sum(cvae_yemb_dims)
        z_cvaes = []
        for _ in range(k):
            z_cvae = self.cvae_enc.reparameterize(mu, logvar).view(N, T, self.ld)  # N x T x ld
            z_cvaes.append(z_cvae)
        z_cvae = torch.mean(torch.stack(z_cvaes, dim=0), dim=0)

        if compute_adv:
            adv_cvae = self.cvae_adv(z_cvae)

        # next, use transformer to produce generalized embedding
        tr_embs = self.transformer_enc(z_cvae, old_ys, old_ts, old_mask, additional_ts=old_other_ts,
                                       uncertainties=old_uncertainties)  # N x T x tr_dim

        if compute_adv:
            adv_tr = self.latent_adv(tr_embs)

        # decode at new ts, new ys
        z_dec = self.transformer_dec(tr_embs, new_ys, new_ts, additional_ts=new_other_ts,
                                     uncertainties=new_uncertainties)  # N x T x ld
        new_cvae_ys = [self.cvae_emb[k](new_ys[k]) for k in range(self.K)]
        xgen = self.cvae_enc.decode(torch.cat([z_dec, *new_cvae_ys], dim=-1).view(N * T_new, -1)).view(N, T_new, M)

        if compute_adv:
            return xgen, mu, logvar, adv_cvae, adv_tr
        
        if get_zdec:
            return xgen, z_dec

        return xgen
    
    def predict(self, xs: Tensor,  # N x T x M
                ts: Tensor,  # N x T
                ys: List[Tensor],  # Len K, ys[i] has shape N x T x 1
                mask: Tensor,  # N x T x M
                other_ts: List[Tensor] = None,
                k: int=10,
                ) -> None:
        N, T, M = xs.shape

        if self.num_cont is not None and (self.num_cont > 1 or other_ts is not None):
            assert len(other_ts) == self.num_cont - 1

        # first, get value embeddings
        cvae_ys = [self.cvae_emb[k](ys[k]) for k in range(self.K)]
        mu, logvar = self.cvae_enc.encode(torch.cat([xs, *cvae_ys], dim=-1).view(N*T, -1))  # NT x M+sum(cvae_yemb_dims)
        z_cvaes = []
        for _ in range(k):
            z_cvae = self.cvae_enc.reparameterize(mu, logvar).view(N, T, self.ld)  # N x T x ld
            z_cvaes.append(z_cvae)
        z_cvae = torch.mean(torch.stack(z_cvaes, dim=0), dim=0)  # N x T x ld

        # now, make sure we have removed categorical information
        adv_cvae = self.cvae_adv(z_cvae)

        # next, use transformer to produce generalized embedding
        tr_embs = self.transformer_enc(z_cvae, ys, ts, mask, other_ts)  # N x T x tr_dim

        # again, make sure that we haven't introduced categorical information
        adv_tr = self.latent_adv(tr_embs)

        # decode at ts, ys
        z_dec = self.transformer_dec(tr_embs, ys, ts, other_ts)  # N x T x ld

        xhat = self.cvae_enc.decode(torch.cat([z_dec, *cvae_ys], dim=-1).view(N*T, -1)).view(N, T, M)

        return xhat, mu, logvar, adv_cvae, adv_tr


class CatConAdversary(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, class_sizes: List[int], hidden_dims: List[int], device: str
                 ) -> None:
        super().__init__()
        self.inp_dim = input_dim
        self.K = num_classes
        assert len(class_sizes) == self.K, 'Expected {} class sizes, but got {}'.format(self.K, len(class_sizes))
        self.class_sizes = class_sizes
        self.device = device

        self.predictors = []
        for k in range(self.K):
            modules = []
            prev_size = self.inp_dim
            for hdim in hidden_dims:
                modules.extend([nn.Linear(prev_size, hdim), nn.ReLU()])
                prev_size = hdim
                if DO_KAIMING_INIT:
                    nn.init.kaiming_normal_(modules[-2].weight, mode='fan_in')
            modules.extend([nn.Linear(prev_size, self.class_sizes[k]), nn.Softmax()])
            self.predictors.append(nn.Sequential(*modules).to(self.device))
        self.predictors = nn.ModuleList(self.predictors)
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, z_unk: Tensor) -> List[Tensor]:
        pred_ys = [pred(z_unk) for pred in self.predictors]
        return pred_ys

    def loss_fn(self, pred_ys: List[Tensor], real_ys: List[Tensor]) -> Tensor:
        assert len(real_ys) == self.K
        for k in range(self.K):
            assert pred_ys[k].shape[-1] == self.class_sizes[k], \
                "Unexpected number of predicted classes: {} instead of {}".format(pred_ys[k].shape, self.class_sizes[k])
            assert len(real_ys[k].shape) == 1, "Expected integer classes! Should have shape (N,), not {}".format(
                real_ys[k].shape)

        N, T, _ = pred_ys[0].shape
        loss = sum(self.loss(pred_ys[k].view(-1, self.class_sizes[k]), torch.stack([real_ys[k] for _ in range(T)], dim=1).view(-1))
                   for k in range(self.K))
        return loss
# ***************************************************************************************************************

class MultiHeadAttention(nn.Module):
    def __init__(self, latent_dim: int, value_dim: int, kq_dim: int, num_heads: int, device: str) -> None:
        super().__init__()
        self.device = device
        self.kq_dim = kq_dim
        self.num_heads = num_heads
        self.ld = latent_dim
        self.vdim = value_dim

        self.attn_linear = nn.ModuleList([nn.Linear(self.kq_dim, self.num_heads * self.kq_dim),  # as in mTAN!
                                          nn.Linear(self.kq_dim, self.num_heads * self.kq_dim)])
        self.attn_out = nn.Linear(self.ld * num_heads, self.ld)
        if DO_KAIMING_INIT:
            nn.init.xavier_normal_(self.attn_linear[0].weight)
            nn.init.xavier_normal_(self.attn_linear[1].weight)
            nn.init.xavier_normal_(self.attn_out.weight)

    def forward(self, value: Tensor, key: Tensor, query: Tensor, mask: Tensor = None) -> Tuple[Tensor]:
        N, T, _ = value.shape  # N x T x value_dim
        N_q, T_q, _ = query.shape
        N_k, T_k, _ = key.shape
        head_queries = self.attn_linear[0](
            query.view(-1, self.kq_dim)).view(N_q, T_q, self.num_heads, self.kq_dim).transpose(2, 1)  # N_q x h x T_q x kq_dim
        head_keys = self.attn_linear[1](
            key.view(-1, self.kq_dim)).view(N_k, T_k, self.num_heads, self.kq_dim).transpose(2, 1)  # N x h x T_q x kq_dim
        scores = torch.matmul(head_queries, head_keys.transpose(-2, -1)) / np.sqrt(self.kq_dim)  # N x h x T_q x T
        scores = scores.unsqueeze(-1).repeat_interleave(self.vdim, dim=-1)  # N x h x T_q x T x vdim
        if mask is not None:
            mask = mask.unsqueeze(1)  # N x 1 x T (so same for each head)
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = nn.functional.softmax(scores, dim=-2)

        out_rep = torch.sum(p_attn * value.unsqueeze(1).unsqueeze(-3), -2).transpose(1, 2).contiguous().view(
            N, -1, self.num_heads * self.vdim)
        out_rep = self.attn_out(out_rep)
        return out_rep, p_attn


class CatConLayers(nn.Module):
    def __init__(self, num_classes: int, class_sizes: List[int], input_latent_dim: int, output_latent_dim: int,
                 temporal_dim: int, cat_dims: List[int], num_heads: int, num_ref_points: int, minT: int, maxT: int,
                 device: str, learn_emb: bool, num_cont: int = None, other_temporal_dims: List[int] = None,
                 other_minT: List[int] = None, other_maxT: List[int] = None, cont_model: str = 'sinusoidal',
                 temporal_uncertainty: bool = False) -> None:
        super().__init__()
        self.K = num_classes
        assert len(class_sizes) == self.K
        self.class_sizes = class_sizes
        self.value_dim = input_latent_dim
        self.ld = output_latent_dim
        self.temp_dims = [temporal_dim] if num_cont is None else [temporal_dim] + other_temporal_dims
        self.kq_dim = int(np.sum(self.temp_dims) + np.sum(cat_dims)) if cont_model not in {
            'conditional_embedding', 'residual_embedding'} else int(np.sum(self.temp_dims))
        self.num_heads = num_heads
        self.num_ref_points = num_ref_points
        self.minT = minT
        self.maxT = maxT
        self.device = device
        self.cont_model = cont_model
        self.temporal_uncertainty = temporal_uncertainty

        self.class_embedders = nn.ModuleList([
            nn.Embedding(class_sizes[k] + 1, cat_dims[k]).to(self.device) for k in range(self.K)])
        self.learn_emb = learn_emb
        self.ref_ts = torch.linspace(minT, maxT, num_ref_points).to(self.device)
        if num_cont is not None:
            self.other_ref_ts = [torch.linspace(
                other_minT[i], other_maxT[i], num_ref_points).to(self.device) for i in range(num_cont - 1)]
            self.other_minT = other_minT
            self.other_maxT = other_maxT
        if learn_emb:
            self.periodic = nn.ModuleList(
                [nn.Linear(1, self.temp_dims[i] - 1) for i in range(len(self.temp_dims))])  # omega, alpha for sin terms
            self.linear = nn.ModuleList([
                nn.Linear(1, 1) for i in range(len(self.temp_dims))])  # omega, alpha for linear term

        self.attn_network = MultiHeadAttention(self.ld, self.value_dim, self.kq_dim, self.num_heads, self.device
                                               ).to(self.device)

        if self.cont_model in {'conditional_embedding', 'residual_embedding'}:
            self.continuous_embedder = nn.ModuleList(
                [nn.Sequential(nn.Linear(1 + int(np.sum(cat_dims)), self.temp_dims[i]), nn.ReLU()) for i in
                 range(num_cont if num_cont is not None else 1)])

        if self.temporal_uncertainty:
            self.uncertainty_model = nn.Sequential(nn.Linear(1 + int(np.sum(cat_dims)), 1), nn.ReLU())  # compute poission lambda

    def get_class_embeddings(self, ys: List[Tensor]) -> Tensor:
        # ys[k] has shape N x T x 1 -> move to N x T x sum(cat_dims)
        return torch.cat([
            self.class_embedders[k](ys[k]) for k in range(self.K)], dim=-1)  # N x T x sum(cat_dims)

    def normalize_ts_sigmoid(self, ts: List[Tensor]) -> List[Tensor]:
        norm_ts = [2 * (ts[0] * self.minT) / (self.maxT - self.minT) - 1]
        norm_ts.extend([2 * (ts[i+1] * self.other_minT[i]) / (self.other_maxT[i] - self.other_minT[i]) - 1
                        for i in range(self.K - 1)])
        return norm_ts

    def get_time_embeddings(self, ts: List[Tensor], zys: Tensor) -> Tensor:
        # N, Ti = ts[i].shape
        N = ts[0].shape[0]
        if self.cont_model == 'sinusoidal' or self.cont_model == 'linear_sinusoidal':
            if self.learn_emb:
                t_linear = [self.linear[i](ts[i].unsqueeze(-1)) for i in range(len(ts))]  # N x T x 1
                t_periodic = [torch.sin(self.periodic[i](ts[i].unsqueeze(-1))) for i in range(len(ts))]  # N x T x self.temp_dim - 1
                z_ts = [torch.cat([t_linear[i], t_periodic[i]], dim=-1) for i in range(len(ts))]  # N x T x self.temp_dim
            else:  # follow mTAN weighting
                z_ts = [torch.zeros(N, ts[i].shape[1], self.temp_dims[i]).to(self.device) for i in range(len(ts))]
                position = [48. * ts[i].unsqueeze(2) for i in range(len(ts))]  # N x T x 1
                div_term = [torch.exp(torch.arange(0, self.temp_dims[i], 2) *
                                      -(np.log(10.0) / self.temp_dims[i])).to(self.device) for i in range(len(ts))
                            ]  # self.temp_dim / 2
                for i in range(len(ts)):
                    z_ts[i][:, :, 0::2] = torch.sin(position[i] * div_term[i])
                    z_ts[i][:, :, 1::2] = torch.cos(position[i] * div_term[i])
                    if self.cont_model == 'linear_sinusoidal':
                        z_ts[i][:, :, 0] = ts[i]
            return torch.cat(z_ts, dim=-1)  # N x T x self.temp_dims
        elif self.cont_model == 'sigmoidal':
            # first, let's normalize the ts to [-1, 1]
            ts = self.normalize_ts_sigmoid(ts)
            if self.learn_emb:
                t_linear = [self.linear[i](ts[i].unsqueeze(-1)) for i in range(len(ts))]
                t_sig = [torch.sigmoid(self.periodic[i](ts[i].unsqueeze(-1))) for i in range(len(ts))]
                z_ts = [torch.cat([t_linear[i], t_sig[i]], dim=-1) for i in range(len(ts))]
            else:
                z_ts = [torch.zeros(N, ts[i].shape[1], self.temp_dims[i]).to(self.device) for i in range(len(ts))]
                position = [48. * ts[i].unsqueeze(2) for i in range(len(ts))]  # N x T x 1
                div_term = [torch.exp(torch.arange(0, self.temp_dims[i], 1) *
                                      -(np.log(10.0) / self.temp_dims[i])).to(self.device) for i in range(len(ts))
                            ]  # self.temp_dim
                for i in range(len(ts)):
                    z_ts[i] = torch.sigmoid(position[i] * div_term[i])
            return torch.cat(z_ts, dim=-1)
        elif self.cont_model == 'conditional_embedding':
            assert zys is not None
            inp_list = [torch.cat([t.unsqueeze(-1), zys], dim=-1) for t in ts]  # N x T x 1 + sum(cat_dims)
            N, T, d = inp_list[0].shape
            t_outs = [self.continuous_embedder[i](inp.view(-1, d)).view(N, T, -1) for i, inp in enumerate(inp_list)]
            return torch.cat(t_outs, dim=-1)
        elif self.cont_model == 'residual_embedding':
            assert zys is not None
            inp_list = [torch.cat([t.unsqueeze(-1), zys], dim=-1) for t in ts]  # N x T x 1 + sum(cat_dims)
            N, T, d = inp_list[0].shape
            t_outs = [self.continuous_embedder[i](inp.view(-1, d)).view(N, T, -1) for i, inp in enumerate(inp_list)]
            tres = [t.unsqueeze(-1) + t_outs for t, t_outs in zip(ts, t_outs)]
            return torch.cat(tres, dim=-1)
        else:
            raise NotImplementedError("Unrecognized embedding method: {}".format(self.cont_model))

    def introduce_uncertainties(self, ts: Tensor, ys: Tensor, uncertainties: Tensor):
        inp = torch.cat([ts.unsqueeze(-1), ys], dim=-1)  # N x T x cat_dims+1
        N, T, d = inp.shape
        poisson_lambdas = self.uncertainty_model(inp.view(-1, d)).view(N, T)
        noise = torch.poisson(poisson_lambdas) * uncertainties
        return ts + noise
# *******************************************************************************
class CatConEncoder(CatConLayers):
    def __init__(self, num_classes: int, class_sizes: List[int], input_latent_dim: int, output_latent_dim: int,
                 temporal_dim: int, cat_dims: List[int], num_heads: int, num_ref_points: int, minT: int, maxT: int,
                 device: str, learn_emb: bool, num_cont: int = None, other_temporal_dims: List[int] = None,
                 other_minT: List[int] = None, other_maxT: List[int] = None, cont_model: str = 'sinusoidal',
                 temporal_uncertainty: bool = False) -> None:
        super().__init__(num_classes, class_sizes, input_latent_dim, output_latent_dim, temporal_dim, cat_dims,
                         num_heads, num_ref_points, minT, maxT, device, learn_emb, num_cont, other_temporal_dims,
                         other_minT, other_maxT, cont_model, temporal_uncertainty)

    def forward(self, z_seq: Tensor, ys: List[Tensor], ts: Tensor, mask: Tensor,
                additional_ts: List[Tensor] = None, uncertainties: Tensor = None) -> Tensor:
        assert len(ys) == self.K
        N, T, d = z_seq.shape  # value to use for transformer
        assert d == self.value_dim

        # build key from data
        if self.K > 0:
            z_ys = self.get_class_embeddings(ys)  # N x T x sum(cat_dims)
        ts_list = [ts] if additional_ts is None else [ts] + additional_ts
        if self.temporal_uncertainty:
            ts_list = [self.introduce_uncertainties(t, z_ys, uncertainties) for t in ts_list]
#             ts_list = [self.introduce_uncertainties(t, z_seq, uncertainties) for t in ts_list]
        z_ts = self.get_time_embeddings(ts_list, z_ys)  # N x T x self.temp_dims
        if self.K > 0 and self.cont_model not in {'conditional_embedding', 'residual_embedding'}:
            key = torch.cat([z_ys, z_ts], dim=-1)  # N x T x self.kq_dim
        else:
            key = z_ts

        # build query from reference points
        if self.K > 0:
            z_unk = self.get_class_embeddings(
                [torch.stack([torch.tensor([Dk for _ in range(1)]).to(self.device) for _ in range(self.num_ref_points)], dim=1)
                 for Dk in self.class_sizes])  # N x T x sum(cat_dims)
        ref_t_list = [self.ref_ts.unsqueeze(0)] if additional_ts is None else \
            [self.ref_ts.unsqueeze(0)] + [self.other_ref_ts[i].unsqueeze(0) for i in range(len(additional_ts))]
        z_reft = self.get_time_embeddings(ref_t_list, z_unk)  # 1 x T x self.temp_dim
        if self.K > 0 and self.cont_model not in {'conditional_embedding', 'residual_embedding'}:
            query = torch.cat([z_unk, z_reft], dim=-1)  # N x T x self.kq_dim
        else:
            query = z_reft

        out, _ = self.attn_network(z_seq, key, query, mask=torch.stack([mask for _ in range(self.ld)], dim=-1))
        return out


class CatConDecoder(CatConLayers):
    def __init__(self, num_classes: int, class_sizes: List[int], input_latent_dim: int, output_latent_dim: int,
                 temporal_dim: int, cat_dims: List[int], num_heads: int, num_ref_points: int, minT: int, maxT: int,
                 device: str, learn_emb: bool, num_cont: int = None, other_temporal_dims: List[int] = None,
                 other_minT: List[int] = None, other_maxT: List[int] = None, cont_model: str = 'sinusoidal',
                 temporal_uncertainty: bool = False) -> None:
        super().__init__(num_classes, class_sizes, input_latent_dim, output_latent_dim, temporal_dim, cat_dims,
                         num_heads, num_ref_points, minT, maxT, device, learn_emb, num_cont, other_temporal_dims,
                         other_minT, other_maxT, cont_model, temporal_uncertainty)

    def forward(self, emb_seq: Tensor, ys: List[Tensor], ts: Tensor, additional_ts: List[Tensor] = None,
                uncertainties: Tensor = None) -> Tensor:
        assert len(ys) == self.K
        N, T, d = emb_seq.shape  # value to use for transformer
        assert d == self.ld

        # build query from data
        if self.K > 0:
            z_ys = self.get_class_embeddings(ys)  # N x T x sum(cat_dims)
        ts_list = [ts] if additional_ts is None else [ts] + additional_ts
        if self.temporal_uncertainty:
            ts_list = [self.introduce_uncertainties(t, z_ys, uncertainties) for t in ts_list]
#             ts_list = [self.introduce_uncertainties(t, emb_seq, uncertainties) for t in ts_list]
        z_ts = self.get_time_embeddings(ts_list, z_ys)  # N x T x self.temp_dims
        if self.K > 0 and self.cont_model not in {'conditional_embedding', 'residual_embedding'}:
            query = torch.cat([z_ys, z_ts], dim=-1)  # N x T x self.kq_dim
        else:
            query = z_ts

        # build key from reference points
        if self.K > 0:
            z_unk = self.get_class_embeddings(
                [torch.tensor([Dk for _ in range(self.num_ref_points)]).to(self.device) 
                 for Dk in self.class_sizes]).unsqueeze(0)  # 1 x T_ref x sum(cat_dims)
        ref_t_list = [self.ref_ts.unsqueeze(0)] if additional_ts is None else \
            [self.ref_ts.unsqueeze(0)] + [self.other_ref_ts[i].unsqueeze(0) for i in range(len(additional_ts))]
        z_reft = self.get_time_embeddings(ref_t_list, z_unk)  # 1 x T_ref x self.temp_dim
        if self.K > 0 and self.cont_model not in {'conditional_embedding', 'residual_embedding'}:
            key = torch.cat([z_unk, z_reft], dim=-1)  # N x T x self.kq_dim
        else:
            key = z_reft

        out, _ = self.attn_network(emb_seq, key, query, mask=None)
        return out

# ******************************************************************************************
As a starting point, I recommend using learn_temporal_emb=False, cont_model='sinusoidal'. If you have any questions about it, let me know!
Also, the cVAE_Sequence class at the bottom of this file is the CatConTransformer.cvae_enc model — I think you’ll need to replace this 
with your own (similar) class to handle the image inputs
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import *


class cVariationalRegressor(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 num_classes: int,
                 class_sizes: List[int],
                 hidden_dims_enc: List[int],
                 hidden_dims_dec: List[int],
                 device: str='cuda:0') -> None:
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
            1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        loss = recons_y_loss + kld_weight * kld_loss
        return {'loss': loss, 'Prediction Loss': recons_y_loss , 'KLD': -kld_loss}

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
                nn.init.kaiming_normal_(enc_layers[-1][0].weight, mode='fan_in')  # use kaiming initialization for linear + relu
            prev_d = hidden_dims[i]
        self.encoder = nn.Sequential(*enc_layers)
        self.mu_fc = nn.Linear(prev_d, latent_dim)
        self.var_fc = nn.Linear(prev_d, latent_dim)

        prev_d = latent_dim + int(np.sum(class_sizes))
        dec_layers = []
        for i in range(len(hidden_dims)-1, -1, -1):  # go backwards through hidden dims
            dec_layers.append(nn.Sequential(nn.Linear(prev_d, hidden_dims[i]), nn.ReLU()))
            if kaiming_init:
                nn.init.kaiming_normal_(dec_layers[-1][0].weight, mode='fan_in')
            prev_d = hidden_dims[i]
        dec_layers.append(nn.Sequential(nn.Linear(prev_d, input_size))) #, nn.ReLU()))  # output is 1-d
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