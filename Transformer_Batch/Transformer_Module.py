import torch
from torch import nn, Tensor
import numpy as np
from typing import *
import sys
import os

sys.path.append(os.path.join(sys.path[0], '../../../'))

DO_KAIMING_INIT = False

# learn_temporal_emb=False
# cont_model='sinusoidal'


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
            query.view(-1, self.kq_dim)).view(N_q, T_q, self.num_heads, self.kq_dim).transpose(2,
                                                                                               1)  # N_q x h x T_q x kq_dim
        head_keys = self.attn_linear[1](
            key.view(-1, self.kq_dim)).view(N_k, T_k, self.num_heads, self.kq_dim).transpose(2,
                                                                                             1)  # N x h x T_q x kq_dim
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

        # print(class_sizes)
        # print(self.K)

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
            self.uncertainty_model = nn.Sequential(nn.Linear(1 + int(np.sum(cat_dims)), 1),
                                                   nn.ReLU())  # compute poission lambda

    def get_class_embeddings(self, ys: List[Tensor]) -> Tensor:
        # ys[k] has shape N x T x 1 -> move to N x T x sum(cat_dims)
        return torch.cat([
            self.class_embedders[k](ys[k]) for k in range(self.K)], dim=-1)  # N x T x sum(cat_dims)

    def normalize_ts_sigmoid(self, ts: List[Tensor]) -> List[Tensor]:
        norm_ts = [2 * (ts[0] * self.minT) / (self.maxT - self.minT) - 1]
        norm_ts.extend([2 * (ts[i + 1] * self.other_minT[i]) / (self.other_maxT[i] - self.other_minT[i]) - 1
                        for i in range(self.K - 1)])
        return norm_ts

    def get_time_embeddings(self, ts: List[Tensor], zys: Tensor) -> Tensor:
        # N, Ti = ts[i].shape
        N = ts[0].shape[0]
        # ts = ts.float()
        if self.cont_model == 'sinusoidal' or self.cont_model == 'linear_sinusoidal':
            if self.learn_emb:
                t_linear = [self.linear[i](ts[i].unsqueeze(-1).float()) for i in range(len(ts))]  # N x T x 1
                t_periodic = [torch.sin(self.periodic[i](ts[i].unsqueeze(-1).float())) for i in
                              range(len(ts))]  # N x T x self.temp_dim - 1
                z_ts = [torch.cat([t_linear[i], t_periodic[i]], dim=-1) for i in
                        range(len(ts))]  # N x T x self.temp_dim
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
        ys = ys[0]
        N, T, d = z_seq.shape  # value to use for transformer
        ys = ys.reshape(-1, 1)
        # print(ys.shape)
        assert d == self.value_dim
        # build key from data
        if self.K > 0:
            z_ys = self.get_class_embeddings([ys]).reshape(N, T, -1)  # N x T x sum(cat_dims)
        ts_list = [ts] if additional_ts is None else [ts] + additional_ts
        if self.temporal_uncertainty:
            ts_list = [self.introduce_uncertainties(t, z_ys, uncertainties) for t in ts_list]
        #             ts_list = [self.introduce_uncertainties(t, z_seq, uncertainties) for t in ts_list]
        z_ts = self.get_time_embeddings(ts_list, z_ys)  # N x T x self.temp_dims
        # print(z_ys.shape)
        # print(z_ts.shape)
        if self.K > 0 and self.cont_model not in {'conditional_embedding', 'residual_embedding'}:
            key = torch.cat([z_ys, z_ts], dim=-1)  # N x T x self.kq_dim
        else:
            key = z_ts

        # build query from reference points
        if self.K > 0:
            z_unk = self.get_class_embeddings(
                [torch.stack([torch.tensor([Dk for _ in range(1)]).to(self.device) for _ in range(self.num_ref_points)],
                             dim=1)
                 for Dk in self.class_sizes])  # N x T x sum(cat_dims)
        ref_t_list = [self.ref_ts.unsqueeze(0)] if additional_ts is None else \
            [self.ref_ts.unsqueeze(0)] + [self.other_ref_ts[i].unsqueeze(0) for i in range(len(additional_ts))]
        z_reft = self.get_time_embeddings(ref_t_list, z_unk)  # 1 x T x self.temp_dim
        if self.K > 0 and self.cont_model not in {'conditional_embedding', 'residual_embedding'}:
            query = torch.cat([z_unk, z_reft], dim=-1)  # N x T x self.kq_dim
        else:
            query = z_reft
        # 用到mask的时候，参照以下代码
        # mask=torch.stack([mask for _ in range(self.ld)], dim=-1)
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
        ys = ys[0]
        # N, T, d = emb_seq.shape  # value to use for transformer
        # assert d == self.ld
        N = ys.shape[0]
        T = ys.shape[1]
        ys = ys.reshape(-1, 1)
        # build query from data
        if self.K > 0:
            z_ys = self.get_class_embeddings([ys]).reshape(N, T, -1)  # N x T x sum(cat_dims)
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