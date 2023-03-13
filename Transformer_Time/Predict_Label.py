import torch
from torch import nn, Tensor
import numpy as np
from typing import *
import sys
import os

DO_KAIMING_INIT = False

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
        # assert len(real_ys) == self.K
        # for k in range(self.K):
        #     assert pred_ys[k].shape[-1] == self.class_sizes[k], \
        #         "Unexpected number of predicted classes: {} instead of {}".format(pred_ys[k].shape, self.class_sizes[k])
        #     assert len(real_ys[k].shape) == 1, "Expected integer classes! Should have shape (N,), not {}".format(
        #         real_ys[k].shape)

        # print(pred_ys[0].squeeze().shape)
        # print(real_ys[0].shape)

        

        # N, T, _ = pred_ys[0].shape
        loss = 0
        for k in range(self.K):
            T = pred_ys[k].squeeze().shape[0]
            # print(pred_ys[k].squeeze().shape)
            # print(real_ys[k].shape)
            # print(real_ys[k][0:T].shape)
            # print(T)
            y = real_ys[k][0]
            label = torch.tensor([y] * T).to(self.device)
            temp = self.loss(pred_ys[k].squeeze(), label)
            loss = loss + temp
        # loss = sum(self.loss(pred_ys[k].view(-1, self.class_sizes[k]),
        #                      torch.stack([real_ys[k] for _ in range(T)], dim=1).view(-1))
        #            for k in range(self.K))
        return loss