import torch.nn as nn
import torch
import numpy as np
from abc import ABC, abstractmethod
import copy


class Builder(ABC):

    @abstractmethod
    def get_encoder(self):
        pass

    @abstractmethod
    def get_encoder_output_dim(self):
        pass

    @abstractmethod
    def get_decoder(self):
        pass

    @abstractmethod
    def get_final_layer(self):
        pass


class SquareConvolutionalBuilder(Builder):
    def __init__(self, in_channels, hidden_dims, kernel_size, stride,
                 padding, output_padding, out_channels=None):
        super(SquareConvolutionalBuilder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def get_encoder(self):
        if self.encoder is not None:
            return self.encoder

        modules = []
        in_channels = self.in_channels
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=self.kernel_size, stride=self.stride,
                              padding=self.padding),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        return nn.Sequential(*modules)

    def get_encoder_output_dim(self, img_height, img_width):
        out_width = img_width
        out_height = img_height
        for _ in self.hidden_dims:
            out_width = int(np.floor(
                (out_width - self.kernel_size + 2 * self.padding)
                / self.stride)) + self.padding
            out_height = int(np.floor(
                (out_height - self.kernel_size + 2 * self.padding)
                / self.stride)) + self.padding
        return np.array([self.hidden_dims[-1], out_height, out_width])

    def get_decoder(self):
        reversed_dims = copy.deepcopy(self.hidden_dims)
        reversed_dims.reverse()

        modules = []
        for i in range(len(reversed_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(reversed_dims[i], reversed_dims[i + 1],
                                   kernel_size=self.kernel_size, stride=self.stride,
                                   padding=self.padding, output_padding=self.output_padding),
                nn.BatchNorm2d(reversed_dims[i + 1]),
                nn.LeakyReLU())
            )
        return nn.Sequential(*modules)

    def get_final_layer(self):
        h_dim = self.hidden_dims[0]
        return nn.Sequential(
            nn.ConvTranspose2d(h_dim, h_dim, kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding, output_padding=self.output_padding),
            nn.BatchNorm2d(h_dim),
            nn.LeakyReLU(),
            nn.Conv2d(h_dim, out_channels=self.out_channels, kernel_size=self.kernel_size,
                      padding=self.padding),
            nn.Sigmoid()
        )


class UnFlatten(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, x):
        nc = x[0].numel() // (self.w ** 2)
        return x.view(x.size(0), nc, self.w, self.w)


class ODE2VAEBuilderFromArch(Builder):
    def __init__(self, in_channels, num_classes, n_filt=32, out_channels=None, name_id=1):
        super().__init__()
        self.name_id = name_id
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.num_classes = num_classes
        self.hidden_dims_enc = [n_filt, n_filt * 2, n_filt * 4, n_filt * 8]
        self.hidden_dims_dec = [n_filt * 4, n_filt * 2, n_filt]
        self.kernel_enc = [3, 3, 3, 3]
        self.stride_enc = [2, 2, 2, 2]
        self.padding_enc = [1, 1, 1, 1]
        self.kernel_dec = [(3, 3), (3, 3), (3, 3)]
        self.stride_dec = [(2, 2), (2, 2), (2, 2)]
        self.padding_dec = [(1, 1), (1, 1), (1, 1)]
        self.output_padding_dec = [(1, 1), (1, 1), (1, 1)]

    def get_encoder(self):
        modules = []
        in_channels = self.in_channels + self.num_classes
        for i in range(len(self.hidden_dims_enc)):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=self.hidden_dims_enc[i],
                              kernel_size=self.kernel_enc[i],
                              stride=self.stride_enc[i], padding=self.padding_enc[i]),
                    # nn.BatchNorm2d(self.hidden_dims_enc[i]),
                    nn.ReLU())
            )
            in_channels = self.hidden_dims_enc[i]

        return nn.Sequential(*modules)

    def get_encoder_output_dim(self, img_height, img_width):
        out_width = img_width
        out_height = img_height
        for i in range(len(self.hidden_dims_enc)):
            out_width = int(np.floor(
                (out_width - self.kernel_enc[i] + 2 * self.padding_enc[i])
                / self.stride_enc[i])) + 1
            out_height = int(np.floor(
                (out_height - self.kernel_enc[i] + 2 * self.padding_enc[i])
                / self.stride_enc[i])) + 1
        return np.array([self.hidden_dims_enc[-1], out_height, out_width])

    def get_decoder(self):
        modules = []
        # modules.append(UnFlatten(4))
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
        return nn.Sequential(*modules)

    def get_final_layer(self):
        name_id = self.name_id
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
        h_dim = self.hidden_dims_dec[-1]
        return nn.Sequential(
            nn.ConvTranspose2d(h_dim, self.out_channels,
                               kernel_size=para_dict[name_id][0], stride=para_dict[name_id][1],
                               padding=para_dict[name_id][2], output_padding=para_dict[name_id][3]),
            # nn.Sigmoid()
        )


class Conv1dBuilder(Builder):
    def __init__(self, in_channels, in_size, hidden_dims, kernel_size,
                 stride, padding, output_padding, out_channels=None):
        super().__init__()
        self.input_size = in_size
        self.in_channels = in_channels
        self.output_channels = out_channels if out_channels is not None else in_channels
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def get_encoder(self):
        modules = []
        in_channels = self.in_channels
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, h_dim,
                              kernel_size=self.kernel_size, stride=self.stride,
                              padding=self.padding)
                )
            )
            in_channels = h_dim

        return nn.Sequential(*modules)

    def get_encoder_output_dim(self, *args):
        out_width = self.input_size
        # TODO:此处曾有报错
        if len(self.hidden_dims) == 0:
            return np.array([self.in_channels, out_width])
        for _ in self.hidden_dims:
            out_width = int(np.floor(
                (out_width - self.kernel_size + 2 * self.padding)
                / self.stride)) + self.padding
        return np.array([self.hidden_dims[-1], out_width])

    def get_decoder(self):
        reversed_dims = copy.deepcopy(self.hidden_dims)
        reversed_dims.reverse()

        modules = []
        for i in range(len(reversed_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose1d(reversed_dims[i], reversed_dims[i + 1],
                                   kernel_size=self.kernel_size, stride=self.stride,
                                   padding=self.padding, output_padding=self.output_padding),
                nn.BatchNorm1d(reversed_dims[i + 1]),
                nn.LeakyReLU())
            )
        return nn.Sequential(*modules)

    def get_final_layer(self):
        h_dim = self.hidden_dims[0] if len(self.hidden_dims) > 0 else self.in_channels
        return nn.Sequential(
            nn.ConvTranspose1d(h_dim, h_dim, kernel_size=self.kernel_size, stride=self.stride,
                               padding=self.padding, output_padding=self.output_padding),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(),
            nn.Conv2d(h_dim, out_channels=self.out_channels, kernel_size=self.kernel_size,
                      padding=self.padding),
            nn.ReLU()  # want output to be in [0, infty)
        )


class LinearBuilder(Builder):
    def __init__(self, input_size, hidden_dims, output_size=None):
        super(LinearBuilder, self).__init__()
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.output_size = output_size if output_size is not None else input_size

    def get_encoder(self):
        modules = []
        prev_size = self.input_size
        for h_dim in self.hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(prev_size, h_dim),
                nn.LeakyReLU())
            )
            prev_size = h_dim
        return nn.Sequential(*modules)

    def get_encoder_output_dim(self, *args):
        if len(self.hidden_dims) == 0:
            return np.array([self.input_size])
        return np.array([self.hidden_dims[-1]])

    def get_decoder(self):
        reverse_hdims = copy.deepcopy(self.hidden_dims)
        reverse_hdims.reverse()

        modules = []
        for i in range(len(reverse_hdims) - 1):
            modules.append(nn.Sequential(
                nn.Linear(reverse_hdims[i], reverse_hdims[i + 1]),
                nn.LeakyReLU())
            )
        return nn.Sequential(*modules)

    def get_final_layer(self):
        in_size = self.input_size if len(self.hidden_dims) == 0 else self.hidden_dims[0]
        return nn.Sequential(
            nn.Linear(in_size, self.output_size),
            nn.Sigmoid())

    def get_final_layer_no_activation(self):
        in_size = self.input_size if len(self.hidden_dims) == 0 else self.hidden_dims[0]
        return nn.Linear(in_size, self.output_size)