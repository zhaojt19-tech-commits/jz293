import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_layers import MaskedConv2d, CroppedConv2d


class CausalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, data_channels):
        super(CausalBlock, self).__init__()
        self.split_size = out_channels

        self.v_conv = CroppedConv2d(in_channels,
                                    2 * out_channels,
                                    (kernel_size // 2 + 1, kernel_size),
                                    padding=(kernel_size // 2 + 1, kernel_size // 2))
        self.v_fc = nn.Conv2d(in_channels,
                              2 * out_channels,
                              (1, 1))
        self.v_to_h = nn.Conv2d(2 * out_channels,
                                2 * out_channels,
                                (1, 1))

        self.h_conv = MaskedConv2d(in_channels,
                                   2 * out_channels,
                                   (1, kernel_size),
                                   mask_type='A',
                                   data_channels=data_channels,
                                   padding=(0, kernel_size // 2))
        self.h_fc = MaskedConv2d(out_channels,
                                 out_channels,
                                 (1, 1),
                                 mask_type='A',
                                 data_channels=data_channels)

    def forward(self, image):
        v_out, v_shifted = self.v_conv(image)
        v_out += self.v_fc(image)
        v_out_tanh, v_out_sigmoid = torch.split(v_out, self.split_size, dim=1)
        v_out = torch.tanh(v_out_tanh) * torch.sigmoid(v_out_sigmoid)

        h_out = self.h_conv(image)
        v_shifted = self.v_to_h(v_shifted)
        h_out += v_shifted
        h_out_tanh, h_out_sigmoid = torch.split(h_out, self.split_size, dim=1)
        h_out = torch.tanh(h_out_tanh) * torch.sigmoid(h_out_sigmoid)
        h_out = self.h_fc(h_out)

        return v_out, h_out


class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, data_channels, num_classes, H_latent, W_latent):
        super(GatedBlock, self).__init__()
        self.split_size = out_channels

        self.H_latent = H_latent
        self.W_latent = W_latent

        self.v_conv = CroppedConv2d(in_channels,
                                    2 * out_channels,
                                    (kernel_size // 2 + 1, kernel_size),
                                    padding=(kernel_size // 2 + 1, kernel_size // 2))
        self.v_fc = nn.Conv2d(in_channels,
                              2 * out_channels,
                              (1, 1))
        self.v_to_h = MaskedConv2d(2 * out_channels,
                                   2 * out_channels,
                                   (1, 1),
                                   mask_type='B',
                                   data_channels=data_channels)

        self.h_conv = MaskedConv2d(in_channels,
                                   2 * out_channels,
                                   (1, kernel_size),
                                   mask_type='B',
                                   data_channels=data_channels,
                                   padding=(0, kernel_size // 2))
        self.h_fc = MaskedConv2d(out_channels,
                                 out_channels,
                                 (1, 1),
                                 mask_type='B',
                                 data_channels=data_channels)

        self.h_skip = MaskedConv2d(out_channels,
                                   out_channels,
                                   (1, 1),
                                   mask_type='B',
                                   data_channels=data_channels)

        # 对标签将独热码映射成他图片尺度
        self.fc_class_0 = nn.Linear(num_classes[0], H_latent * W_latent)
        self.fc_class_1 = nn.Linear(num_classes[1], H_latent * W_latent)
        self.conv_label0 = nn.Conv2d(1,
                              out_channels,
                              (1, 1))
        self.conv_label1 = nn.Conv2d(1,
                              out_channels,
                              (1, 1))


    def forward(self, x):
        v_in, h_in, skip, label_onehot_0, label_onehot_1 = x[0], x[1], x[2], x[3], x[4]
        
        # 线性变换
        label_map_0 = self.fc_class_0(label_onehot_0).reshape(-1, 1, self.H_latent, self.W_latent)
        label_map_1 = self.fc_class_1(label_onehot_1).reshape(-1, 1, self.H_latent, self.W_latent)

        # 卷积将通道数由1改为60
        label_map_0 = self.conv_label0(label_map_0)
        label_map_1 = self.conv_label1(label_map_1)

        v_out, v_shifted = self.v_conv(v_in)
        v_out += self.v_fc(v_in)
        v_out_tanh, v_out_sigmoid = torch.split(v_out, self.split_size, dim=1)
        v_out = torch.tanh(v_out_tanh + label_map_0 + label_map_1) * torch.sigmoid(v_out_sigmoid + label_map_0 + label_map_1)

        h_out = self.h_conv(h_in)
        v_shifted = self.v_to_h(v_shifted)
        h_out += v_shifted
        h_out_tanh, h_out_sigmoid = torch.split(h_out, self.split_size, dim=1)
        h_out = torch.tanh(h_out_tanh + label_map_0 + label_map_1) * torch.sigmoid(h_out_sigmoid + label_map_0 + label_map_1)

        # skip connection
        skip = skip + self.h_skip(h_out)

        h_out = self.h_fc(h_out)

        # residual connections
        h_out = h_out + h_in
        v_out = v_out + v_in

        return {0: v_out, 1: h_out, 2: skip, 3: label_onehot_0, 4: label_onehot_1}


class PixelCNN(nn.Module):
    def __init__(self, cfg, num_classes, H_latent, W_latent):
        super(PixelCNN, self).__init__()
        

        DATA_CHANNELS = 1

        self.H_latent = H_latent
        self.W_latent = W_latent
        self.num_label = len(num_classes) # 标签数目
        self.hidden_fmaps = cfg.hidden_fmaps
        self.color_levels = cfg.color_levels

        self.causal_conv = CausalBlock(DATA_CHANNELS,
                                       cfg.hidden_fmaps,
                                       cfg.causal_ksize,
                                       data_channels=DATA_CHANNELS)

        self.hidden_conv = nn.Sequential(
            *[GatedBlock(cfg.hidden_fmaps, cfg.hidden_fmaps, cfg.hidden_ksize, DATA_CHANNELS, num_classes, H_latent, W_latent) for _ in range(cfg.hidden_layers)]
        )
        

        # 对标签将独热码映射成他图片尺度
        self.fc_class_0 = nn.Linear(num_classes[0], H_latent * W_latent)
        self.fc_class_1 = nn.Linear(num_classes[1], H_latent * W_latent)
        self.conv_label0 = nn.Conv2d(1,
                              cfg.hidden_fmaps,
                              (1, 1))
        self.conv_label1 = nn.Conv2d(1,
                              cfg.hidden_fmaps,
                              (1, 1))

        self.out_hidden_conv = MaskedConv2d(cfg.hidden_fmaps,
                                            cfg.out_hidden_fmaps,
                                            (1, 1),
                                            mask_type='B',
                                            data_channels=DATA_CHANNELS)

        self.out_conv = MaskedConv2d(cfg.out_hidden_fmaps,
                                     DATA_CHANNELS * cfg.color_levels,
                                     (1, 1),
                                     mask_type='B',
                                     data_channels=DATA_CHANNELS)

    def forward(self, image, label_onehot_0, label_onehot_1):
        count, data_channels, height, width = image.size()

        v, h = self.causal_conv(image)

        _, _, out, _, _ = self.hidden_conv({0: v,
                                         1: h,
                                         2: image.new_zeros((count, self.hidden_fmaps, height, width), requires_grad=True),
                                         3: label_onehot_0,
                                         4: label_onehot_1}).values()

        # 线性变换
        label_map_0 = self.fc_class_0(label_onehot_0).reshape(-1, 1, self.H_latent, self.W_latent)
        label_map_1 = self.fc_class_1(label_onehot_1).reshape(-1, 1, self.H_latent, self.W_latent)

        # 卷积将通道数由1改为60
        label_map_0 = self.conv_label0(label_map_0)
        label_map_1 = self.conv_label1(label_map_1)

        # add label bias
        out += label_map_0
        out += label_map_1
        out = F.relu(out)
        out = F.relu(self.out_hidden_conv(out))
        out = self.out_conv(out)

        out = out.view(count, self.color_levels, data_channels, height, width)

        return out

    def sample(self, shape, count, label_onehot_0=None, label_onehot_1=None, images=None, device='cuda'):
        channels, height, width = shape
        if(images==None):
            samples = torch.zeros(count, *shape).to(device)
        else:
            samples = images
        with torch.no_grad():
            for i in range(height):
                for j in range(width):
                    for c in range(channels):
                        unnormalized_probs = self.forward(samples, label_onehot_0, label_onehot_1)
                        pixel_probs = torch.softmax(unnormalized_probs[:, :, c, i, j], dim=1)
                        sampled_levels = torch.multinomial(pixel_probs, 1).squeeze().float()
                        samples[:, c, i, j] = sampled_levels

        return samples
