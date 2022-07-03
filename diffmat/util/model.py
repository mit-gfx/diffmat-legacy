import numpy as np
import torch as th
import torch.nn as nn
from torchvision.models.vgg import vgg19

from diffmat.util.descriptor import TextureDescriptor

# define network model
class ParamPredNet(nn.Module):
    def __init__(self, in_channels, in_height, in_width, num_params):
        super(ParamPredNet, self).__init__()
        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width
        self.num_params = num_params
        self.leaky_relu_threshold = 0.2

    def build_fc_after_td(self, device, num_pyramid, fix_conv=True):
        self.td_length = 610304
        self.multiplier = 3
        self.td = TextureDescriptor(device)
        self.num_pyramid = num_pyramid
        self.fc_max = 960
        self.active_fc_num = min(self.num_params*self.multiplier, self.fc_max)
        
        if fix_conv is True:
            for param in self.td.parameters():
                param.requires_grad = False  # fix vgg network parameters

        self.fc_net = nn.Sequential(
            # fully convolution (input spatial size / 16, num_filters)
            nn.Linear(self.td_length * (self.num_pyramid + 1), self.active_fc_num),
            nn.LeakyReLU(self.leaky_relu_threshold, inplace=True),
            nn.Linear(self.active_fc_num, self.active_fc_num),
            nn.LeakyReLU(self.leaky_relu_threshold, inplace=True),
            nn.Linear(self.active_fc_num, self.num_params),
            nn.Sigmoid()
        )

        return self.td
        
    def build_fc_after_vgg(self, device, fix_conv=True):
        # get VGG19 feature network in evaluation mode
        self.vgg = vgg19(True).features.to(device)
        self.vgg_length = 512 * 16 * 16
        self.multiplier = 3
        self.fc_max = 960
        self.active_fc_num = min(self.num_params*self.multiplier, self.fc_max)

        if fix_conv:
            for param in self.vgg.parameters():
                param.requires_grad = False  # fix vgg network parameters

        # change max pooling to average pooling
        for i, x in enumerate(self.vgg):
            if isinstance(x, nn.MaxPool2d):
                self.vgg[i] = nn.AvgPool2d(kernel_size=2)

        self.fc_net = nn.Sequential(
            # fully convolution (input spatial size / 16, num_filters)
            nn.Linear(self.vgg_length, self.active_fc_num),
            nn.LeakyReLU(self.leaky_relu_threshold, inplace=True),
            nn.Linear(self.active_fc_num, self.active_fc_num),
            nn.LeakyReLU(self.leaky_relu_threshold, inplace=True),
            nn.Linear(self.active_fc_num, self.num_params),
            nn.Sigmoid()
        )

    def forward(self, x):
        has_vgg = getattr(self, "vgg", None)
        has_td = getattr(self, "td_length", None)
        if has_vgg:
            vgg_feature = self.vgg(x)
            vgg_feature = vgg_feature.flatten(start_dim=1)
            fc_out = self.fc_net(vgg_feature)
        elif has_td:
            td_out = self.td.eval_CHW_tensor(x)
            if self.num_pyramid:
                for scale in range(self.num_pyramid):
                    td_out_ = self.td.eval_CHW_tensor(nn.functional.interpolate(x, scale_factor = 1.0/(2.0**(scale+1)), mode='bilinear', align_corners=True))
                    td_out = th.cat([td_out, td_out_], dim=1)
            fc_out = self.fc_net(td_out)
        else:
            conv_out = self.conv_net(x)
            conv_out = conv_out.view((conv_out.shape[0], -1))
            fc_out = self.fc_net(conv_out)
                
        return fc_out