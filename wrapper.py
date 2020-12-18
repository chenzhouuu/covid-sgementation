import torch.nn as nn
from unet import UNet
from unet_2d import Unet_2D

class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()

        # def weight_init(m):
        #   if type(m) == nn.Linear:
        #     #nn.init.xavier_uniform(m.weight)
        #     m.weight.data.uniform_(0.0, 0.0)
        #     m.bias.data.fill_(0)

        def init_weights(self):
            for m in self.modules():
                if type(m) in init_set:
                    nn.init.kaiming_normal_(
                        m.weight.data, mode='fan_out', nonlinearity='relu', a=0
                    )
                    if m.bias is not None:
                        fan_in, fan_out = \
                            nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                        bound = 1 / math.sqrt(fan_out)
                        nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)
        un_output = self.unet(bn_output)
        fn_output = self.final(un_output)
        return fn_output


class UNet2DWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_batchnorm = nn.BatchNorm2d(kwargs['n_channels'])
        self.unet_2d = Unet_2D(**kwargs)

        def init_weights(self):
            for m in self.modules():
                if type(m) in init_set:
                    nn.init.kaiming_normal_(
                        m.weight.data, mode='fan_out', nonlinearity='relu', a=0
                    )
                    if m.bias is not None:
                        fan_in, fan_out = \
                            nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                        bound = 1 / math.sqrt(fan_out)
                        nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)
        fn_output = self.unet_2d(bn_output)
        return fn_output