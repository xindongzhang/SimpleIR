import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
from torch.nn.quantized import FloatFunctional

def create_model(args):
    return ESPCN(args)

class Conv2D(nn.Module):
    def __init__(self, inp_planes, out_planes, ksize=3, act_type='prelu', with_bn=False):
        super(Conv2D, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.ksize = ksize
        self.act_type = act_type
        self.with_bn = with_bn

        self.block = [nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=ksize, padding=ksize//2)]
        if self.with_bn:
            self.block += [nn.BatchNorm2d(self.out_planes)]
        ## activation selection
        if self.act_type == 'prelu':
            self.block += [nn.PReLU(num_parameters=self.out_planes)]
        elif self.act_type == 'relu':
            self.block += [nn.ReLU(inplace=True)]
        elif self.act_type == 'rrelu':
            self.block += [nn.RReLU(lower=-0.05, upper=0.05)]
        elif self.act_type == 'softplus':
            self.block += [nn.Softplus()]
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')
        ## initialize block
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        x = self.block(x)
        return x

class ESPCN(nn.Module):
    def __init__(self, args):
        super(ESPCN, self).__init__()

        self.scale = args.scale
        self.colors = args.colors
        self.with_bn = args.with_bn
        self.act_type = args.act_type
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.backbone = []
        ## first part
        self.backbone += [
            Conv2D(self.colors, 64, ksize=5, act_type=self.act_type),
            Conv2D(64, 32, ksize=3, act_type=self.act_type)
        ]
        ## last part
        self.backbone += [
            Conv2D(32, self.colors*self.scale*self.scale, ksize=3),
            nn.PixelShuffle(self.scale)
        ]
        self.backbone = nn.Sequential(*self.backbone)

    def fuse_model(self):
        for m in self.modules():
            if type(m) == Conv2D:
                if m.act_type == 'relu':
                    if self.with_bn:
                        torch.quantization.fuse_modules(m.block, ['0', '1', '2'], inplace=True)
                    else:
                        torch.quantization.fuse_modules(m.block, ['0', '1'], inplace=True)
                else:
                    if self.with_bn:
                        torch.quantization.fuse_modules(m.block, ['0', '1'], inplace=True)

    def forward(self, x):
        x = self.quant(x)
        y = self.backbone(x)
        y = torch.clamp(y, min=0.0, max=255.0)
        y = self.dequant(y)
        return y
    
if __name__ == '__main__':
    pass



