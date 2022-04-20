import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
from torch.nn.quantized import FloatFunctional

def create_model(args):
    return PlainSR(args)

class Conv3X3(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type='prelu', with_bn=False):
        super(Conv3X3, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        self.with_bn = with_bn

        self.block = [nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)]
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

class PlainSR(nn.Module):
    def __init__(self, args):
        super(PlainSR, self).__init__()
        self.m_plainsr = args.m_plainsr
        self.c_plainsr = args.c_plainsr
        self.scale = args.scale
        self.colors = args.colors
        self.with_bn = args.with_bn
        self.act_type = args.act_type
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.backbone = None
        self.upsampler = None

        backbone = []
        backbone += [Conv3X3(inp_planes=self.colors, out_planes=self.c_plainsr, act_type=self.act_type, with_bn=self.with_bn)]
        for i in range(self.m_plainsr):
            backbone += [Conv3X3(inp_planes=self.c_plainsr, out_planes=self.c_plainsr, act_type=self.act_type, with_bn=self.with_bn)]
        backbone += [Conv3X3(inp_planes=self.c_plainsr, out_planes=self.colors*self.scale*self.scale, act_type='linear', with_bn=self.with_bn)]
        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)
        self.shortcut = FloatFunctional()

    def fuse_model(self):
        for m in self.modules():
            if type(m) == Conv3X3:
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
        y = self.shortcut.add(self.backbone(x), x.repeat(1, self.colors*self.scale*self.scale, 1, 1).contiguous())
        y = self.upsampler(y)
        y = torch.clamp(y, min=0.0, max=255.0)
        y = self.dequant(y)
        return y