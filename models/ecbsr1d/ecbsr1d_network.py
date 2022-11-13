import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.ecbsr.ecbsr_block import ECB, Conv3X3
except ModuleNotFoundError:
    from ecbsr_block import ECB, Conv3X3

try:
    from models.ecbsr1d.ecb1d_block import ECB1d_conv, ECB_all, ECB_all_test
except ModuleNotFoundError:
    from ecb1d_block import ECB1d_conv, ECB_all, ECB_all_test
    
from torch.quantization import QuantStub, DeQuantStub
from torch.nn.quantized import FloatFunctional
from torchsummaryX import summary

def create_model(args):
    return ECBSR1d(args)

class ECBSR1d(nn.Module):
    def __init__(self, args):
        super(ECBSR1d, self).__init__()
        self.m_ecbsr = args.m_ecbsr
        self.c_ecbsr = args.c_ecbsr
        self.scale = args.scale
        self.colors = args.colors
        self.chns_exp = 2.0
        self.with_idt = args.with_idt
        self.with_bn = args.with_bn
        self.act_type = args.act_type
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.backbone = None
        self.upsampler = None

        backbone = []
        backbone += [ECB(self.colors, self.c_ecbsr, depth_multiplier=self.chns_exp, act_type=self.act_type, with_idt = self.with_idt, with_bn = self.with_bn)]
        for i in range(self.m_ecbsr):
            backbone += [ECB_all(self.c_ecbsr, self.c_ecbsr, depth_multiplier=self.chns_exp, act_type=self.act_type, with_idt = self.with_idt, with_bn = self.with_bn)]
        backbone += [ECB(self.c_ecbsr, self.colors*self.scale*self.scale, depth_multiplier=2.0, act_type='linear', with_idt = self.with_idt, with_bn = self.with_bn)]
        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)
        self.shortcut = FloatFunctional()

    def fuse_model(self):
        ## reparam as plainsr
        for idx, blk in enumerate(self.backbone):
            if type(blk) == ECB:
                RK, RB  = blk.rep_params()
                conv3x3 = Conv3X3(blk.inp_planes, blk.out_planes, act_type=blk.act_type, with_bn=False)
                ## update weights & bias for conv3x3
                conv3x3.block[0].weight.data = RK
                conv3x3.block[0].bias.data   = RB
                ## update weights & bias for activation
                if blk.act_type == 'prelu':
                    conv3x3.block[1].weight = blk.act.weight
                ## update block for backbone
                self.backbone[idx] = conv3x3.to(RK.device)
        ## fused modules
        for m in self.modules():
            if type(m) == Conv3X3:
                if m.act_type == 'relu':
                    torch.quantization.fuse_modules(m.block, ['0', '1'], inplace=True)
    def forward(self, x):
        x = self.quant(x)
        y = self.shortcut.add(self.backbone(x), x.repeat(1, self.colors*self.scale*self.scale, 1, 1).contiguous())
        y = self.upsampler(y)
        y = torch.clamp(y, min=0.0, max=255.0)
        y = self.dequant(y)
        return y



class ECBSR1d_test(nn.Module):
    def __init__(self, args):
        super(ECBSR1d_test, self).__init__()
        self.m_ecbsr = args.m_ecbsr
        self.c_ecbsr = args.c_ecbsr
        self.scale = args.scale
        self.colors = args.colors
        self.chns_exp = 2.0
        self.with_idt = args.with_idt
        self.with_bn = args.with_bn
        self.act_type = args.act_type
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.backbone = None
        self.upsampler = None

        backbone = []
        backbone += [nn.Conv2d(self.colors, self.c_ecbsr, kernel_size=3, padding=1)]
        for i in range(self.m_ecbsr):
            backbone += [ECB_all_test(self.c_ecbsr, self.c_ecbsr, depth_multiplier=self.chns_exp, act_type=self.act_type, with_idt = self.with_idt, with_bn = self.with_bn)]
        backbone += [nn.Conv2d(self.c_ecbsr, self.colors*self.scale*self.scale, kernel_size=3, padding=1)]
        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)
        self.shortcut = FloatFunctional()

    def forward(self, x):
        x = self.quant(x)
        y = self.shortcut.add(self.backbone(x), x.repeat(1, self.colors*self.scale*self.scale, 1, 1).contiguous())
        y = self.upsampler(y)
        y = torch.clamp(y, min=0.0, max=255.0)
        y = self.dequant(y)
        return y

if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser(description='')
    args.m_ecbsr = 4
    args.c_ecbsr = 16
    args.with_idt = 1
    args.with_bn = 1
    args.act_type = 'relu'
    args.model = 'ecbsr'
    args.scale = 4
    args.colors = 1

  
    # model = ECBSR1d(args).eval().to('cuda')
    model = ECBSR1d_test(args).eval().to('cuda')
 

    in_ = torch.randn(1, 1, round(720/args.scale), round(1280/args.scale)).to('cuda')
    summary(model, in_)