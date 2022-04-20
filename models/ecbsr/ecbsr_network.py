import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.ecbsr.ecbsr_block import ECB, Conv3X3
except ModuleNotFoundError:
    from ecbsr_block import ECB, Conv3X3
from torch.quantization import QuantStub, DeQuantStub
from torch.nn.quantized import FloatFunctional

def create_model(args):
    return ECBSR(args)

class ECBSR(nn.Module):
    def __init__(self, args):
        super(ECBSR, self).__init__()
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
            backbone += [ECB(self.c_ecbsr, self.c_ecbsr, depth_multiplier=self.chns_exp, act_type=self.act_type, with_idt = self.with_idt, with_bn = self.with_bn)]
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

    x = torch.rand(1,1,128,128)
    model = ECBSR(args).eval()
    y0 = model(x)

    model.fuse_model()
    y1 = model(x)

    print(model)
    print(y0-y1)