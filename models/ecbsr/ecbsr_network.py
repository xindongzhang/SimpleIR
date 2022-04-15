import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ecbsr.ecb import ECB

def create_model(args):
    return ECBSR(args)

class ECBSR(nn.Module):
    # def __init__(self, module_nums, channel_nums, with_idt, act_type, scale, colors, step_wise=False):
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
        self.backbone = None
        self.upsampler = None

        backbone = []
        backbone += [ECB(self.colors, self.c_ecbsr, depth_multiplier=self.chns_exp, act_type=self.act_type, with_idt = self.with_idt, with_bn = self.with_bn)]
        for i in range(self.m_ecbsr):
            backbone += [ECB(self.c_ecbsr, self.c_ecbsr, depth_multiplier=self.chns_exp, act_type=self.act_type, with_idt = self.with_idt, with_bn = self.with_bn)]
        backbone += [ECB(self.c_ecbsr, self.colors*self.scale*self.scale, depth_multiplier=2.0, act_type='linear', with_idt = self.with_idt, with_bn = self.with_bn)]
        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        y = self.backbone(x) + x
        y = self.upsampler(y)
        return y
