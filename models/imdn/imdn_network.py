import torch.nn as nn

try:
    from models.imdn import imdn_block as B
except ModuleNotFoundError:
    import imdn_block as B
import torch

def create_model(args):
    return IMDN(args)

class IMDN(nn.Module):
    def __init__(self, args):
        super(IMDN, self).__init__()
        ## network parameters
        in_nc = args.colors
        out_nc = args.colors
        num_modules = args.num_modules
        nf = args.nf
        upscale = args.scale

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        # IMDBs
        self.IMDB1 = B.IMDModule(in_channels=nf)
        self.IMDB2 = B.IMDModule(in_channels=nf)
        self.IMDB3 = B.IMDModule(in_channels=nf)
        self.IMDB4 = B.IMDModule(in_channels=nf)
        self.IMDB5 = B.IMDModule(in_channels=nf)
        self.IMDB6 = B.IMDModule(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output

if __name__ == '__main__':
    import numpy as np
    from torch.autograd import Variable
    import argparse
    args = argparse.ArgumentParser(description='')
    args.colors = 1
    args.num_modules = 6
    args.nf = 16
    args.scale = 4

    model = IMDN(args).cpu().eval()
    model.cpu()
    model.eval()
