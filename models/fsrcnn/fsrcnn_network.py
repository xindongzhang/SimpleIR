import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
from torch.nn.quantized import FloatFunctional

def create_model(args):
    return FSRCNN(args)

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

class FSRCNN(nn.Module):
    def __init__(self, args):
        super(FSRCNN, self).__init__()

        m = 4
        s = 12
        d = 56

        self.scale = args.scale
        self.colors = args.colors
        self.with_bn = args.with_bn
        self.act_type = args.act_type
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.backbone = []
        ## first part
        self.backbone += [Conv2D(self.colors, d, ksize=5, act_type=self.act_type)]
        ## mid part
        self.backbone += [Conv2D(d, s, ksize=1, act_type=self.act_type)]
        for i in range(m):
            self.backbone += [Conv2D(s, s, ksize=3, act_type=self.act_type)]
        self.backbone += [Conv2D(s, d, ksize=1, act_type=self.act_type)]
        ## last part
        self.backbone += [nn.ConvTranspose2d(d, self.colors, kernel_size=9, stride=self.scale, padding=9//2,output_padding=self.scale-1)]
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
    import numpy as np
    from torch.autograd import Variable
    import argparse
    args = argparse.ArgumentParser(description='')
    args.m_plainsr = 4
    args.c_plainsr = 16
    args.with_idt = 1
    args.with_bn = 1
    args.act_type = 'relu'
    args.model = 'plainsr'
    args.scale = 4
    args.colors = 1

    model = PlainSR(args).eval()

    from tinynn.converter import TFLiteConverter
    model.cpu()
    model.eval()

    
    dummy_input = torch.rand((1, 1, 256, 256))

    torch.onnx.export(model, dummy_input, './plainsr.onnx', export_params=True, opset_version=10, do_constant_folding=True, input_names = ['input'], output_names = ['output'])

    converter = TFLiteConverter(model, dummy_input, './plainsr.tflite', input_transpose=True, output_transpose=True, group_conv_rewrite=True)
    converter.convert()

    from tinynn.graph.tracer import model_tracer, trace

    with model_tracer():
        # Prapare the model
        # It's okay to put the construction of the model out of the
        # with-block, but actually leave it here would be better.
        # The latter one guarantees that the arguments that is used
        # to build the model is caught, while the other one doesn't.

        # After tracing the model, we will get a TraceGraph object
        graph = trace(model, dummy_input)

        # We can use it to generate the code for the original model
        # But be careful that it has to be in the with-block.
        graph.generate_code('gen_plainsr.py', 'plainsr.pth', 'plainsr')