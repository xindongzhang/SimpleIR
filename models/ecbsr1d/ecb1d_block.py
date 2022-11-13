import torch
import torch.nn as nn
import torch.nn.functional as F



class SeqConv1d(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier=1, with_bn=False):
        super(SeqConv1d, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.with_bn = with_bn
        
        if self.with_bn:
            self.bn = nn.BatchNorm2d(num_features=out_planes)

        if self.type == 'conv1x1-conv3x1':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=(3,1))
            self.k1 = conv1.weight
            self.b1 = conv1.bias

        if self.type == 'conv1x1-conv1x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=(1,3))
            self.k1 = conv1.weight
            self.b1 = conv1.bias
            
        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=(1,1), padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 1, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 2.0
                self.mask[i, 0, 0, 2] = -2.0
                
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=(1,1), padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 1), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 2.0
                self.mask[i, 0, 2, 0] = -2.0
                
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)


    def forward(self, x):
        if self.type == 'conv1x1-conv1x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 0, 0), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        elif self.type == 'conv1x1-conv3x1':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (0, 0, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        elif self.type == 'conv1x1-sobelx':
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 0, 0), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
  
        elif self.type == 'conv1x1-sobely':
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (0, 0, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)

        if self.with_bn:
            y1 = self.bn(y1)
        return y1    

class ECB1d(nn.Module):
    def __init__(self, inp_planes, out_planes, type='x_axis', depth_multiplier=2,act_type='prelu', with_idt = False, with_bn = False):
        super(ECB1d, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        self.with_bn = with_bn
        self.type = type

        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        if with_bn:
            self.conv3x3 = nn.Sequential(
                nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_planes)
            )
        if self.type == 'x_axis':
            self.conv1x1_1x3 = SeqConv1d('conv1x1-conv1x3', self.inp_planes, self.out_planes, self.depth_multiplier, self.with_bn)
            self.conv1x1_sbx = SeqConv1d('conv1x1-sobelx', self.inp_planes, self.out_planes, -1, self.with_bn)
        else:
            self.conv1x1_3x1 = SeqConv1d('conv1x1-conv3x1', self.inp_planes, self.out_planes, self.depth_multiplier, self.with_bn)
            self.conv1x1_sby = SeqConv1d('conv1x1-sobely', self.inp_planes, self.out_planes, -1, self.with_bn)
        # self.conv1x1_lpl = SeqConv1d('conv1x1-laplacian', self.inp_planes, self.out_planes, -1, self.with_bn)
        
        self.act = nn.LeakyReLU(0.1)
        

    def forward(self, x):
     
        if self.type == 'x_axis':
            y = self.conv1x1_1x3(x) + \
                self.conv1x1_sbx(x) 
        
        else:
            y = self.conv1x1_3x1(x) + \
                self.conv1x1_sby(x) 

        return y
             

class ECB1d_conv(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier=2,act_type='prelu', with_idt = False, with_bn = False):
        super(ECB1d_conv, self).__init__()
        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        self.with_bn = with_bn

        self.conv_x = ECB1d(self.inp_planes, self.out_planes,  type='x_axis')
        self.conv_y = ECB1d(self.inp_planes, self.out_planes,  type='y_axis')

    def forward(self, x):
        oup = self.conv_x(x)
        oup = self.conv_y(oup)

        return oup 

class LKA_noatt(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return attn

class ECB_all(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier=2,act_type='prelu', with_idt = False, with_bn = False):
        super(ECB_all, self).__init__()
        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        self.with_bn = with_bn

        if self.inp_planes == self.out_planes:
            self.lka = LKA_noatt(self.inp_planes)

        self.ecb1d = ECB1d_conv(self.inp_planes, self.out_planes)
        


    def forward(self, x):
        oup = self.ecb1d(x)
        if self.inp_planes == self.out_planes:
            oup = self.lka(oup)
        return oup

class ECB_all_test(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier=2,act_type='prelu', with_idt = False, with_bn = False):
        super(ECB_all_test, self).__init__()
        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        self.with_bn = with_bn

        if self.inp_planes == self.out_planes:
            self.lka = LKA_noatt(self.inp_planes)

        self.ecb1d_x = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=(1,3), padding=(0,1))
        self.ecb1d_y = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=(3,1), padding=(1,0))
        


    def forward(self, x):
        oup = self.ecb1d_x(x)
        oup = self.ecb1d_y(oup)
        if self.inp_planes == self.out_planes:
            oup = self.lka(oup)
        return oup



if __name__ == '__main__':
    x0 = torch.randn(1, 64, 28, 28)
    con1d_x = SeqConv1d('conv1x1-sobelx', 64, 64)
    con1d_y = SeqConv1d('conv1x1-sobely', 64, 64)
    conv1d = ECB1d_conv(64, 64)
    conv_atten = ECB_all(64, 64)
    conv_test = ECB_all_test(64, 64)

    y0 = con1d_x(x0)
    print(y0.shape)

    y0 = con1d_y(x0)
    print(y0.shape)

    y0 = conv1d(x0)
    print(y0.shape)

    y0 = conv_atten(x0)
    print(y0.shape)

    y0 = conv_test(x0)
    print(y0.shape)

    