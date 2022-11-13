import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier, with_bn=False):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.with_bn = with_bn
        
        if self.with_bn:
            self.bn = nn.BatchNorm2d(num_features=out_planes)

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
            
        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
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
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        if self.with_bn:
            y1 = self.bn(y1)
        return y1
    
    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1-conv3x3':
            # re-param conv kernel
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1,) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1,) + b1
        
        if self.with_bn:
            v = torch.sqrt(self.bn.running_var + self.bn.eps)
            m = self.bn.running_mean
            s = self.bn.weight
            b = self.bn.bias
            RK = (s/v).reshape(self.out_planes, 1, 1, 1) * RK 
            RB = (s/v) * (RB - m) + b
        return RK, RB

def pad_tensor(t, pattern):
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0)
    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern

    return t

class RRRB(nn.Module):
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.
    Diagram:
        ---Conv1x1--Conv3x3-+-Conv1x1--+--
                   |________|
         |_____________________________|
    Args:
        n_feats (int): The number of feature maps.
        ratio (int): Expand ratio.
    """

    def __init__(self, n_feats, ratio=2):
        super(RRRB, self).__init__()
        self.expand_conv = nn.Conv2d(n_feats, ratio*n_feats, 1, 1, 0)
        self.fea_conv = nn.Conv2d(ratio*n_feats, ratio*n_feats, 3, 1, 0)
        self.reduce_conv = nn.Conv2d(ratio*n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
  
        out = self.expand_conv(x)
        out_identity = out
        
        # explicitly padding with bias for reparameterizing in the test phase
        b0 = self.expand_conv.bias
        out = pad_tensor(out, b0)

        out = self.fea_conv(out) + out_identity
        out = self.reduce_conv(out)
        out += x

        return out


class SESR(nn.Module):
    def __init__(self, n_feats, ratio=2):
        super(SESR, self).__init__()
        self.expand_conv = nn.Conv2d(n_feats, ratio*n_feats, 3, 1, 1)
        self.squeeze_conv = nn.Conv2d(ratio*n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        
        identity = x
        out = self.expand_conv(x)
        out = self.squeeze_conv(out)
        
        out += identity

        return out


class ECB(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier, conv3_type='conv3', act_type='prelu', with_idt = False, with_bn = False):
        super(ECB, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        self.with_bn = with_bn

        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        if with_bn:
            self.conv3x3 = nn.Sequential(
                nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_planes)
            )
        elif conv3_type == 'conv3':
            self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        elif conv3_type == 'RRRB':
            self.conv3x3 = RRRB(self.inp_planes)
        else:
            self.conv3x3 = SESR(self.inp_planes, ratio=2)
        
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier, self.with_bn)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.inp_planes, self.out_planes, -1, self.with_bn)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.inp_planes, self.out_planes, -1, self.with_bn)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes, -1, self.with_bn)
        
        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'lrelu':
            self.act = nn.LeakyReLU(0.1)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

        

    def forward(self, x):
        if self.training:
    
            y = self.conv3x3(x)     + \
                self.conv1x1_sbx(x) + \
                self.conv1x1_sby(x) + \
                self.conv1x1_lpl(x) + \
                  self.conv1x1_3x3(x) 

                  
            # y = self.conv3x3(x)     + \
            #     self.rrrb(x) + \
            #     self.conv1x1_sbx(x) + \
            #     self.conv1x1_sby(x) + \
            #     self.conv1x1_lpl(x)
            if self.with_idt:
                y += x
        else:
            RK, RB = self.rep_params()
            y = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1) 
        if self.act_type != 'linear':
            y = self.act(y)
        return y

    # def forward(self, x):

    #     y = self.conv3x3(x)   + \
    #         self.conv1x1_sbx(x) + \
    #         self.conv1x1_sby(x) + \
    #         self.conv1x1_lpl(x)

    #     if self.with_idt:
    #         y += x

    #     return y 


    def rep_params(self):
        if self.with_bn:
            K0, B0 = self.conv3x3[0].weight, self.conv3x3[0].bias
            v = torch.sqrt(self.conv3x3[1].running_var + self.conv3x3[1].eps)
            m = self.conv3x3[1].running_mean
            s = self.conv3x3[1].weight
            b = self.conv3x3[1].bias
            K0 = (s/v).reshape(self.out_planes, 1, 1, 1) * K0 
            B0 = (s/v) * (B0 - m) + b 
        else:
            K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        RK, RB = (K0+K1+K2+K3+K4), (B0+B1+B2+B3+B4)

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB

if __name__ == '__main__':

    # # # test seq-conv
    # x = torch.randn(1, 3, 5, 5).cuda()
    # conv = SeqConv3x3('conv1x1-conv3x3', 3, 3, 2, with_bn=True).cuda().eval()
    # y0 = conv(x)
    # RK, RB = conv.rep_params()
    # y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    # print(y0-y1)

    # test ecb
    x = torch.randn(1, 3, 5, 5).cuda() * 200
    ecb = ECB(3, 3, 2, act_type='linear', with_idt=True, with_bn=True).cuda().eval()
    y0 = ecb(x)

    RK, RB = ecb.rep_params()
    y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    print(y0-y1)