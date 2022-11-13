import sys 
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
try:
    from ecbsr.ecbsr_block import ECB
except:
    from models.ecbsr.ecbsr_block import ECB

lrelu_value = 0.1
act = nn.LeakyReLU(lrelu_value)


def create_model(args, parent=False):
    return TEST_FMEN(args)


# class RRRB(nn.Module):
#     def __init__(self, n_feats):
#         super(RRRB, self).__init__()
#         self.rep_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

#     def forward(self, x):
#         out = self.rep_conv(x)

#         return out



# class ERB(nn.Module):
#     def __init__(self, n_feats):
#         super(ERB, self).__init__()
#         self.conv1 = RRRB(n_feats)
#         self.conv2 = RRRB(n_feats)

#     def forward(self, x):
#         res = self.conv1(x)
#         res = act(res)
#         res = self.conv2(res)

#         return res

class ERB(nn.Module):
    def __init__(self, n_feats):
        super(ERB, self).__init__()
        self.conv1 = ECB(n_feats, n_feats, depth_multiplier=2, act_type='lrelu', with_idt = True)
        self.conv2 = ECB(n_feats, n_feats, depth_multiplier=2, act_type='linear', with_idt = True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)

        return res

class HFAB(nn.Module):
    def __init__(self, n_feats, up_blocks, mid_feats):
        super(HFAB, self).__init__()
        
        self.squeeze = nn.Conv2d(n_feats, mid_feats, 3, 1, 1)
        convs = [ERB(mid_feats) for _ in range(up_blocks)]
        self.convs = nn.Sequential(*convs)
        self.excitate = nn.Conv2d(mid_feats, n_feats, 3, 1, 1)
    
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = act(self.squeeze(x))
        out = act(self.convs(out))
        out = self.excitate(out)
        out = self.sigmoid(out)
        out *= x

        return out


def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor
    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''

        return pixel_unshuffle(input, self.downscale_factor)


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()

        self.bn1 = nn.BatchNorm2d(n_feats)

        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv_un = PixelUnshuffle(2)
        self.con_ = conv(4 * f, f, kernel_size=1, padding=1)
        self.conv_sh = nn.PixelShuffle(2)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):


        c1_ = (self.conv1(x))
        c1 = self.conv_un(c1_)
        c1_p = F.max_pool2d(c1, kernel_size=7, stride=3)
        c1_p = self.relu(c1_p)
        c2 = self.con_(c1_p)
        c2 = self.relu(c2)
        c3 = F.interpolate(c2, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m

class TEST_FMEN(nn.Module):
    def __init__(self, args):
        super(TEST_FMEN, self).__init__()

        self.down_blocks = args.down_blocks

        # up_blocks = args.up_blocks
   
        n_feats = args.n_feats
        n_colors = args.colors
        scale = args.scale

        # define head module
        # self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)

        # warm up
        # self.warmup = nn.Sequential(
        #     nn.Conv2d(n_feats, n_feats, 3, 1, 1),
        #     # HFAB(n_feats, up_blocks[0], mid_feats-4)
        #     ESA(n_feats, nn.Conv2d)
        # )

        self.head = ECB(n_colors, n_feats, depth_multiplier=2, act_type='lrelu', with_idt = True)
        self.warmup = nn.Sequential(
            ECB(n_feats, n_feats, depth_multiplier=2, act_type='lrelu', with_idt = True),
            # HFAB(n_feats, up_blocks[0], mid_feats-4)
            ESA(n_feats, nn.Conv2d)
        )

        # define body module
        ERBs = [ERB(n_feats) for _ in range(self.down_blocks)]
        # HFABs  = [HFAB(n_feats, up_blocks[i+1], mid_feats) for i in range(self.down_blocks)]
        HFABs  = [ESA(n_feats, nn.Conv2d) for i in range(self.down_blocks)]
        
        self.ERBs = nn.ModuleList(ERBs)
        self.HFABs = nn.ModuleList(HFABs)
      
   

        self.lr_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        # define tail module
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_colors*(scale**2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )


    def forward(self, x):
        x = self.head(x)

        h = self.warmup(x)
        for i in range(self.down_blocks):
            h = self.ERBs[i](h)
            h = self.HFABs[i](h)
        
        h = self.lr_conv(h)

        h += x
        x = self.tail(h)

        return x 


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))





class Args:
    def __init__(self):
        self.down_blocks = 4
        self.up_blocks = [2, 1, 1, 1, 1]
        self.n_feats = 50
        self.mid_feats = 16

        self.scale = [4]
        self.rgb_range = 255
        self.n_colors = 3

if __name__ == '__main__':
    args = Args()
    model = TEST_FMEN(args).to('cuda')
    in_ = torch.randn(1, 3, round(720/args.scale[0]), round(1280/args.scale[0])).to('cuda')
    summary(model, in_)