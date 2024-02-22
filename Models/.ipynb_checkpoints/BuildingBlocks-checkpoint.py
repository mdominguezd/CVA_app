import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Function

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, resunet = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.resunet = resunet
        
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        if resunet:
            # Identity mapping
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 0, stride = 1, bias = False)

    def forward(self, x):
        y = self.double_conv(x)

        if self.resunet:
            s = self.shortcut(x)
            y = y + s
        return y


class Down(nn.Module):
    """Downsampling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, resunet = False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resunet = resunet
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, resunet = resunet)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, attention = False, resunet = False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resunet = resunet
        self.bilinear = bilinear
        self.attention = attention

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, resunet = resunet)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, resunet = resunet)
        
        if attention:
            self.attn = Attention_block(in_channels//2, in_channels//2, in_channels//4)

    def forward(self, x1, x2):
        
        a1 = self.up(x1)
        
        diffY = x2.size()[2] - a1.size()[2]
        diffX = x2.size()[3] - a1.size()[3]
        
        a1 = F.pad(a1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        if self.attention:
            x2 = self.attn(g=a1,x=x2)

        x = torch.cat([x2, a1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    """
        Final convolutional layer used for segmentation purposes.
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.in_channels =  in_channels
        self.out_channels =    out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class OutDisc(nn.Module):
    def __init__(self, in_feat, mid_layers):
        super(OutDisc, self).__init__()
        self.disc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_feat, out_features=mid_layers, bias = False),
            nn.ReLU(),
            nn.Linear(in_features = mid_layers, out_features = mid_layers//2, bias = False),
            nn.ReLU(),
            nn.Linear(in_features = mid_layers//2, out_features = 1, bias = False)
        )

    def forward(self, x):
        return self.disc(x)

# https://github.com/CuthbertCai/pytorch_DANN/tree/master

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

############ OPTIONALS ##############


### Possible attention block to be added to unet (REFERENCE: https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py)

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        
        super(Attention_block,self).__init__()
        
        self.F_g = F_g
        self.F_l = F_l
        self.F_int = F_int
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

