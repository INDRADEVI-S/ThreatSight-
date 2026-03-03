import torch.nn as nn
import torch
from .attention import SpatialAttention

def conv(in_c, out_c, ks=3, st=1, pad=1):
    return nn.Sequential(nn.Conv2d(in_c,out_c,ks,st,pad,bias=False), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2,inplace=True))

def deconv(in_c, out_c, ks=4, st=2, pad=1):
    return nn.Sequential(nn.ConvTranspose2d(in_c,out_c,ks,st,pad,bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.e1 = conv(in_ch, base, ks=3, st=1, pad=1)
        self.e2 = conv(base, base*2, ks=3, st=2, pad=1)
        self.e3 = conv(base*2, base*4, ks=3, st=2, pad=1)
        self.e4 = conv(base*4, base*8, ks=3, st=2, pad=1)
        self.b  = conv(base*8, base*8, ks=3, st=2, pad=1)
        self.sa3 = SpatialAttention(); self.sa4 = SpatialAttention()
        self.d1 = deconv(base*8, base*8)
        self.d2 = deconv(base*16, base*4)
        self.d3 = deconv(base*8, base*2)
        self.d4 = deconv(base*4, base)
        self.out = nn.Sequential(nn.Conv2d(base*2, in_ch, 3, padding=1), nn.Tanh())
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b  = self.b(e4)
        d1 = self.d1(b)
        e4a, a4 = self.sa4(e4)
        d2 = self.d2(torch.cat([d1, e4a], dim=1))
        e3a, a3 = self.sa3(e3)
        d3 = self.d3(torch.cat([d2, e3a], dim=1))
        d4 = self.d4(torch.cat([d3, e2], dim=1))
        out = self.out(torch.cat([d4, e1], dim=1))
        return out, (a3, a4)
