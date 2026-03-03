import torch.nn as nn
import torch

def Dblock(in_c, out_c, stride=2, norm=True):
    import torch.nn.utils as utils
    layers = [utils.spectral_norm(nn.Conv2d(in_c,out_c,4,stride,1))]
    if norm: layers += [nn.BatchNorm2d(out_c)]
    layers += [nn.LeakyReLU(0.2,inplace=True)]
    return nn.Sequential(*layers)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.l1 = Dblock(in_ch, base, stride=2, norm=False)
        self.l2 = Dblock(base, base*2, stride=2)
        self.l3 = Dblock(base*2, base*4, stride=2)
        self.l4 = Dblock(base*4, base*8, stride=1)
        self.out = torch.nn.utils.spectral_norm(nn.Conv2d(base*8,1,4,1,1))
    def forward(self, x):
        h = self.l1(x); h = self.l2(h); h = self.l3(h); h = self.l4(h)
        return self.out(h)
