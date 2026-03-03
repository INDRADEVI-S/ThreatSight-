import torch.nn as nn
import torch

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx,_ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg, mx], dim=1)
        att = torch.sigmoid(self.conv(a))
        return x * att, att
