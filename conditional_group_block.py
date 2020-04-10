import torch
import torch.nn as nn
import torch.nn.functional as F

class CGB(nn.Module):
    def __init__(self, f_channels, out_channels, l_channels, hidden_channel=280, G=8):
        super(CGB, self).__init__()
        self.norm1 = CGNorm(f_channels, l_channels, hidden_channel, G)
        self.gconv1 = nn.Conv2d(f_channels, f_channels, kernel_size=3, padding=1, groups=G)
        self.norm2 = CGNorm(f_channels, l_channels, hidden_channel, G)
        self.gconv2 = nn.Conv2d(f_channels, out_channels, kernel_size=3, padding=1, groups=G)

        self.norm_mid = CGNorm(f_channels, l_channels, hidden_channel, G)
        self.gconv_mid = nn.Conv2d(f_channels, out_channels, kernel_size=3, padding=1, groups=G)

    def forward(self, f, m):
        x = self.gconv1(F.leaky_relu(self.norm1(f, m), 0.2))
        x = self.gconv2(F.leaky_relu(self.norm2(x, m), 0.2))
        r = self.gconv_mid(self.norm_mid(f, m))
        return x + r

class CGNorm(nn.Module):
    def __init__(self, f_channels, l_channels, hidden_channels, G=8):
        super(CGNorm, self).__init__()
        #self.norm = nn.SyncBatchNorm(f_channels, affine=True)
        self.norm = nn.InstanceNorm2d(f_channels, affine=True)
        self.gconv1 = nn.Conv2d(l_channels, hidden_channels, kernel_size=3, padding=1, groups=1) # G?
        self.gconv2_1 = nn.Conv2d(hidden_channels, f_channels, kernel_size=3, padding=1, groups=G)
        self.gconv2_2 = nn.Conv2d(hidden_channels, f_channels, kernel_size=3, padding=1, groups=G)

    def forward(self, f, m):
        x = self.norm(f)
        m = F.interpolate(m, x.size()[-2:], mode='nearest')
        m = F.relu(self.gconv1(m))
        gamma = self.gconv2_1(m)
        beta = self.gconv2_2(m)
        fo = gamma * x + beta
        return fo
