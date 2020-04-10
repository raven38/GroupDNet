import torch
import torch.nn as nn
import torch.nn.functional as F

from conditional_group_block import CGB

class Encoder(nn.Module):
    def __init__(self, in_channels, C, G):
        super(Encoder, self).__init__()
        # refer Figure 8

        conv_base = lambda x, y, g: nn.Sequential(nn.Conv2d(x, y, kernel_size=3, stride=2, padding=1, groups=g), nn.InstanceNorm2d(y), nn.LeakyReLU(0.2))
        self.net = nn.Sequential(
            conv_base(in_channels, 1*C*G, G),
            conv_base(1*C*G, 2*C*G, G),
            conv_base(2*C*G, 4*C*G, G),
            conv_base(4*C*G, 8*C*G, G),
            conv_base(8*C*G, 8*C*G, G),
            conv_base(8*C*G, 8*C*G, G),
            )
        self.tail_mu = nn.Conv2d(8*C*G, 8*G, kernel_size=3, stride=1, padding=1, groups=G)
        self.tail_sigma = nn.Conv2d(8*C*G, 8*G, kernel_size=3, stride=1, padding=1, groups=G)

    def forward(self, x):
        x = self.net(x)
        mu = self.tail_mu(x)
        sigma = self.tail_sigma(x)
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, l_channels, C, Gs):
        super(Decoder, self).__init__()
        # refer Figure 9
        class cgb_base(nn.Module):
            def __init__(self, in_channels, out_channels, l_channels, kernel_size=3, padding=1, groups=1):
                super(cgb_base, self).__init__()
                self.cgb = CGB(in_channels, out_channels, l_channels, G=groups)
                self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

            def forward(self, x, m):
                x = self.cgb(x, m)
                x = self.upsample(x)
                return x

        self.layer1 = cgb_base(in_channels, 16*C, l_channels, groups=Gs[0])
        self.layer2 = cgb_base(16*C, 16*C, l_channels, groups=Gs[1])
        self.layer3 = cgb_base(16*C, 16*C, l_channels, groups=Gs[2])
        self.layer4 = cgb_base(16*C, 8*C, l_channels, groups=Gs[3])
        self.layer5 = cgb_base(8*C, 4*C, l_channels, groups=Gs[4])
        self.layer6 = cgb_base(4*C, 2*C, l_channels, groups=Gs[5])
        self.layer7 = CGB(2*C, 1*C, l_channels, G=Gs[6])
        self.layer8 = nn.Conv2d(1*C, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, m):
        x = self.layer1(x, m)
        x = self.layer2(x, m)
        x = self.layer3(x, m)
        x = self.layer4(x, m)
        x = self.layer5(x, m)
        x = self.layer6(x, m)
        x = self.layer7(x, m)
        x = self.layer8(F.leaky_relu(x, 0.2))
        return torch.tanh(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        # refer Figure 7
        conv_base = lambda x, y, z: nn.Sequential(nn.Conv2d(x, y, kernel_size=4, padding=2, stride=z), nn.InstanceNorm2d(y), nn.LeakyReLU(0.2))
        self.net_a = nn.Sequential(
            conv_base(in_channels, 64, 2),
            conv_base(64, 128, 2),
            conv_base(128, 256, 2),
            conv_base(256, 512, 1),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=2)
            )
        self.net_b = nn.Sequential(
            nn.AvgPool2d(2),
            conv_base(in_channels, 64, 2),
            conv_base(64, 128, 2),
            conv_base(128, 256, 2),
            conv_base(256, 512, 1),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=2)
            )

    def forward(self, x, m):
        x = torch.cat((x,m), 1)
        a = [x]
        for layer in self.net_a.children():
            a.append(layer(a[-1]))
        b = [x]
        for layer in self.net_b.children():
            b.append(layer(b[-1]))
        return a[1:], b[1:]
