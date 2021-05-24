import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, stride=1, ksize=3):
        w = math.sqrt(2)
        super().__init__()

        self.c1 = nn.Conv2d(n_in, n_out, ksize, stride=stride, padding=1)
        nn.init.constant_(self.c1.weight, w)
        self.c2 = nn.Conv2d(n_out, n_out, ksize, stride=1, padding=1)
        nn.init.constant_(self.c2.weight, w)
        self.b1 = nn.BatchNorm2d(n_out)
        self.b2 = nn.BatchNorm2d(n_out)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        """
        if x.data.shape != h.data.shape:
            xp = chainer.cuda.get_array_module(x.data)
            n, c, hh, ww = x.data.shape
            pad_c = h.data.shape[1] - c
            p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
            p = chainer.Variable(p, volatile=test)
            x = torch.cat((p, x))
            if x.data.shape[2:] != h.data.shape[2:]:
                x = F.average_pooling_2d(x, 1, 2)
        """
        return h + x


class CNNAE2ResNet(nn.Module):
    def __init__(self, train=True):
        super(CNNAE2ResNet, self).__init__()
        self.c0 = nn.Conv2d(3, 64, 4, stride=2, padding=1)  # 1024 -> 512
        nn.init.normal_(self.c0.weight, 0.0, 0.02)
        self.c1 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # 512 -> 256
        nn.init.normal_(self.c1.weight, 0.0, 0.02)
        self.c2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)  # 256 -> 128
        nn.init.normal_(self.c2.weight, 0.0, 0.02)
        self.c3 = nn.Conv2d(256, 512, 4, stride=2, padding=1)  # 128 -> 64
        nn.init.normal_(self.c3.weight, 0.0, 0.02)
        self.c4 = nn.Conv2d(512, 512, 4, stride=2, padding=1)  # 64 -> 32
        nn.init.normal_(self.c4.weight, 0.0, 0.02)
        self.c5 = nn.Conv2d(512, 512, 4, stride=2, padding=1)  # 32 -> 16
        nn.init.normal_(self.c5.weight, 0.0, 0.02)

        self.ra = ResidualBlock(512, 512)
        self.rb = ResidualBlock(512, 512)

        self.dc0a = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        nn.init.normal_(self.dc0a.weight, 0.0, 0.02)
        self.dc1a = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        nn.init.normal_(self.dc1a.weight, 0.0, 0.02)
        self.dc2a = nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1)
        nn.init.normal_(self.dc2a.weight, 0.0, 0.02)
        self.dc3a = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1)
        nn.init.normal_(self.dc3a.weight, 0.0, 0.02)
        self.dc4a = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        nn.init.normal_(self.dc4a.weight, 0.0, 0.02)
        self.dc5a = nn.ConvTranspose2d(128, 9, 4, stride=2, padding=1)
        nn.init.normal_(self.dc5a.weight, 0.0, 0.02)

        self.dc0b = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        nn.init.normal_(self.dc0b.weight, 0.0, 0.02)
        self.dc1b = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        nn.init.normal_(self.dc1b.weight, 0.0, 0.02)
        self.dc2b = nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1)
        nn.init.normal_(self.dc2b.weight, 0.0, 0.02)
        self.dc3b = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1)
        nn.init.normal_(self.dc3b.weight, 0.0, 0.02)
        self.dc4b = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        nn.init.normal_(self.dc4b.weight, 0.0, 0.02)
        self.dc5b = nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)
        nn.init.normal_(self.dc5b.weight, 0.0, 0.02)

        self.c0l = nn.Conv2d(512 * 3, 512, 4, stride=2, padding=1)  # 16 -> 8
        nn.init.normal_(self.c0l.weight, 0.0, 0.02)
        self.c1l = nn.Conv2d(512, 256, 4, stride=2, padding=1)  # 8 -> 4
        nn.init.normal_(self.c1l.weight, 0.0, 0.02)
        self.c2l = nn.Conv2d(256, 128, 4, stride=2, padding=1)  # 4 -> 2
        nn.init.normal_(self.c2l.weight, 0.0, 0.02)
        self.c3l = nn.Conv2d(128, 27, 4, stride=2, padding=1)  # 2 -> 1
        nn.init.normal_(self.c3l.weight, 0.0, 0.02)

        self.bnc1 = nn.BatchNorm2d(128)
        self.bnc2 = nn.BatchNorm2d(256)
        self.bnc3 = nn.BatchNorm2d(512)
        self.bnc4 = nn.BatchNorm2d(512)
        self.bnc5 = nn.BatchNorm2d(512)

        self.bndc0a = nn.BatchNorm2d(512)
        self.bndc1a = nn.BatchNorm2d(512)
        self.bndc2a = nn.BatchNorm2d(256)
        self.bndc3a = nn.BatchNorm2d(128)
        self.bndc4a = nn.BatchNorm2d(64)

        self.bndc0b = nn.BatchNorm2d(512)
        self.bndc1b = nn.BatchNorm2d(512)
        self.bndc2b = nn.BatchNorm2d(256)
        self.bndc3b = nn.BatchNorm2d(128)
        self.bndc4b = nn.BatchNorm2d(64)

        self.bnc0l = nn.BatchNorm2d(512)
        self.bnc1l = nn.BatchNorm2d(256)
        self.bnc2l = nn.BatchNorm2d(128)

        self.train_dropout = train

    def forward(self, xi):
        hc0 = F.leaky_relu(self.c0(xi))
        hc1 = F.leaky_relu(self.bnc1(self.c1(hc0)))
        hc2 = F.leaky_relu(self.bnc2(self.c2(hc1)))
        hc3 = F.leaky_relu(self.bnc3(self.c3(hc2)))
        hc4 = F.leaky_relu(self.bnc4(self.c4(hc3)))
        hc5 = F.leaky_relu(self.bnc5(self.c5(hc4)))

        hra = self.ra(hc5)

        ha = F.relu(
            F.dropout(self.bndc0a(self.dc0a(hra)), 0.5, training=self.train_dropout)
        )
        ha = torch.cat((ha, hc4), 1)
        ha = F.relu(
            F.dropout(self.bndc1a(self.dc1a(ha)), 0.5, training=self.train_dropout)
        )
        ha = torch.cat((ha, hc3), 1)
        ha = F.relu(
            F.dropout(self.bndc2a(self.dc2a(ha)), 0.5, training=self.train_dropout)
        )
        ha = torch.cat((ha, hc2), 1)
        ha = F.relu(self.bndc3a(self.dc3a(ha)))
        ha = torch.cat((ha, hc1), 1)
        ha = F.relu(self.bndc4a(self.dc4a(ha)))
        ha = torch.cat((ha, hc0), 1)
        ha = self.dc5a(ha)

        hrb = self.rb(hc5)

        hb = F.relu(
            F.dropout(self.bndc0b(self.dc0b(hrb)), 0.5, training=self.train_dropout)
        )
        hb = torch.cat((hb, hc4), 1)
        hb = F.relu(
            F.dropout(self.bndc1b(self.dc1b(hb)), 0.5, training=self.train_dropout)
        )
        hb = torch.cat((hb, hc3), 1)
        hb = F.relu(
            F.dropout(self.bndc2b(self.dc2b(hb)), 0.5, training=self.train_dropout)
        )
        hb = torch.cat((hb, hc2), 1)
        hb = F.relu(self.bndc3b(self.dc3b(hb)))
        hb = torch.cat((hb, hc1), 1)
        hb = F.relu(self.bndc4b(self.dc4b(hb)))
        hb = torch.cat((hb, hc0), 1)
        hb = torch.clamp(self.dc5b(hb), 0.0, 1.0)

        hc = torch.cat((hc5, hra, hrb), 1)

        hc = F.leaky_relu(self.bnc0l(self.c0l(hc)))
        hc = F.leaky_relu(self.bnc1l(self.c1l(hc)))
        hc = F.leaky_relu(self.bnc2l(self.c2l(hc)))
        hc = torch.reshape(self.c3l(hc), (xi.size(0), 9, 3))

        return ha, hb, hc
