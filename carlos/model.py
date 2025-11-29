import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from torchvision.models import ResNet34_Weights


class ResNetEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:] = resnet.conv1.weight.mean(dim=1, keepdim=True)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x = (x - 0.449) / 0.226
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)
        del x1

        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x, x2, x3, x4, x5


def icnr(tensor, scale=2, init_func=init.kaiming_normal_):
    ni, nf, h, w = tensor.shape
    ni2 = int(ni / (scale ** 2))
    k = init_func(torch.zeros([ni2, nf, h, w]))
    k = k.repeat_interleave(scale ** 2, 0)
    with torch.no_grad():
        tensor.copy_(k)


class PixelShuffleICNR(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale ** 2), kernel_size=3, padding=1)
        icnr(self.conv.weight, scale=scale)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        self.pixel_shuffle = PixelShuffleICNR(64, 16, scale=2)
        self.final = nn.Conv2d(16, 2, kernel_size=3, padding=1)

    def forward(self, x5, x4, x3, x2, x1):
        d4 = self.dec4(x5, x4)
        d3 = self.dec3(d4, x3)
        del d4, x4, x3
        d2 = self.dec2(d3, x2)
        del d3, x2
        d1 = self.dec1(d2, x1)
        del d2, x1
        out = self.pixel_shuffle(d1)
        del d1
        out = self.final(out)
        return torch.tanh(out)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.decoder = Decoder()

    def forward(self, x):
        x, x2, x3, x4, x5 = self.encoder(x)
        return self.decoder(x5, x4, x3, x2, x)
