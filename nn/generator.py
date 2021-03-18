from collections import OrderedDict
from torch import nn
import torch


class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        super().__init__()

        features = init_features

        # Pooling
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # First conv block
        self.encoder1 = Unet._block(in_channels, features, name="enc1")

        # Second conv block
        self.encoder2 = Unet._block(features, features * 2, name="enc2")

        # Third conv block
        self.encoder3 = Unet._block(features * 2, features * 4, name="enc3")

        # Forth conv block
        self.encoder4 = Unet._block(features * 4, features * 8, name="enc4")

        # Bottleneck
        self.bottleneck = Unet._block(features * 8, features * 16, name="bottleneck")

        # First deconv block
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=(2, 2), stride=(2, 2))
        self.decoder4 = Unet._block((features * 8) * 2, features * 8, name="dec4")

        # Second deconv block
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=(2, 2), stride=(2, 2))
        self.decoder3 = Unet._block((features * 4) * 2, features * 4, name="dec3")

        # Third deconv block
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=(2, 2), stride=(2, 2))
        self.decoder2 = Unet._block((features * 2) * 2, features * 2, name="dec2")

        # Forth deconv block
        self.upconv1 = nn.ConvTranspose2d((features * 2) * 2, features * 2, kernel_size=(2, 2), stride=(2, 2))
        self.decoder1 = Unet._block(features * 2, features, name="dec1")

        # Final conv
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=(1, 1))
        self.final = nn.Sequential(nn.Tanh())

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.max_pooling(enc1))
        enc3 = self.encoder3(self.max_pooling(enc2))
        enc4 = self.encoder4(self.max_pooling(enc3))

        bottleneck = self.bottleneck(self.max_pooling(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat((dec4, enc4), dim=1))

        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))

        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))

        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))

        return self.final(self.final_conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(OrderedDict([
            (f"{name}-conv1", nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=False
            )),
            (f"{name}-bn1", nn.BatchNorm2d(
                num_features=features
            )),
            (f"{name}-relu1", nn.ReLU(
                inplace=True
            )),
            (f"{name}-conv2", nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=False
            )),
            (f"{name}-norm2", nn.BatchNorm2d(
                num_features=features
            )),
            (f"{name}-relu2", nn.ReLU(
                inplace=True
            ))
        ]))


# Syntactic sugar
class Generator(Unet):
    def __init__(self):
        super().__init__()
