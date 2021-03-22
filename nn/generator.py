from torch import nn
import torch


class UnetDown(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, features, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=features)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.batch_norm(out)
        out = self.activation(out)

        return out


class UnetUp(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            features,
            kernel_size=2,
            stride=2,
            bias=False
        )
        self.conv1 = nn.Conv2d(features * 2, features, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=features)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        out = self.conv_transpose(x1)
        out = self.conv1(torch.cat([out, x2], dim=1))
        out = self.batch_norm(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.batch_norm(out)
        out = self.activation(out)

        return out


class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        super().__init__()

        features = init_features

        # Pooling
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # First conv block
        self.down1 = UnetDown(in_channels, features)

        # Second conv block
        self.down2 = UnetDown(features, features * 2)

        # Third conv block
        self.down3 = UnetDown(features * 2, features * 4)

        # Forth conv block
        self.down4 = UnetDown(features * 4, features * 8)

        # Bottleneck
        self.bottleneck = UnetDown(features * 8, features * 16)

        # First deconv block
        self.up4 = UnetUp(features * 16, features * 8)

        # Second deconv block
        self.up3 = UnetUp(features * 8, features * 4)

        # Third deconv block
        self.up2 = UnetUp(features * 4, features * 2)

        # Forth deconv block
        self.up1 = UnetUp(features * 2, features)

        # Final conv
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=(1, 1))
        self.final = nn.Sequential(nn.Tanh())

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(self.max_pooling(down1))
        down3 = self.down3(self.max_pooling(down2))
        down4 = self.down4(self.max_pooling(down3))

        bottleneck = self.bottleneck(self.max_pooling(down4))

        up4 = self.up4(bottleneck, down4)
        up3 = self.up3(up4, down3)
        up2 = self.up2(up3, down2)
        up1 = self.up1(up2, down1)

        out = self.final_conv(up1)
        out = self.final(out)

        return out


# Syntactic sugar
class Generator(Unet):
    def __init__(self):
        super().__init__()
