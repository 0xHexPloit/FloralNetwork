from torch import nn
import torch


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.block1 = self._block(in_channels * 2, 64, normalization=False)
        self.block2 = self._block(64, 128)
        self.block3 = self._block(128, 256)
        self.block4 = self._block(256, 512)

        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.final_conv = nn.Conv2d(512, 1, kernel_size=(4, 4), padding=(1, 1), bias=False)

    def forward(self, sketch_image, colorful_image):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((sketch_image, colorful_image), 1)
        bc1 = self.block1(img_input)
        bc2 = self.block2(bc1)
        bc3 = self.block3(bc2)
        bc4 = self.block4(bc3)
        out = self.zero_pad(bc4)
        return self.final_conv(out)

    @staticmethod
    def _block(in_filters, out_filters, normalization=True):
        layers = [nn.Conv2d(in_filters, out_filters, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))]

        if normalization:
            layers.append(nn.BatchNorm2d(out_filters))

        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)
