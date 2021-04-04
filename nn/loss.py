from torch import nn
import nn.config as config
import torch

# Defining losses that we are going to use to make both our discriminator and generator better
l1_loss = nn.L1Loss()
bce_loss = nn.BCEWithLogitsLoss()


def loss_generator(sketches, gan_images, true_images, discriminator):
    sketches_gan_images = torch.cat([sketches, gan_images], dim=1)

    predictions = discriminator(sketches_gan_images)
    target = torch.ones_like(predictions)
    loss_fake = bce_loss(predictions, target)

    l1 = l1_loss(gan_images, true_images) * config.LAMBDA_PARAM
    return loss_fake + l1


def loss_discriminator(sketches, gan_images, true_images, discriminator):
    sketches_gan_images = torch.cat([sketches, gan_images.detach()], dim=1)
    sketches_true_images = torch.cat([sketches, true_images], dim=1)

    pred = discriminator(sketches_gan_images)
    target = torch.zeros_like(pred)
    loss_fake = bce_loss(pred, target)

    pred = discriminator(sketches_true_images)
    target = torch.ones_like(pred)
    loss_real = bce_loss(pred, target)

    return (loss_fake + loss_real) * 0.5
