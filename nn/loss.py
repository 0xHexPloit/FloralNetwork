from torch import nn
import nn.config as config
import torch

# Defining losses that we are going to use to make both our descriminator and generator better
l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()


def loss_discriminator(predictions_fake, predictions_true):
    target_fake = torch.zeros_like(predictions_fake)
    loss_fake = mse_loss(predictions_fake, target_fake)

    target_true = torch.zeros_like(predictions_true)
    loss_true = mse_loss(predictions_true, target_true)

    return 0.5 * (loss_true + loss_fake)


def loss_generator(predictions, generated_img, true_img):
    target = torch.ones_like(predictions)

    return mse_loss(predictions, target) + config.LAMBDA_PARAM * l1_loss(generated_img, true_img)
