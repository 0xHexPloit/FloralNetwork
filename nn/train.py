from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import Compose
from nn.dataset import FloralDataset
from nn.transforms import Rescale, Normalize, ToTensor
from nn.generator import Generator
from nn.discriminator import Discriminator
from nn.loss import loss_generator, loss_discriminator
from console import Console
import torch
import nn.config as config
import numpy as np

################################################
# TRANSFORMS
################################################
composed = Compose([
    Rescale((config.IMG_INPUT_SIZE, config.IMG_INPUT_SIZE)),
    Normalize(),
    ToTensor()
])

################################################
# DATASET
################################################
Console.print_info("Loading train and validation datasets")

train_floral_dataset = FloralDataset(
    root_dir="../data/train",
    transform=composed
)

val_floral_dataset = FloralDataset(
    root_dir="../data/val",
    transform=composed
)

################################################
# DATA LOADER
################################################
train_dataloader = DataLoader(
    train_floral_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
val_dataloader = DataLoader(
    val_floral_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

################################################
# TRAINING
################################################

# Defining number batches
number_batches = len(train_floral_dataset) // config.BATCH_SIZE

# Calculate output size of the image discriminator (PatchGan)
patch = (1, config.IMG_INPUT_SIZE // 2 ** 4, config.IMG_INPUT_SIZE // 2 ** 4)

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Defining optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(),
    lr=config.LEARNING_RATE,
    betas=(config.BETA_1, config.BETA_2)
)

optimizer_D = torch.optim.Adam(
    discriminator.parameters(),
    lr=config.LEARNING_RATE,
    betas=(config.BETA_1, config.BETA_2)
)



# we use GPU if available, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Console.print_info("Training is about to start...")

for epoch in range(config.NUM_EPOCHS):
    Console.print_epoch(epoch + 1, config.NUM_EPOCHS)

    for batch_idx, batch in enumerate(train_dataloader):
        # Model inputs
        sketches = Variable(batch["sketch"]).to(device)
        flowers = Variable(batch["flower"]).to(device)

        # Adversarial ground truth
        valid = Variable(torch.FloatTensor(np.ones((sketches.size(0), *patch)))).to(device)
        fake = Variable(torch.FloatTensor(np.zeros((sketches.size(0), *patch)))).to(device)

        # ------------------
        #  Train Generators
        # ------------------

        # Resetting gradient to zero for generator
        optimizer_G.zero_grad()

        # Computing generator loss
        fake_flowers = generator(sketches)
        predictions_fake = discriminator(sketches, fake_flowers)

        # Computing the loss
        loss_G = loss_generator(predictions_fake, fake_flowers, flowers)

        # Computing gradient
        loss_G.backward()

        # Updating parameters
        optimizer_G.step()

        # ------------------
        #  Train Discriminator
        # ------------------

        # Resetting gradient to zero for generator
        optimizer_D.zero_grad()

        # Computing predictions for real and fake images
        predictions_real = discriminator(sketches, flowers)
        predictions_fake = discriminator(sketches, fake_flowers.detach())

        # Computing loss
        loss_D = loss_discriminator(predictions_fake, predictions_real)

        # Computing gradient
        loss_D.backward()

        # Updating parameters
        optimizer_D.step()

        Console.print_info(f" Both model has been trained on batch [{batch_idx+1}/{number_batches}]")

    # Printing test results
    Console.print_test_results(
        epoch + 1,
        config.NUM_EPOCHS,
        generator_loss=loss_G.item(),
        discriminator_loss=loss_D.item()
    )

    # Visualising data generated
    if epoch > 0 and epoch == config.IMAGE_DISPLAY_VERBOSE:
        for batch_idx, batch in enumerate(val_dataloader):
            # Model inputs
            sketches = Variable(batch["sketch"]).to(device)
            flowers = Variable(batch["flower"]).to(device)

            fake_flowers = generator(sketches)

            flowers = flowers.detach().numpy()
            fake_flowers = fake_flowers.detach().numpy()

            true_flower = flowers[0]
            true_flower = true_flower.transpose((1, 2, 0))

            fake_flower = fake_flowers[0]
            fake_flower = fake_flower.transpose((1, 2, 0))

            if np.min(fake_flower) >= 0:
                Console.display_gan_image(fake_flower, true_flower)
            else:
                Console.print_info("Cannot visualize the picture")
