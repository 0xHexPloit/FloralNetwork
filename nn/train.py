from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import Compose
from nn.dataset import FloralDataset
from nn.transforms import Rescale, Normalize, ToTensor
from nn.generator import Generator
from nn.discriminator import Discriminator
from nn.loss import loss_generator, loss_discriminator
from utils.console import Console
from utils.logs import Logs
from utils import storage, image
import torch
import nn.config as config
import numpy as np
import random

# Getting hyper_params
hp = config.hyper_params

################################################
# TRANSFORMS
################################################
composed = Compose([
    Rescale((hp.IMG_INPUT_SIZE, hp.IMG_INPUT_SIZE)),
    Normalize(),
    ToTensor()
])

################################################
# DATASET
################################################
Console.print_info("Loading train and validation datasets")

train_floral_dataset = FloralDataset(
    folder_name="train",
    transform=composed
)

val_floral_dataset = FloralDataset(
    folder_name="val",
    transform=composed
)

################################################
# DATA LOADER
################################################
train_dataloader = DataLoader(
    train_floral_dataset,
    batch_size=hp.BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
val_dataloader = DataLoader(
    val_floral_dataset,
    batch_size=hp.BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

################################################
# Logs
################################################
logs = Logs()
logs.initialize(hyperparams=hp)

################################################
# TRAINING
################################################

# Defining number batches
number_batches = (len(train_floral_dataset) // hp.BATCH_SIZE) + 1

# Calculate output size of the image discriminator (PatchGan)
patch = (1, hp.IMG_INPUT_SIZE // 2 ** 4, hp.IMG_INPUT_SIZE // 2 ** 4)

# we use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Console.print_info("Training is about to start...")

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Defining optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(),
    lr=hp.LEARNING_RATE,
    betas=(hp.BETA_1, hp.BETA_2)
)

optimizer_D = torch.optim.Adam(
    discriminator.parameters(),
    lr=hp.LEARNING_RATE,
    betas=(hp.BETA_1, hp.BETA_2)
)

# Keeping trace of losses
loss_G = None
previous_loss = np.inf

for epoch in range(1, hp.NUM_EPOCHS + 1):
    # Setting previous loss for generator
    if loss_G is not None:
        previous_loss = loss_G

    Console.print_epoch(epoch, hp.NUM_EPOCHS)

    # Setting generator into training mode
    generator.train()

    for batch_idx, batch in enumerate(train_dataloader):
        # Model inputs
        sketches = Variable(batch["sketch"]).to(device)
        flowers = Variable(batch["flower"]).to(device)

        # Generating fake images
        fake_flowers = generator(sketches)

        # ------------------------
        #  Updating discriminator
        # ------------------------
        discriminator.requires_grad(True)
        optimizer_D.zero_grad()

        # Computing the loss
        loss_D = loss_discriminator(sketches, fake_flowers, flowers, discriminator)

        # Computing gradient
        loss_D.backward()

        # Updating parameters
        optimizer_D.step()

        # ---------------------
        #  Updating generator
        # ---------------------

        # Resetting gradient to zero for generator and preventing discriminator to be updated
        discriminator.requires_grad(False)
        optimizer_G.zero_grad()

        # Computing loss
        loss_G = loss_generator(sketches, fake_flowers, flowers, discriminator)

        # Computing gradient
        loss_G.backward()

        # Updating parameters
        optimizer_G.step()

        Console.print_info(f" Both model has been trained on batch [{batch_idx+1}/{number_batches}]")

    # Saving training results
    logs.write_epoch_data(
        epoch,
        hp.NUM_EPOCHS,
        loss_G.item(),
        loss_D.item(),
        print_to_console=True
    )

    # Saving generator if performance increased during training
    if config.save_generator and loss_G < previous_loss:
        storage.save_generator(generator, config.generator_filename)

    # Visualising data generated
    if epoch % config.IMAGE_DISPLAY_VERBOSE == 0:
        generator.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                # Model inputs
                sketches = Variable(batch["sketch"]).to(device)
                flowers = Variable(batch["flower"]).to(device)

                fake_flowers = generator(sketches)

                flowers = flowers.detach().cpu().numpy()
                fake_flowers = fake_flowers.detach().cpu().numpy()

                image_idx = random.choice(range(len(flowers)))

                true_flower = flowers[image_idx]
                true_flower = true_flower.transpose((1, 2, 0))

                fake_flower = fake_flowers[image_idx]
                fake_flower = fake_flower.transpose((1, 2, 0))

                # Enregistrement de l'image
                storage.save_images_training(epoch, true_flower, image.rescale_image(fake_flower))

                Console.display_gan_image(fake_flower, true_flower)

                break