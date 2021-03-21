from utils import path
from glob import glob
from nn.generator import Generator
import numpy as np
import cv2
import os
import torch

# Defining useful paths
sep = os.path.sep
generator_path = f"{path.get_project_path()}{sep}nn{sep}models{sep}"


def save_image_to_disk(image: np.array, file_name: str, output_path: str):
    """
    A function to save an image to disk
    :param image: The image to save
    :param file_name: the name that we should give to the image that we are going to save
    :param output_path: Where to save the image
    """
    # Redefining working directory
    default_wd = os.getcwd()
    os.chdir(output_path)

    # Saving image
    cv2.imwrite(file_name, image)

    # Redefining default working directory
    os.chdir(default_wd)


def save_generator(generator: torch.nn.Module, filename: str):
    """
    Save a generator model
    :param generator: the generator to save
    :param filename: the name that we should give to the generator that we are going to save
    """
    torch.save(generator.state_dict(), f"{generator_path}{filename}.pt")


def load_generator() -> torch.nn.Module:
    """
    A function to load the generator model. If several models were saved, we return the first one that we will found
    :return: A flower generator model
    """
    file_path = glob(f"{generator_path}generator*.pt")[0]

    generator = Generator()
    generator.load_state_dict(torch.load(file_path))

    # Setting the model for evaluation
    generator.eval()

    return generator


if __name__ == "__main__":
    load_generator()
