from utils.console import Console
from utils.storage import load_generator
from utils.image import rescale_image
from nn.config import hyper_params
import cv2
import numpy as np
import preprocessing
import torch
import matplotlib.pyplot as plt

def generate_image(image: cv2.cv2) -> np.array:
    Console.print_info("Generating a sketch of the image...")
    sketch = preprocessing.get_sketched_image(image)

    Console.print_info("Loading generator...")
    generator = load_generator()

    Console.print_info("Applying some transformations to the sketch...")
    img_size = hyper_params.IMG_INPUT_SIZE
    input = cv2.resize(sketch, (img_size, img_size))
    input = input.reshape((img_size, img_size, 1))
    input = input.transpose((2, 0, 1))
    tensor_image = torch.from_numpy(np.array([input])).float()

    Console.print_info("Colouring of the sketch in progress...")
    output = generator(tensor_image)
    output = output.detach().numpy().squeeze()
    output = output.transpose((1, 2, 0))
    output = rescale_image(output)

    plt.imshow(output)
    plt.show()

    Console.print_info("The generated image is ready !")

    return output
