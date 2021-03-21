import numpy as np
import cv2
import os


def save_image_to_disk(image: np.array, file_name: str, output_path: str):
    # Redefining working directory
    default_wd = os.getcwd()
    os.chdir(output_path)

    # Saving image
    cv2.imwrite(file_name, image)

    # Redefining default working directory
    os.chdir(default_wd)
