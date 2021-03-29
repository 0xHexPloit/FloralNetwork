import cv2
import numpy as np


def get_image_from_path(path=str) -> cv2.cv2:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def rescale_image(image: np.array) -> np.array:
    output = (image + 1) / 2
    output = output * 255
    output = output.astype(np.uint8)

    return output

