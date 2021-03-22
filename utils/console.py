from typing import List
from utils.image import rescale_image
from matplotlib import pyplot as plt
import numpy as np

class Console:
    @staticmethod
    def print_epoch(epoch, total_number_epochs):
        print("\n##############################################")
        print(f"# Starting epoch {epoch}/{total_number_epochs}")
        print("##############################################")

    @staticmethod
    def print_info(content):
        print(f"[INFO] {content}")

    @staticmethod
    def display_gan_image(fake_image, true_image):
        print("########################################")
        print(f"# Visualization of a generated image")
        print("########################################")

        plt.imshow(np.hstack([rescale_image(fake_image), true_image]))
        plt.show()

    @staticmethod
    def print_lines(lines: List):
        for line in lines:
            print(line.rstrip('\n'))
