from matplotlib import pyplot as plt
import numpy as np

class Console:
    @staticmethod
    def print_epoch(epoch, total_number_epochs):
        print("##############################################")
        print(f"# Starting epoch {epoch}/{total_number_epochs}")
        print("##############################################")

    @staticmethod
    def print_info(content):
        print(f"[INFO] {content}")


    @staticmethod
    def print_test_results(epoch, total_number_epochs, discriminator_loss, generator_loss):
        print("##############################################")
        print(f"# Evaluation of the models {epoch}/{total_number_epochs}")
        print("##############################################")

        Console.print_info(f"Generator's loss: {generator_loss}")
        Console.print_info(f"Discriminator's loss: {discriminator_loss}")

    @staticmethod
    def display_gan_image(fake_image, true_image):
        print("########################################")
        print(f"# Visualization of a generated image")
        print("########################################")

        plt.imshow(np.hstack([fake_image, true_image]))
