from argparse import ArgumentParser
import utils
import storage
import numpy as np

arg_parser = ArgumentParser()

# Adding arguments to our parser
arg_parser.add_argument("-i", "--image", help="path to the image to be regenerated", type=str, required=True)
arg_parser.add_argument("-o", "--output", help="path of the directory to save the generated image", type=str)

if __name__ == "__main__":
    args = arg_parser.parse_args()

    # Determining output path
    file_name, directory_path = utils.get_output_path_data(args.image, args.output)

    # Fake image
    fake_image = np.zeros((50, 50, 3))

    # Saving generated image
    storage.save_image_to_disk(fake_image, file_name, directory_path)
