from argparse import ArgumentParser
from utils import storage, path, image
import generator

arg_parser = ArgumentParser()

# Adding arguments to our parser
arg_parser.add_argument("-i", "--image", help="path to the image to be regenerated", type=str, required=True)
arg_parser.add_argument("-o", "--output", help="path of the directory to save the generated image", type=str)

if __name__ == "__main__":
    args = arg_parser.parse_args()

    # Determining output path
    file_name, directory_path = path.get_output_path_data(args.image, args.output)

    # Generate image
    image = image.get_image_from_path(args.image)
    generated_image = generator.generate_image(image)

    # Saving generated image
    storage.save_image_to_disk(generated_image, file_name, directory_path)
