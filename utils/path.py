from typing import Tuple
import os


def get_project_path() -> str:
    path = f"{os.path.sep}".join(os.path.abspath(__file__).split(os.path.sep)[:-2])
    return path


def get_directory_path(path=str) -> str:
    return f"{os.path.sep}".join(path.split(os.path.sep)[:-1])


def get_filename(path=str) -> str:
    return path.split(os.path.sep)[-1].split('.')[0]


def get_extension(path=str) -> str:
    return path.split(os.path.sep)[-1].split('.')[1]


def get_output_path_data(image_path=str, output_path=None) -> Tuple[str, str]:
    sep = os.path.sep
    file_name_with_extension = f"{get_filename(image_path)}_gen.{get_extension(image_path)}"
    directory_path = get_directory_path(output_path) if output_path else get_directory_path(image_path)

    return file_name_with_extension, f"{directory_path}{sep}"
