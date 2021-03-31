from datetime import datetime
from utils.console import Console
from path import LOGS_PATH


class Logs:
    _path: str

    def __init__(self, file_name=None):
        file_name = f"{file_name}.txt" if file_name else f"training_logs_{datetime.now()}.txt"
        self._path = f"{LOGS_PATH}{file_name}"

    def initialize(self, hyperparams: dict):
        with open(self._path, "w") as file:
            file.writelines([
                "=====================\n",
                "  TRAINING LOGS\n"
                "======================\n"
            ])

            file.write("\n----- HYPER-PARAMETERS -----\n")
            for key, value in hyperparams.items():
                file.write(f"{key}: {value}\n")
            file.write("------------------------------\n")

    def write_epoch_data(self, epoch: int, epochs: int, generator_loss: float, discriminator_loss: float, print_to_console=False):
        with open(self._path, "a") as file:
            data_to_write = [
                f"\n--------- TRAINING PERFORMANCE [{epoch}/{epochs}] ---------\n",
                f"G_LOSS: {generator_loss}\n",
                f"D_LOSS: {discriminator_loss}\n",
                f"---------------------------------------------------\n"
            ]

            file.writelines(data_to_write)

            if print_to_console:
                Console.print_lines(data_to_write)