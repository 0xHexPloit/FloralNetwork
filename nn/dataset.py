from torch.utils.data import Dataset
from glob import glob
import torch
import cv2


class FloralDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = glob(f"{root_dir}/sketches/*")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        file = self.files[idx].split("/")[-1]
        flower_path = f"{self.root_dir}/flowers/{file}"
        sketch_path = f"{self.root_dir}/sketches/{file}"

        flower_img = cv2.imread(flower_path)
        flower_img = cv2.cvtColor(flower_img, cv2.COLOR_BGR2RGB)

        sketch_img = cv2.imread(sketch_path)
        sketch_img = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2GRAY)

        sample = {"flower": flower_img, "sketch": sketch_img}

        if self.transform:
            sample = self.transform(sample)

        return sample
