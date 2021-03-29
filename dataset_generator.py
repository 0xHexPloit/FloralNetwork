#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 10:18:19 2021

@author: eva

flower dataset : https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1ECTVN
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
"""

### Imports ###


from preprocessing import get_sketched_image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image

import cv2
import torchvision.transforms as transforms
import torch
import numpy as np
import glob
import random
from math import ceil
import os
import tarfile
import wget
import time 
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

### Utils ###

start = time.time()
# Ensure reproductibility in all places
def seed_all(seed):
    print(">>> Using Seed : ", seed, " <<<")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return None


seed_all(2905)

# check pytorch version : 1.7.1
print("Using PyTorch version ", torch.__version__)  # 1.6.0


def showImage(path, opt=1):
    img = cv2.imread(path, opt)  # BGR colors
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # to RGB
    plt.imshow(rgb, interpolation='nearest')
    plt.show()


### Wget the original dataset ###

# create data folder
os.makedirs("./data", exist_ok=True)

# download tgz file
print("Downloading the 102 flowers dataset... This will take a couple of minutes!")
url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
filename = wget.download(url, out="./data")  # out=dir_path
print("Sucessfully downloaded the 102 flowers dataset.")
print("Filename : ", filename)
print("Time elapsed : ", (time.time()-start))

# unzip file, it is a gzipped tar file
file = tarfile.open(filename)  # open file
file.extractall('./data/flowers102')  # extracting file
file.close()
# remove the tgz file, no longer needed
os.remove(filename)

# test : plot an image    
test_img_path = './data/flowers102/jpg/image_02501.jpg'
showImage(test_img_path)
img1 = cv2.imread(test_img_path)
print(img1.size)
print(img1.shape)

### Params ###

directory = "./data/flowers102/jpg"  # "./sample_flowers" #"./data/flowers102/jpg"
# "./sample_flowers" # to be replaced by ./data/flowers102/jpg full directory
batch_size = 4  # to increase : 8  images


### Dataset class ###

class MyDataset(Dataset):
    """
    image_list : list of image paths as strings
    trf : pytorch transforms sequence
    sketcher : sketching function taking a color image as input, opened with cv2
    """

    def __init__(self, image_list, trf=None, sketcher=None):
        self.image_list = image_list
        self.transforms = trf
        self.sketcher = sketcher

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        img = plt.imread(self.image_list[i])
        img = Image.fromarray(img).convert('RGB')
        img = np.array(img).astype(np.uint8)
        if self.transforms is not None:
            img = self.transforms(img)
        return torch.tensor(img, dtype=torch.float).clone().detach()  # sourceTensor.clone().detach()


### Train / Val / Test ###

image_list = glob.glob(directory + '/*.jpg')
N_tot = len(image_list)
print("%s images in directory %s." % (N_tot, directory))

# shuffle list : seed has been set above
print(image_list[:3])
random.shuffle(image_list)
print(image_list[:3])

# compute number of images
p1 = 0.9  # train_val percentage from initial dataset
p2 = 0.05  # val_percentage from train_val : only for visual checking
N_test = ceil(N_tot * (1 - p1))
N_val = ceil(N_tot * p1 * p2)
N_train = N_tot - N_val - N_test
print("\nNumber of train images :", N_train)
print("Number of val images :", N_val)
print("Number of test images :", N_test)

# Build file lists according to the above proportions
train_files = []
val_files = []
test_files = []
test_files += image_list[:N_test]
val_files += image_list[N_test:N_test + N_val]
train_files += image_list[N_test + N_val:]

# sanity check
print("Train set: %s, \nVal set :   %s, \nTest set :  %s." % (len(train_files), len(val_files), len(test_files)))

# make & fill sub directories 
os.makedirs("./data/train", exist_ok=True)  # succeeds even if directory exists.
os.makedirs("./data/val", exist_ok=True)  # succeeds even if directory exists.
os.makedirs("./data/test", exist_ok=True)  # succeeds even if directory exists.
# create subfolders photo (flowers) and sketches
for folder in ["./data/train", "./data/val", "./data/test"]:
    os.makedirs(folder + "/flowers", exist_ok=True)
    os.makedirs(folder + "/sketches", exist_ok=True)


### Basic data augmentation ###

## Transforms

# basic transform

def resize(img_list, batch_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(500),
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ])
    return DataLoader(MyDataset(img_list, transform), batch_size=batch_size, shuffle=False)


def rotation(img_list, batch_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation((-20, 20)),
        transforms.CenterCrop(312),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return DataLoader(MyDataset(img_list, transform), batch_size=batch_size, shuffle=False)


# transforms.RandomResizedCrop((256, 256), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333))

def cropping(img_list, batch_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop((500, 500)),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return DataLoader(MyDataset(img_list, transform), batch_size=batch_size, shuffle=False)


def hflip(img_list, batch_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(1.0),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return DataLoader(MyDataset(img_list, transform), batch_size=batch_size, shuffle=False)


def vflip(img_list, batch_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(1.0),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return DataLoader(MyDataset(img_list, transform), batch_size=batch_size, shuffle=False)


def colors(img_list, batch_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((500, 500)),
        transforms.ColorJitter(brightness=0.1, contrast=0.5, saturation=0.2, hue=0.1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return DataLoader(MyDataset(img_list, transform), batch_size=batch_size, shuffle=False)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def gauss(img_list, batch_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        AddGaussianNoise(0.1, 0.08)
    ])
    return DataLoader(MyDataset(img_list, transform), batch_size=batch_size, shuffle=False)


## Fill all directories with resized images
loader_id = 0

for [files, to_folder] in [[test_files, "./data/test/flowers"], [val_files, "./data/val/flowers"],
                           [train_files, "./data/train/flowers"]]:
    k = 0
    loader = resize(files, batch_size)
    print("\nNumber of images to resize : ", len(files))
    data = iter(loader)
    for i in range(ceil(len(files) / batch_size)):
        images_batch = data.next()
        for j in range(images_batch.shape[0]):
            img_aug = images_batch[j]
            path = to_folder + '/img_' + str(loader_id) + str(k) + '.jpg'
            save_image(img_aug, path)
            k += 1
    print("Total number of resized images in directory %s : %s" % (to_folder, k))

## Fill train directory with augmented images
# output dir : train subfolder
# train_files : still the original, full dimension images
# only train images will be augmented, so that val and test images remain unknown

output_directory = "data/train/flowers"
loaders = [rotation(train_files, batch_size),
           cropping(train_files, batch_size),
           hflip(train_files, batch_size),
           vflip(train_files, batch_size),
           colors(train_files, batch_size),
           gauss(train_files, batch_size)]

loader_id = 1
for l in loaders:
    k = 0
    data = iter(l)
    for i in range(ceil(N_train / batch_size)):
        images_batch = data.next()
        for j in range(images_batch.shape[0]):
            img_aug = images_batch[j]
            path = output_directory + '/imgaug_' + str(loader_id) + str(k) + '.jpg'
            save_image(img_aug, path)
            k += 1
    print("Total number of processsed images with loader %s : %s" % (loader_id, k))
    loader_id += 1


# Delete the original images which are in .data/flower102/
# os.remove('./data/flowers102') PermissionError: [Errno 1] Operation not permitted: './data/flowers102'

### Sketcher ###
# for all the images stored in the "flowers" subfolders, create a sketch equivalent
# and store it under the same name in the corresponding "sketches" subfolder

class SketchesDataset(Dataset):
    """
    image_list : list of image paths as strings
    trf : pytorch transforms sequence
    sketcher : sketching function taking a color image as input, opened with cv2
    """

    def __init__(self, image_list, trf=None, sketcher=None):
        self.image_list = image_list
        self.transforms = trf
        self.sketcher = sketcher

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        img = plt.imread(self.image_list[i])
        img = Image.fromarray(img).convert('RGB')
        img = np.array(img).astype(np.uint8)

        if self.transforms is not None:
            img = self.transforms(img)
        if self.sketcher is not None:
            BGR_img = cv2.imread(self.image_list[i],1) # open as BGR image
            img = self.sketcher(BGR_img) # transform to sketch
        return (torch.tensor(img,dtype=torch.float).clone().detach(), self.image_list[i]) # sourceTensor.clone().detach()


train_files = glob.glob(("./data/train/flowers/" + '*.jpg'))
val_files = glob.glob(("./data/val/flowers/" + '*.jpg'))
test_files = glob.glob(("./data/test/flowers/" + '*.jpg'))

for [files, to_folder] in [[test_files, "./data/test/sketches"], [val_files, "./data/val/sketches"],
                           [train_files, "./data/train/sketches"]]:
    k = 0
    loader = DataLoader(SketchesDataset(files, sketcher=get_sketched_image), batch_size=batch_size, shuffle=False)
    print("\nNumber of images to convert to sketches : ", len(files))
    data = iter(loader)
    for i in range(ceil(len(files) / batch_size)):
        images_batch, filenames = data.next()
        for j in range(images_batch.shape[0]):
            img_aug = images_batch[j]
            filename = filenames[j].split('/')[-1]
            path = os.path.join(to_folder, filename)
            save_image(img_aug, path)
            k += 1
    print("Total number of sketch images in directory %s : %s" % (to_folder, k))

# Delete the original images which are in .data/flowers102/
for path in glob.glob(("./data/flowers102/"+'*.jpg')):
    os.remove(path)
os.rmdir('./data/flowers102/')


### Recap ###

print("\n\nRecap: ")
N = 0  # total files
for dirpath, dirnames, filenames in os.walk("./data"):
    N_c = len(filenames)
    N += N_c
    print("Number of files in %s : %s" % (dirpath, N_c))

print("Total number of files : %s" % N)

print("\n\nTotal time elapsed to build the dataset : ", (time.time()-start))

