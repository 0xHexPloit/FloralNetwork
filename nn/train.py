from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from nn.dataset import FloralDataset
from nn.transforms.tensor import ToTensor
from nn.transforms.rescale import Rescale
from nn.config import NUM_EPOCHS, BATCH_SIZE

################################################
# TRANSFORMS
################################################
composed = Compose([
    Rescale((124, 124)),
    ToTensor()
])

################################################
# DATASET
################################################
print("[INFO] Loading train and validation datasets")

train_floral_dataset = FloralDataset(
    root_dir="../data/train",
    transform=composed
)

val_floral_dataset = FloralDataset(
    root_dir="../data/val",
    transform=composed
)

################################################
# DATA LOADER
################################################
train_dataloader = DataLoader(train_floral_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_floral_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

################################################
# TRAINING
################################################
for epoch in range(NUM_EPOCHS):
    # Training
    for batch_idx, batch in enumerate(train_dataloader):
        print("BATCH_IDX")
        print(batch_idx)

        print("TEST BATCH")
        print(batch["flower"].shape)

        break
    break
