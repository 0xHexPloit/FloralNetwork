import torch


class ToTensor(object):
    def __call__(self, sample):
        sketch, flower = sample["sketch"], sample["flower"]

        # Swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if len(sketch.shape) != 3:
            shape = sketch.shape
            sketch = sketch.reshape((shape[0], shape[1], 1))

        sketch = sketch.transpose((2, 0, 1))
        flower = flower.transpose((2, 0, 1))

        return {
            "sketch": torch.from_numpy(sketch),
            "flower": torch.from_numpy(flower)
        }
