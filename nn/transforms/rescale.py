import cv2

class Rescale(object):
    """
    Rescale the image in a sample to a given size.
   """
    def __init__(self, output_size):
        """

        :param output_size (tuple or int): Desired output size. If tuple, output is
               matched to output_size. If int, smaller of image edges is matched
               to output_size keeping aspect ratio the same.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        sketch, flower = sample['sketch'], sample['flower']

        h, w = sketch.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        sketch = cv2.resize(sketch, (new_h, new_w))
        flower = cv2.resize(flower, (new_h, new_w))

        return {
            "sketch": sketch,
            "flower": flower
        }
