import cv2
from random import randint
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import mnist

# was 256, this is after cropping. Used to be 227x227 with crop, but 224 (even) makes the math easier
DATA_W = 225
DATA_H = 225
DATA_C = 3

class Rescale(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, s):
        # default interpolation=cv2.INTER_LINEAR (rec., fast and ok quality)
        return cv2.resize(s, self.output_size)

class RandomCrop(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, s):
        h, w = s.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        crop_im = s[top : top + new_h, left : left + new_w]
        return crop_im

class RandHorizontalFlip(object):

    def __call__(self, s):
        if randint(0,1):
            return np.flip(s, 1).copy()
        else:
            return s

class ToTensor(object):

    def  __call__(self, s):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # You can do the reshape((W,H,C)) to get the original (numpy format) back
        return torch.from_numpy(s.transpose((2,0,1))).float()

class Normalize(object):

    def __call__(self, s):
        return s / 255

class Normalize01(object):

    """Normalize between 0-1"""

    def __call__(self, s):
        return (s + 1)/2

class NormalizeMean(object):

    def __call__(self, s):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        return normalize(s)

class ColoredMNIST(Dataset):
    
    def __init__(self, train=True, transform=None,):
        self.transform = transforms.Compose(transform)
        self.train = train
        lena = cv2.imread('../Datasets/lena.jpg')
        if train:
            data = mnist.train_images()
        else:
            data = mnist.test_images()
        self.nsamples = len(data)
        # resize (so that lena fits better)
        data = np.asarray([cv2.resize(im, (64,64)) for im in data])
        data = data.reshape(self.nsamples, 64, 64, 1)
        # extend to rgb
        data = np.concatenate([data, data, data], axis=3)
        # convert to binary (so that lena overlay can be used)
        data = (data > 0.5)

        rgb_data = np.zeros((self.nsamples, 64, 64, 3))
        crop = RandomCrop((64, 64))
        for i in range(self.nsamples):
            # Take a random crop of the Lena image (background)
            x_c = np.random.randint(0, lena.shape[0] - 64)
            y_c = np.random.randint(0, lena.shape[1] - 64)
            # lena_c = lena.crop((x_c, y_c, x_c + 64, y_c + 64))
            lena_c = crop(lena)
            # Conver the image to float between 0 and 1
            lena_c = np.asarray(lena_c) / 255.0

            # Change color distribution
            for j in range(3):
                lena_c[:, :, j] = (lena_c[:, :, j] + np.random.uniform(0, 1)) / 2.0

            # Invert the colors at the location of the number
            lena_c[data[i]] = 1 - lena_c[data[i]]
            rgb_data[i] = lena_c

        self.data = rgb_data
     
    def __getitem__(self, index):
        return self.transform(self.data[index])

    def __len__(self):
        return self.nsamples
        
if __name__ == "__main__":
    # load preprocess
    transform = [Rescale((232, 232)), RandomCrop((DATA_W, DATA_H))]
    # transform = [Rescale((256, 256)), 
    #               ToTensor(), Normalize()]
    dataset = ColoredMNIST(train=False, transform=transform)
