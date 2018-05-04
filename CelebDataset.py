import cv2
from random import randint
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset

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

class CelebDataset(Dataset):

    def __init__(self, train=True, transform=None, datadir="../Datasets/CelebA_Align/"):
        self.datadir = datadir
        self.train = train
        self.transform = transforms.Compose(transform)
        total_s = 202598 # one less than there actually is but eveness
        train_range = int(0.8 * total_s) - 1 # 80% n of ims
        self.train_range = train_range
        self.nsamples = train_range if train else total_s - train_range
        
    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        imfmt = "%06d.jpg"
        start = 1 if self.train else self.train_range
        im_name = self.datadir + imfmt % (start+index)
        im = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)
        return self.transform(im)

if __name__ == "__main__":
    transform = [Rescale((232, 232)), RandomCrop((DATA_W, DATA_H))]
    dataset = CelebDataset(train=False, transform=transform)
    cv2.imshow("hey", dataset[0])
    cv2.waitKey(0)
    cv2.imshow("hey", dataset[1])
    cv2.waitKey(0)
