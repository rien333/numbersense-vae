import cv2
from random import shuffle
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import SOSDataset

# was 256, this is after cropping. Used to be 227x227 with crop, but 224 (even) makes the math easier
DATA_W = SOSDataset.DATA_W
DATA_H = SOSDataset.DATA_H
DATA_C = SOSDataset.DATA_C

class SynDataset(Dataset):

    def __init__(self, train=True, transform=None, datadir="../Datasets/synthetic/"):
        self.datadir = datadir
        self.train = train
        self.transform = transforms.Compose(transform)

        files_txt = datadir+"files.txt"
        with open(files_txt, "r") as f:
            files = f.read().splitlines()
        
        # Is it okay to shuffle the train and test set everytime?
        shuffle(files)
        self.files = files
        self.nsamples = len(files)
        self.train_range = int(0.8 * self.nsamples) # 320
        print(self.train_range)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        start = 0 if self.train else self.train_range
        im_name = self.datadir + self.files[start+index]
        im = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)
        label = im_name[-5]
        return self.transform((im, label))

if __name__ == "__main__":
    transform = [SOSDataset.Rescale((232, 232)), SOSDataset.RandomCrop((DATA_W, DATA_H))]
    dataset = SynDataset(train=False, transform=transform)
    cv2.imshow("hey", dataset[0][0])
    cv2.waitKey(0)
    cv2.imshow("hey", dataset[1][0])
    cv2.waitKey(0)
