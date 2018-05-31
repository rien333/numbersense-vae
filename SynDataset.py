import cv2
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import SOSDataset
import pickle

# was 256, this is after cropping. Used to be 227x227 with crop, but 224 (even) makes the math easier
DATA_W = SOSDataset.DATA_W
DATA_H = SOSDataset.DATA_H
DATA_C = SOSDataset.DATA_C

class SynDataset(Dataset):

    def __init__(self, train=True, transform=None, datadir="../Datasets/", sorted_loc="/tmp/", n=None, split=0.8):
        self.datadir = datadir
        self.train = train
        self.transform = transforms.Compose(transform)
        self.datadir = datadir + "synthetic/"

        files_txt = self.datadir+"files.txt"
        with open(files_txt, "r") as f:
            files = f.read().splitlines()

        self.files = files[:n]
        nfiles = len(self.files)
        self.train_range = int(split * nfiles) # convert to idx
        self.nsamples = self.train_range if train else nfiles - self.train_range
        self.sorted_loc = sorted_loc + "sorted_classes_syn.pickle"
        if os.path.isfile(self.sorted_loc):
            with open (self.sorted_loc, 'rb') as f:
                sorted_classes = pickle.load(f)
            if sum([len(l) for l in sorted_classes]) != self.nsamples:
                print("Warning: pickle for syn data is outdated")

    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        start = 0 if self.train else self.train_range
        im_name = self.datadir + self.files[start+index]
        im = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)
        label = int(im_name[-5])
        return self.transform((im, label))

    def load_sorted_classes(self):
        # Sorting all the indices by class takes really long for some reason, so save and read from file
        if os.path.isfile(self.sorted_loc):
            with open (self.sorted_loc, 'rb') as f:
                c = pickle.load(f)
        else:
            c = self.sorted_classes()
            # save
            with open(self.sorted_loc, 'wb') as f:
                pickle.dump(c, f)
        return c

    def sorted_classes(self):
        """ Returns a list with all examples sorted by class """

        classes = [[]] * 5
        for i in range(self.nsamples):
            c = int(self[i][1])
            classes[c] = classes[c] + [i]
        return classes

if __name__ == "__main__":
    transform = [SOSDataset.Rescale((232, 232))]
    dataset = SynDataset(train=True, transform=transform, split=1.0)

    from collections import Counter
    classes = Counter()
    samples = len(dataset)
    for s in range(samples):
        try:
            classes[int(dataset[s][1])] += 1
        except:
            print(s)
            print(samples)
            exit(0)

    print("All", sorted(classes.items(), key=lambda pair: pair[0], reverse=False))

    # print(sorted(dataset.files, key=lambda k: k[-5])[:10])
    classes = dataset.load_sorted_classes()
    # for l in classes:
    #     print(len(l))
    # cv2.imwrite("test.jpg", cv2.cvtColor(dataset[dataset.sorted()[3][8]][0], cv2.COLOR_BGR2RGB))
