import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import SOSDataset
import SynDataset
import cv2
from random import sample, shuffle
import itertools

class HybridEqualDataset(Dataset):

    # maybe have two different transforms for the datasets
    # Consider the dynamic between test and train (syn are totally random, and pseuod generated)
    # as syn examples are pseudo generated, consider importing way more


    def __init__(self, epochs, transform=None, grow_f=0.38, t=0.0, datadir="../Datasets/", sorted_loc="/tmp",
                 syn_samples=None, real_samples=None, train=True,):
        """
        grow_f is a factor [0,1] by how much we should grow the datasize with synthetic examples
        """

        self.train = train
        self.classes = 5
        self.sos = SOSDataset.SOSDataset(train=train, extended=True, transform=transform, datadir=datadir, sorted_loc=sorted_loc)
        # load the sorted list from a file for speed
        self.sos_sort = self.sos.load_sorted_classes()
        self.sos_n = [len(s) for s in self.sos_sort]
        r_samples = len(self.sos)
        s_samples = round(grow_f * r_samples)
        self.nsamples = r_samples + s_samples
        # make nsamples a multiple of the amount of classes
        # number of examples per class for perfect balance
        self.class_n = round(self.nsamples / self.classes)
        self.nsamples += self.classes - (self.nsamples % self.classes)

        # Specify absolute amounts
        if not syn_samples is None:
            t = 1.1
        self.syn_samples = syn_samples

        if not real_samples is None:
            t = 1.1
        self.real_samples = real_samples

        self.syn = SynDataset.SynDataset(train=True, transform=transform, split=1, datadir=datadir, sorted_loc=sorted_loc)
        # load the sorted list from a file for speed
        self.syn_sort = self.syn.load_sorted_classes()
        self.t_incr = 1/(epochs+1)
        self.t = t # should equal zero ofc
        self.u1 = -0.01 # bezier steepness in the beginning (flat 0 at start if negative)
        self.u2 = 0.1 # bezier steepness towards the end
        self.syn_ratio = self.__bezier(self.t, self.u1, self.u2)
        self.datasets = [self.sos, self.syn]
        from collections import Counter
        self.syn_counter = 0
        self.generate_samples()
        self.nsamples = len(self.samples)
        if self.nsamples % self.classes != 0:
            print("Number of samples", self.nsamples, "should be divisible by", self.classes)
            exit(0)
        if train:
            print("Training with %s hybrid samples" % (len(self.samples)))

    def __bezier(self, t, u1, u2):
        # instead of nsamples use len(self)? or update self.nsamples in the len function occiasinaly?
        # u0 = 0.0 # fixed
        # u3 = 1.0 # fixed
        # see bezier.py to graph stuff and some extra settings?
        return max(0, min(1, (3*u1*((1-t)**2))*t+3*u2*(1-t)*(t**2)+t**3))

    def generate_samples(self):
        n_real_samples = np.clip(np.round_(self.syn_ratio * np.array(self.sos_n)), 0, self.class_n)
        # You can take the absolute value of this array i think
        missing_real_samples = n_real_samples - self.class_n
        if self.real_samples is None:
            real_samples = [sample(self.sos_sort[idx], int(n)) if n > 0 else [] for idx, n in enumerate(n_real_samples)]
        else:
            real_samples = [sample(self.sos_sort[idx], n) for idx, n in enumerate(self.real_samples)]
        if self.syn_samples is None:
            if self.real_samples:
                missing_real_samples = self.real_samples - self.class_n
            syn_samples = [sample(self.syn_sort[idx], abs(int(n))) if n < 0 else [] for idx, n in enumerate(missing_real_samples)]
        else:
            syn_samples = [sample(self.syn_sort[idx], n) for idx, n in enumerate(self.syn_samples)]
        syn_samples = list(itertools.chain.from_iterable(syn_samples)) # flatten
        real_samples = list(itertools.chain.from_iterable(real_samples)) # flatten
        self.samples = [(0,s) for s in real_samples] # idxs refer to the two datasets
        self.samples += [(1,s) for s in syn_samples]
        shuffle(self.samples)

    # Maybe make a choise if you want a balanced sample from the real data as well
    def __getitem__(self, index):
        s = self.samples[index]
        d = self.datasets[s[0]]

        if index >= self.nsamples-1:
            self.t += self.t_incr
            self.syn_ratio = self.__bezier(self.t, self.u1, self.u2)
            self.generate_samples()
        if s[0] == 1:
            self.syn_counter += 1
        return d[s[1]]

    # We will probably update this dynamically, but keep it in sync with the batch size! (nice and divedable?)
    def __len__(self):
        return self.nsamples 

if __name__ == "__main__":

    from collections import Counter
    t = [SOSDataset.Rescale((232, 232)), SOSDataset.RandomCrop((SOSDataset.DATA_W, SOSDataset.DATA_H))]
    epochs=20
    # syn_samples = [4700, 5400, 8023, 8200, 8700]
    # real_samples = [1101, 1100, 1604, 1058, 853]
    
    hd = HybridEqualDataset(epochs=epochs, transform=t, train=True, t=0.775, grow_f=6.2952)

    samples = len(hd)
    for epoch in range(epochs+2):
        classes = Counter()
        for s in range(samples):
            try:
                classes[int(hd[s][1])] += 1
            except:
                print(s)
                print(samples)
                exit(0)

        print("All", sorted(classes.items(), key=lambda pair: pair[0], reverse=False))
        # print(hd.syn_counter)
        # print("All", sorted(hd.syn_counter.items(), key=lambda pair: pair[0], reverse=False))
        hd.syn_counter = 0
        print("------------------------------")
        # exit(0)
