import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import SOSDataset
import SynDataset
import cv2
import random

class HybridDataset(Dataset):

    # maybe have two different transforms for the datasets
    # Consider the dynamic between test and train (syn are totally random, and pseuod generated)
    # as syn examples are pseudo generated, consider importing way more
    # so yeah def use all files when this works!

    def __init__(self, epochs, transform=None, grow_f=0.38, train=True,):
        """
        grow_f is a factor [0,1] by how much we should grow the datasize with synthetic examples
        """

        self.train = train
        self.classes = 5
        # MAKE EXTENDED 🐝🐝🐝
        self.sos = SOSDataset.SOSDataset(train=train, extended=True, transform=transform)
        self.sos_sort = self.sos.sorted_classes()
        self.sos_n = [len(s) for s in self.sos_sort]
        self.r_samples = len(self.sos)
        s_samples = round(self.r_samples * grow_f)
        self.nsamples = self.r_samples + s_samples
        # number of examples per class for perfect balance
        self.class_n = self.nsamples / self.classes
        # generate only training examples
        self.syn = SynDataset.SynDataset(train=True, transform=transform, split=1)
        self.syn_sort = self.syn.sorted_classes()
        self.idx = 0
        self.ridx = 0
        # wait this dependent on samples right
        # although you might want to control these seperately
        self.t_incr = 1/(epochs+1)
        self.t = 0
        # curve bending, slow fade in
        self.u1 = -0.03 # bezier steepness in the beginning (flat 0 at start if negative)
        self.u2 = 0.03 # bezier steepness towards the end
        # idk, but the imbalance ratio might not work because #c1 > class_n (ratio of 1.6)
        imbalance_ratio = np.clip(np.array([n / self.class_n for n in self.sos_n]), 0, 0.96)
        sum_w = np.sum(imbalance_ratio)
        # Normalize
        imbalance_weights = (imbalance_ratio / sum_w)
        # class_w are ordered on most to least samples in respect to the size of their weights, so reverse
        self.imbalance_weights = imbalance_weights
        ws = (self.update_class_weights(1))**(10)

        self.sort_map = [list(ws).index(w) for w in sorted(ws)]
        print(self.sort_map)
        self.class_w_cum = np.cumsum(np.ones(5, dtype=np.float)) # weigh equally if only synthetic data
        # self.class_w_cum = np.cumsum(sorted(ws/np.sum(ws))) # weigh equally if only synthetic data
        # self.class_w_cum = np.cumsum(sorted(ws)) # weigh equally if only synthetic data
        self.syn_ratio = self.__bezier(self.t, self.u1, self.u2)
        # self.syn_ratio = 1         # self.t = 1
        from  collections import Counter
        self.syn_counter = Counter()
      
    def __bezier(self, t, u1, u2):
        # instead of nsamples use len(self)? or update self.nsamples in the len function occiasinaly?
        # u0 = 0.0 # fixed
        # u3 = 1.0 # fixed
        # see bezier.py to graph stuff and some extra settings?
        return max(0, min(1, (3*u1*((1-t)**2))*t+3*u2*(1-t)*(t**2)+t**3))

    def update_class_weights(self, syn_ratio):
        # Bring the weights from being the same increasling close to reflecting the imbalance
        # instead of one try the cumsum (or normalize the weights)
        return 1 - (self.imbalance_weights * syn_ratio)

    # samples a class according to the imbalance in the dataset with weighted distributions 
    def balanced_sample(self):
        interval = np.random.uniform() * self.class_w_cum[-1]
        i = np.searchsorted(self.class_w_cum, interval)
        # i is the index of the weight corresponding to the class that needs to be generated
        return self.sort_map[i]
        
    def synthetic_sample(self):
        # if synthetic, sample to composotate for the class imbalance
        c = self.balanced_sample()
        # Sample a random image from that class
        s_idx = random.choice(self.syn_sort[c])
        return self.syn[s_idx]

    # Maybe make a choise if you want a balanced sample from the real data as well
    def __getitem__(self, index):
        r = np.random.uniform()
        # not to sure about the grow f here
        if (r < self.syn_ratio) and self.ridx < self.r_samples:
            # sample real
            # consider calling balanced sample here as well
            s = self.sos[self.ridx]
            self.ridx += 1 # a teranry check is less expensive than a modulo, although wraapping might be desired
        else:
            s = self.synthetic_sample()
            self.syn_counter[s[1]] += 1
            
        
        self.idx += 1
        if self.idx >= self.nsamples:
            self.t += self.t_incr
            print("t", self.t)
            n_syn_ratio = self.__bezier(self.t, self.u1, self.u2)
            print("new ratio: ", n_syn_ratio)
            n_ws = (self.update_class_weights(n_syn_ratio))**(self.syn_ratio*10)
            # normalize
            n_ws /= np.sum(n_ws)
            self.class_w_cum = np.cumsum(sorted(n_ws))
            print("new weights", n_ws)
            print("sorted cum", self.class_w_cum)
            self.syn_ratio = n_syn_ratio
            # self.ridx = 0 # Uncomment!
            self.idx = 0
            

        return s

    # We will probably update this dynamically, but keep it in sync with the batch size! (nice and divedable?)
    def __len__(self):
        return self.nsamples 

if __name__ == "__main__":
    from collections import Counter
    t = [SOSDataset.Rescale((232, 232)), SOSDataset.RandomCrop((SOSDataset.DATA_W, SOSDataset.DATA_H))]
    epochs=20
    hd = HybridDataset(epochs=epochs, transform=t, train=True, grow_f=0.50)
    samples = len(hd)
    for epoch in range(epochs+2):
        classes = Counter()
        for s in range(samples):
            try:
                classes[int(hd[s][1])] += 1
            except:
                print("Incorrect class label I think?")
                print("idx", s)
                print("idx", hd[s], "len", len(hd[s]))

        print("Real examples:", hd.ridx)
        hd.ridx = 0 # remove!
        print("Syn", sorted(hd.syn_counter.items(), key=lambda pair: pair[0], reverse=False))
        hd.syn_counter = Counter()
        print("All", sorted(classes.items(), key=lambda pair: pair[0], reverse=False))
        print("------------------------------")
        # exit(0)

