import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class Rescale(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, s):
        # default interpolation=cv2.INTER_LINEAR (rec., fast and ok quality)
        return cv2.resize(s[0], self.output_size), s[1]

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, s):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # You can do the transpose((W,H,C)) to get the original (numpy format) back
        # flatten/(i.e. view(-1)) deals with grayscale and RGB case
        return torch.from_numpy(s[0]).view(-1).float(), s[1]

class Normalize(object):
    
    def __call__(self, s):
        return s[0] / 255, s[1]

class SOSDataset(Dataset):

    # Maybe add the ability to either load data from disk or in RAM
    def __init__(self, train=True, transform=None, datadir="../Datasets/SOS/", grayscale=False, load_ram=False):
        import h5py # for .mat importing

        self.datadir = datadir
        self.train = train
        self.test_data = []
        self.train_data = []
        self.transform = transform
        self.load_ram = load_ram
        f = h5py.File(self.datadir + "imgIdx.mat")
        imgIdx = f["imgIdx"]

        for istest, label, fname in zip(imgIdx["istest"][:,0], imgIdx["label"][:,0],
                                        imgIdx["name"][:,0]):
            # get filename
            im = np.array(f[fname], dtype=np.uint8).tostring().decode("ascii")
            if load_ram: # load actual image, faster but more resources
                # Maybe it's smart to do some preprocessing here to save on ram
                # Yeah I guess like the transforms here?
                im = cv2.imread(self.datadir+im)

            if f[istest][0]:
                if not self.train:
                    self.test_data.append((im, f[label][0,0]))
            else:
                if self.train:
                    self.train_data.append((im, f[label][0,0]))

        self.nsamples = len(self.train_data) if self.train else len(self.test_data)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        s = self.train_data[index] if self.train else self.test_data[index]
        if not self.load_ram:
            s = cv2.imread(self.datadir + s[0]), s[1]
        if self.transform: # "if" needed? (esp. with preprocessing step)
            s = self.transform(s)
        return s
