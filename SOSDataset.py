import cv2
from random import randint, gauss
import numpy as np
from numpy import linalg as LA
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.utils import save_image

# disable h5py warning, but disables pytorch warnings as well!!!
np.warnings.filterwarnings('ignore')

# was 256, this is after cropping. Used to be 227x227 with crop, but 224 (even) makes the math easier
DATA_W = 225
DATA_H = 225
DATA_C = 3

class RandomColorShift(object):
    # source for fancy colorshift
    # https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image

    def __call__(self, s):
        im = s[0].astype(np.int16)
        # add = [gauss(0, 12), gauss(0, 12), gauss(0, 12)]
        add = [gauss(0, 10), gauss(-0.5, 3.5), gauss(0, 9.5)]
        add_v = np.tile(add, (DATA_W, DATA_H, 1)).astype(np.int16)
        return (np.add(im, add_v)).clip(0, 255).astype(np.uint8), s[1]

class Rescale(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, s):
        # default interpolation=cv2.INTER_LINEAR (rec., fast and ok quality)
        return cv2.resize(s[0], self.output_size), s[1]

class RandomCrop(object):

    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, s):
        h, w = s[0].shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        crop_im = s[0][top : top + new_h, left : left + new_w]
        return crop_im, s[1]

class RandHorizontalFlip(object):

    def __call__(self, s):
        if randint(0,1):
            return np.flip(s[0], 1).copy(), s[1]
        else:
            return s

class ToTensor(object):

    def __init__(self):
        self.t_transform = transforms.ToTensor()

    def  __call__(self, s):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # You can do the reshape((W,H,C)) to get the original (numpy format) back
        # im = torch.from_numpy(s[0].transpose((2,0,1))).float()
        return self.t_transform(s[0]), torch.Tensor([s[1]]).byte()

class Normalize(object):

    def __call__(self, s):
        return s[0] / 255, s[1]

class Normalize01(object):

    """Normalize between 0-1, from -1 and 1"""

    def __call__(self, s):
        return (s[0] + 1)/2, s[1]

class NormalizeMin1_1(object):

    """Normalize between 0-1"""

    def __call__(self, s):
        return (s[0] + 1)/2, s[1]

class NormalizeMean(object):

    def __call__(self, s):
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5))
        return normalize(s[0]), s[1]

class NormalizeMeanVGG(object):

    def __call__(self, s):
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
        return normalize(s[0]), s[1]

class Log(object):

    def __call__(self, s):
        return s[0].log(), s[1]

class Grayscale(object):

    def __call__(self, s):
        pass

class SOSDataset(Dataset):

    # Maybe add the ability to either load data from disk or in RAM
    def __init__(self, train=True, transform=None, datadir="../Datasets/", 
                 grayscale=False, load_ram=False, extended=False):

        self.datadir = datadir
        self.train = train
        self.test_data = []
        self.train_data = []
        self.load_ram = load_ram
        if transform:
            self.transform = transforms.Compose(transform)
            self.transform_name = ''.join([t.__class__.__name__ for t in transform])
        else:
            self.transform = None

        # Read in the .mat file
        if extended:
            import scipy.io as sio
            self.datadir += "ESOS/"
            f = sio.loadmat(self.datadir + "imgIdx.mat")
            imgIdx = f["imgIdx"]
            sos_it = zip(imgIdx["istest"][0,:],imgIdx["label"][0,:],imgIdx["name"][0,:])
            mat_get = lambda t: t[0]
        else:
            import h5py # for newer (?) .mat importing
            self.datadir += "SOS/"
            f = h5py.File(self.datadir + "imgIdx.mat")
            imgIdx = f["imgIdx"]
            sos_it = zip(imgIdx["istest"][:,0],imgIdx["label"][:,0],imgIdx["name"][:,0])
            mat_get = lambda t: f[t]

        for istest, label, fname in sos_it:
            im = mat_get(fname)
            if not extended:
                im = np.array(im, dtype=np.uint8).tostring().decode("ascii")

            if mat_get(istest)[0]:
                if not self.train:
                    self.test_data.append((im, mat_get(label)[0]))
            else:
                if self.train:
                    self.train_data.append((im, mat_get(label)[0]))
        # 10966 for train, 2741 for test
        self.nsamples = len(self.train_data) if self.train else len(self.test_data)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        # comment if preprocessing seems undoable
        # if self.preprocessed:
        #     s = self.train_data if self.train else self.test_data
        #     return s[0][index], s[1][index]

        s = self.train_data[index] if self.train else self.test_data[index]
        s = cv2.cvtColor(cv2.imread(self.datadir + s[0]), cv2.COLOR_BGR2RGB), s[1]
        return self.transform(s)

if __name__ == "__main__":
    # load preprocess
    transform = [Rescale((256, 256)), RandomCrop((DATA_W, DATA_H)), RandomColorShift(), ToTensor(), Normalize(), NormalizeMeanVGG()]
    # transform = [Rescale((256, 256)), 
    #               ToTensor(), Normalize()]
    dataset = SOSDataset(train=False, transform=transform, extended=True)
    print(torch.unique(dataset[1][0], sorted=True))
    exit(0)
    for i in range(0, 10):
        cv2.imshow("im", cv2.cvtColor(dataset[i][0], cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.imshow("im", cv2.cvtColor(dataset[i][0], cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)

    # Save preprocess 
    # data_transform = [Rescale((DATA_2W, DATA_H)), FlattenArrToTensor(), Normalize()]
    # dataset = SOSDataset(train=True, transform=data_transform, preprocessed=False)
    # dataset.save()
