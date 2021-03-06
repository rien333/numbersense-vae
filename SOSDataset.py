import cv2
from random import randint, gauss, uniform
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.utils import save_image
import pickle
from PIL import Image
from imgaug import augmenters as iaa

# disable h5py warning, but disables pytorch warnings as well!!!
np.warnings.filterwarnings('ignore')

# was 256, this is after cropping. Used to be 227x227 with crop, but 224 (even) makes the math easier
DATA_W = 161
DATA_H = 161
DATA_C = 3

class RandomColorShift(object):
    # source for fancy colorshift
    # https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image

    def __call__(self, s):
        im = s[0].astype(np.int16)
        h, w = s[0].shape[:2]
        add = [uniform(0, 12), uniform(0, 12), uniform(0, 12)]
        # add = [gauss(0, 10), gauss(-0.5, 3.5), gauss(0, 9.5)]
        add_v = np.tile(add, (h, w, 1)).astype(np.int16)
        return (np.add(im, add_v)).clip(0, 255).astype(np.uint8), s[1]

class Rescale(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, s):
        # default interpolation=cv2.INTER_LINEAR (rec., fast and ok quality)
        return cv2.resize(s[0], self.output_size,), s[1]

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

class RandomRandomCrop(object):

    """
    Crops a random area of a random size, limited by max_size
    max_f: the max factor by which pixels will be removed from 
    """

    def __init__(self, max_f):
        self.max_f = max_f 

    def __call__(self, s):
        h, w = s[0].shape[:2]
        f = 1 - uniform(0, self.max_f)
        new_h, new_w = (int(d*f) for d in (h,w))
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

    def __init__(self, train=True, transform=None, datadir="../Datasets/",  sorted_loc="/tmp", extended=True):

        self.datadir = datadir
        self.train = train
        self.test_data = []
        self.train_data = []
        if transform:
            self.transform = transforms.Compose(transform)
            self.transform_name = ''.join([t.__class__.__name__ for t in transform])
        else:
            self.transform = None
        self.sorted_loc = sorted_loc + "/sorted_classes_sos_" + str(self.train)+".pickle"

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


    def load_sorted_classes(self):
        # Sorting all the indices by class takes really long for some reason, so save and read from file
        if os.path.isfile(self.sorted_loc):
            with open (self.sorted_loc, 'rb') as f:
                c = pickle.load(f,encoding='latin1')
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

class RandomGrayscale(object):

    def __call__(self, s):
        # default interpolation=cv2.INTER_LINEAR (rec., fast and ok quality)
        g = iaa.Grayscale(abs(gauss(0.0, 0.091)))
        return g.augment_image(s[0]), s[1]

class PerspectiveTransform(object):

    def __call__(self, s):
        p = iaa.PerspectiveTransform(abs(gauss(0.0, 0.095)))
        return p.augment_image(s[0]), s[1]

class ContrastNormalization(object):

    def __call__(self, s):
        c = iaa.ContrastNormalization(abs(gauss(1.0, 0.099)))
        return c.augment_image(s[0]), s[1]

class AugmentWrapper(object):
    
    def __init__(self):
        import Augmentor
        p = Augmentor.Pipeline()
        p.rotate(probability=0.75, max_left_rotation=12, max_right_rotation=12)
        p.zoom(probability=0.7, min_factor=1.00, max_factor=1.06)
        p.random_color(0.3, 0.9, 1.0)
        p.skew(probability=0.7, magnitude=0.24)
        self.p = p.torch_transform()

    def __call__(self, s):
        return self.p(s[0]), s[1]

class ToPILImage(object):

    def __call__(self, s):
        # return self.t(s[0]), s[1]
        return Image.fromarray(s[0]), s[1]

class ToNumpy(object):

    def __call__(self, s):
        return np.array(s[0]), s[1]

if __name__ == "__main__":
    
    transform = [ToPILImage(), AugmentWrapper(), ToNumpy(), RandomColorShift(), ContrastNormalization(), Rescale((250, 250))]
    st = [Rescale((250, 250))]
    # transform = [Rescale((256, 256)), 
    #               ToTensor(), Normalize()]
    dataset = SOSDataset(train=True, transform=transform, extended=False)
    # print(torch.unique(dataset[1][0], sorted=True))
    classes = dataset.load_sorted_classes()
    # for l in classes:
    #     print(len(l))
    t = transforms.Compose(transform)
    st = transforms.Compose(st)
    for i in classes[3]:
        e = dataset[i]
        print(e[1])
        cv2.imshow("norm", st(e)[0])
        cv2.waitKey(0)
        cv2.imshow("crop", t(e)[0])
        cv2.waitKey(0)
    # cv2.imwrite("test.jpg", cv2.cvtColor(dataset[dataset.sorted()[2][8]][0], cv2.COLOR_BGR2RGB))
    
    # Save preprocess 
    # data_transform = [Rescale((DATA_2W, DATA_H)), FlattenArrToTensor(), Normalize()]
    # dataset = SOSDataset(train=True, transform=data_transform, preprocessed=False)
    # dataset.save()
