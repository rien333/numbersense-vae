import cv2
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset

# disable h5py warning
np.warnings.filterwarnings('ignore')

DATA_W = 256
DATA_H = 256
DATA_C = 3

class Rescale(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, s):
        # default interpolation=cv2.INTER_LINEAR (rec., fast and ok quality)
        return cv2.resize(s[0], self.output_size), s[1]

class ToTensor(object):

    def  __call__(self, s):
        # see below for channel explanition
        im = torch.from_numpy(s[0].transpose((2,0,1))).float()
        return im, torch.Tensor([s[1]]).byte()
    
# Maybe change this to a seperate flatten method
class FlattenArrToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, s):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # You can do the reshape((W,H,C)) to get the original (numpy format) back
        # flatten/(i.e. view(-1)) deals with grayscale and RGB case
        return torch.from_numpy(s[0]).view(-1).float(), torch.Tensor([s[1]]).byte()

class Normalize(object):

    def __call__(self, s):
        return s[0] / 255, s[1]

class Grayscale(object):

    def __call__(self, s):
        pass

class SOSDataset(Dataset):

    # Maybe add the ability to either load data from disk or in RAM
    def __init__(self, train=True, transform=None, datadir="../Datasets/SOS/", 
                 grayscale=False, load_ram=False, preprocessed=False):
        import h5py # for .mat importing

        self.datadir = datadir
        self.train = train
        self.test_data = []
        self.train_data = []
        self.load_ram = load_ram
        self.preprocessed = preprocessed
        if transform:
            self.transform = transforms.Compose(transform)
            self.transform_name = ''.join([t.__class__.__name__ for t in transform])
        else:
            self.transform = None

        if preprocessed:
            self.load_ram = True
            if self.train:
                ims = torch.load(self.datadir + "train.pth")
                # also load the labels
                self.train_data = ims, torch.load(self.datadir + "train_labels.pth")
            else:
                ims = torch.load(self.datadir + "test.pth")
                self.test_data = ims, torch.load(self.datadir + "test_labels.pth")
            self.nsamples = ims.shape[0]
            return

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
    
    # hmhmm this should rather save the images I guess?
    def save(self):
        """Save a tensor with all given transformations applied"""

        pre_data = torch.zeros(self.nsamples, DATA_W*DATA_H*DATA_C).float()
        pre_data_lbl = torch.zeros(self.nsamples).byte()
        data = self.train_data if self.train else self.test_data
        for idx, s in enumerate(data):
            s = cv2.imread(self.datadir + s[0]), s[1]
            s = self.transform(s)
            pre_data[idx] = s[0]
            pre_data_lbl[idx] = s[1]
        f_dir = "%s/%s/" % (self.datadir, self.transform_name)
        f_dir = os.path.join(os.path.dirname(__file__), f_dir) # relative paths
        f_name = "%s.pth" % ("train" if self.train else "test")
        f_name_lbl = "%s_labels.pth" % ("train" if self.train else "test")
        os.makedirs(f_dir, exist_ok=True)
        torch.save(pre_data, f_dir + f_name)
        torch.save(pre_data_lbl, f_dir + f_name_lbl)

    def __getitem__(self, index):
        # comment if preprocessing seems undoable
        # if self.preprocessed:
        #     s = self.train_data if self.train else self.test_data
        #     return s[0][index], s[1][index]

        s = self.train_data[index] if self.train else self.test_data[index]
        if not self.load_ram:
            s = cv2.imread(self.datadir + s[0]), s[1]
        if self.transform:
            s = self.transform(s)
        return s

if __name__ == "__main__":
    # load preprocess
    # dataset = SOSDataset(train=False, datadir="../Datasets/SOS/RescaleToTensorNormalize/", 
    #                      preprocessed=True)
    # # normally this has kwargs stuff for cuda loading!!
    # test_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    # print("read stuff")
    # for idx, s in enumerate(test_loader):
    #     cv2.imshow("img", s[0][0].data.numpy().reshape((DATA_W,DATA_H,DATA_C)))
    #     print(s[1][0])
    #     cv2.waitKey(0)
    #     print()
    #     cv2.imshow("img", s[0][1].data.numpy().reshape((DATA_W,DATA_H,DATA_C)))
    #     print(s[1][1])
    #     cv2.waitKey(0)
    #     if idx > 3:
    #         break
    #     print("🌸")

    # Save preprocess 
    data_transform = [Rescale((DATA_W, DATA_H)), FlattenArrToTensor(), Normalize()]
    dataset = SOSDataset(train=True, transform=data_transform, preprocessed=False)
    dataset.save()
