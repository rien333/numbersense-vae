import conv_vae_pytorch as vae_pytorch
import SOSDataset
import torch
import numpy as np
import cv2
from torchvision import transforms
import os
from torchvision.utils import save_image

# Read in a reconstruction, and produce an output with another model

DATA_W = SOSDataset.DATA_W
DATA_H = SOSDataset.DATA_H
DATA_C = 3
# torch creates a little border between images
b_px = 2

if vae_pytorch.args.cuda:
    print("Please don't use cuda as this will crash stuff")
    exit(0)

model = vae_pytorch.model
# toggle model to test / inference mode
model.eval()
# if not training the VAE will select the zs with highest probability
model.training = False

def read_crop(im, n):
    x = b_px + (n * (DATA_W+b_px))
    y = b_px
    return im[y:y+DATA_H, x:x+DATA_W]


im_p = "/tmp/beta0.212/reconstruction_1.png"
n_im_p = "/tmp/" + os.path.basename(im_p)
im = cv2.cvtColor(cv2.imread(im_p), cv2.COLOR_BGR2RGB)
trans = transforms.ToTensor()

ims = torch.cat([trans(read_crop(im, n)).view(-1, DATA_C, DATA_H, DATA_W) for n in range(7)])
recon_batch, _, _ = model(ims) # Do not use cuda as this will def crash stuff
comparison = torch.cat([ims.cpu(), recon_batch.cpu()])
# call pytorch conv shit here
# print(comparison.shape)
save_image(comparison, n_im_p, nrow=7)
os.system("imgcat " + n_im_p)
