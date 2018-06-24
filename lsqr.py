# import matplotlib
# matplotlib.use('agg') # remove dependency on tkinter (thus should be first)
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit
import numpy as np
import matplotlib
import random
import image_generator
import conv_vae_pytorch as vae
import SOSDataset
import torch
from torchvision import transforms
import cv2
from torchvision.utils import save_image
import sys

DATA_W = SOSDataset.DATA_W
DATA_H = SOSDataset.DATA_H
DATA_C = SOSDataset.DATA_C
C = 4 # highest class value
data_t = transforms.Compose([transforms.ToTensor()])
ZDIMS = vae.args.z_dims


model = vae.model
# toggle model to test / inference mode
model.eval()
# if not training the VAE will select the zs with highest probability
model.training = False

if vae.args.cuda:
    model.cuda() # need to call this here again 

nan_eps = 1e-6

# idk what to do with epsilon but maybe just add a small constant to the result
def residual(params, cum_area, N, activation, eps):
    bias = params['B0'].value # number of objects coefficient
    b1 = params['B1'].value # number of objects coefficient
    b2 = params['B2'].value # cum. area size coefficient

    # Consider normalizing this early on
    N_norm = (N + nan_eps) / (C+nan_eps)
    cum_area_norm = (cum_area+nan_eps) / ((DATA_H*DATA_W) + nan_eps)

    # print(N)
    # print(cum_area_norm)
    # residual between model fit and ground truth neuron activations
    model = b1 * np.log(N_norm) + b2 * np.log(cum_area_norm) + bias
    # return (activation - model) / eps # not sure what this does, and performance is better without
    return activation - model

params = Parameters()

# You could give some of the values @stoianov2012 report as initial values
params.add('B0', value=0.1) # bias
params.add('B1', value=0.5) # Number of objects
params.add('B2', value=0.5) # Area
eps = 5e-5

im_bsize = 128
samples = im_bsize * 250 # was like 70?
# I guess you can pregenerate the object sizes like this but idk
# obj_s = (DATA_W*DATA_H) * 0.15

N = np.random.randint(low=0, high=5, size=samples)
cum_area = np.array([])
activations = [] # append numpy arrays here, and then np.array(act) converts it to a 2d arr

ni = 0
Nsize = len(N)
switch = 0.19 # chance of switching up source images
_, obj_f = image_generator.single_obj()
b = image_generator.background_im()

while ni != Nsize: # can probably be done beter
    sys.stdout.write('Done with %s/%s\r' % (ni, Nsize))
    sys.stdout.flush()
    ims = torch.Tensor()
    for _ in range(im_bsize):
        if random.random() < switch:
            _, obj_f = image_generator.single_obj()
            b = image_generator.background_im()
        # consider setting the threshold differently for different classes
        im, a = image_generator.generate_image(N[ni], thresh=1.0, background=b.copy(), obj_f=obj_f)
        im = np.uint8(im)
        im = data_t(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).view(1, 3, DATA_H, DATA_W)
        cum_area = np.append(cum_area, a)
        ims = torch.cat((ims, im))
        ni += 1
    # Write images out to say if they look good
    # zs = mu in train time so you don't have to call reparameterize
    with torch.no_grad():
        mu, logvar = model.module.encode(ims.cuda())
        activations.append(model.module.reparameterize(mu, logvar).cpu().numpy())

np.set_printoptions(precision=1, linewidth=238, suppress=True, edgeitems=9)
activations = np.array(activations).reshape(-1, ZDIMS)
print(activations.shape)
print(cum_area.shape)
# For sanity, plot N versus cum area?
fig = plt.figure(figsize=(20, 16))
plt.scatter(N, cum_area)
plt.xlabel('Number of objecs')
plt.ylabel('Cumaltative area')
plt.show()

noi = []
R_sum = 0
for i, a in enumerate(activations.T): # iterate over collumns
    out = minimize(residual, params, args=(cum_area, N, a, eps),) # note changed method
    R = 1 - out.residual.var() / np.var(a)
    if R < 0.033:
        continue
    print("\nNeuron", i)
    print("-"*20)
    report_fit(out)
    print("R^2:", R)
    noi.append(i)
    R_sum += R

print("R sum", (R_sum / len(noi)))
np.save("noi2.npy", np.array(noi))
np.save("activations2.npy", activations)
np.save("cum_area2.npy", cum_area)
np.save("N2.npy", N)
# fig = plt.figure()
# for i, n in enumerate(noi):
#     ax = fig.add_subplot(2, int(len(noi)/2), i+1, projection='3d')
#     ax.scatter(N, cum_area, activations[:, n], c=np.random.random(size=(1,3)))
#     ax.set_title('Neuron %s' % (n))
#     ax.set_xlabel('N')
#     ax.set_ylabel('A')
#     ax.set_zlabel('R')
# plt.show()

# what wikipedia answer to what R^2 is
# print("R^2:",  1 - (np.sum((out.residual)**2) / np.sum((a - np.mean(a))**2)))
