from random import randint
import conv_vae_pytorch as vaepytorch

import torch
from torch.autograd import Variable
from torchvision.utils import save_image

# Run with batch size 1!

model = vaepytorch.model
model.eval()
model.train = False
test_loader = vaepytorch.test_loader
DATA_H = vaepytorch.DATA_H
DATA_W = vaepytorch.DATA_W
DATA_C = vaepytorch.DATA_C

# sample = Variable(torch.randn(1, vaepytorch.args.z_dims))
# Resize to batch size
n_ims = vaepytorch.args.batch_size
im_idxs = [randint(0, 1380) for i in range(n_ims)]
ims = torch.zeros(n_ims, DATA_C, DATA_H, DATA_W).cuda()
for idx, i in enumerate(im_idxs):
    ims[idx, :, :, :] = vaepytorch.test_loader.dataset[i][0].view(-1,DATA_H, DATA_W)
# im = vaepytorch.test_loader[im_idxs]

mu, logvar = model.encode(ims.cuda())
sample = model.reparameterize(mu, logvar)
add = torch.zeros(n_ims, vaepytorch.args.z_dims).cuda()
add[:, 0] = 0.1 # consider moving over multiple dims
# add[0:1] = (0.05, 0.5) # consider moving over multiple dims

for i in range(140):
    # sample = torch.add(sample, add)
    # torch.add(sample, add, nsample)
    sample.add_(add)
    decoded_sample = model.decode(sample)
    out = torch.cat([ims, decoded_sample.data.view(n_ims, DATA_C, DATA_H, DATA_W)], dim=2)
    # out = out.view(2,  -1, DATA_C, DATA_H, DATA_W)
    save_image(out, 'misc/test/test%s.png' % (i))
