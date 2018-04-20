import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import vae_pytorch
# Model is loaded automatically if you supply arguments

model = vae_pytorch.model
DATA_H = vae_pytorch.DATA_H
DATA_W = vae_pytorch.DATA_W
DATA_C = vae_pytorch.DATA_C

def compare(data_loader):
    # toggle model to test / inference mode
    model.eval()

    # each data is of args.batch_size (default 128) samples
    for di, (data, labels) in enumerate(data_loader):
        if vae_pytorch.args.cuda:
            data = data.cuda()

        data = Variable(data) # Unneeded?
        with torch.no_grad():
            recon_batch, _, _ = model(data)
            recon_batch = recon_batch.view(vae_pytorch.args.batch_size, -1, DATA_W, DATA_H)
            data_ordered = torch.Tensor(5, DATA_C, DATA_W, DATA_H)
            recon_ordered = torch.Tensor(5, DATA_C, DATA_W, DATA_H)
            for i in range(5):
                for lbl_idx, n in enumerate(labels):
                    if n.item() == i % 5:
                        break
                # print(recon_ordered[i])
                # print(n)
                # print(recon_batch[n])
                recon_ordered[i] = recon_batch[lbl_idx]
                data_ordered[i] = data[lbl_idx]

            n = min(data.size(0), 5)
            # for the first 128 batch of the epoch, show the first 8 input digits
            # with right below them the reconstructed output digits
            # the -1 is decide dim_size yourself, so could be 3 or 1 depended on color channels
            # I think we don't need the data view?
            # comparison = torch.cat([data[:n],
            #                          recon_batch.view(argsqqq.batch_size, -1, DATA_W, DATA_H)[:n]])
            comparison = torch.cat([data_ordered, recon_ordered])
            save_image(comparison.data.cpu(),
                       'comparison/reconstruction' + str(di) + '.png', nrow=n)


data = vae_pytorch.test_loader
compare(data)
