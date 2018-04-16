import os, argparse
import random
import cv2
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset

# disable h5py warning
np.warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--z-dims', type=int, default=20, metavar='N',
                    help='dimenstionality of the latent z variable')
parser.add_argument('--load-model', type=str, default='', metavar='P', 
                    help='load a torch model from given path (default: create new model)')
parser.add_argument('--start-epoch', type=int, default=1, metavar='N', 
                    help='epoch to start at (only affects logging)')
parser.add_argument('--test-epoch', type=int, default=1, metavar='N', 
                    help='when to run a test epoch')
parser.add_argument('--full_con_size', type=int, default=400, metavar='N', 
                    help='size of the fully connected layer (prob. leave it)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Width and height of a sample from the dataset
DATA_W = 256
DATA_H = 256
DATA_SIZE = DATA_W * DATA_H


class Rescale(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, s):
        # default interpolation=cv2.INTER_LINEAR (rec, fast and ok quality)
        return cv2.resize(s[0], self.output_size), s[1]

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, s):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # You can do the transpose((1,2,0)) to get the numpy order back
        return torch.from_numpy(s[0].transpose((2,0,1))).float(), s[1]



class SOSDataset(Dataset):

    # Maybe add the ability to either load data from disk or in RAM
    def __init__(self, train=True, transform=None, datadir="../Datasets/SOS/", load_ram=False):
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
            # print(s[0].shape)
            # s = s[0].transpose((2,0,1)), s[1]
            # print(s[0].shape)
        if self.transform:
            s = self.transform(s)
        return s

# connections through the autoencoder bottleneck
# in the pytorch VAE example, this is 20
# ZDIMS = # (was 20)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

data_transform = transforms.Compose([Rescale((DATA_W, DATA_H)), ToTensor()])

# shuffle data at every epoch
# TODO: experiment with load_ram = True

train_loader = torch.utils.data.DataLoader(
    SOSDataset(train=True, transform=data_transform, load_ram=False),
    batch_size=args.batch_size, shuffle=True, **kwargs)
#     # datasets.MNIST('data', train=True, download=True,
#     #                transform=transforms.ToTensor()),
#     # batch_size=args.batch_size, shuffle=True, **kwargs)

# Same for test data
test_loader = torch.utils.data.DataLoader(
    SOSDataset(train=False, transform=data_transform, load_ram=False),
    batch_size=args.batch_size, shuffle=True, **kwargs)
    # datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    # batch_size=args.batch_size, shuffle=True, **kwargs)

# for idx, s in enumerate(test_loader):
#     cv2.imshow("img", np.swapaxes(s[0][0].data.numpy(),0,2))
#     print(s[1][0])
#     cv2.waitKey(0)
#     cv2.imshow("img", np.swapaxes(s[0][1].data.numpy(),0,2))
#     print(s[1][1])
#     cv2.waitKey(0)
#     if idx > 3:
#         break
#     print("🌸")


# exit(0)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # ENCODER
        # 28 x 28 pixels = 784 input pixels (for minst), 400 outputs
        self.fc1 = nn.Linear(DATA_SIZE, args.full_con_size)
        # rectified linear unit layer from 400 to 400
        # max(0, x)
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(args.full_con_size, args.z_dims) # mu layer
        self.fc22 = nn.Linear(args.full_con_size, args.z_dims) # logvariance layer
        # this last layer bottlenecks through args.z_dims connections

        # DECODER
        # from bottleneck to hidden 400
        self.fc3 = nn.Linear(args.z_dims, args.full_con_size)
        # from hidden 400 to 784 outputs
        self.fc4 = nn.Linear(args.full_con_size, DATA_SIZE)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x: Variable) -> (Variable, Variable):
        """Input vector x -> fully connected 1 -> ReLU -> (fully connected
        21, fully connected 22)

        Parameters
        ----------
        x : [128, 784] matrix; 128 digits of 28x28 pixels each

        Returns
        -------

        (mu, logvar) : ZDIMS mean units one for each latent dimension, ZDIMS
            variance units one for each latent dimension

        """

        # h1 is [128, 400] (batch, + the size of the first fully connected layer)
        h1 = self.relu(self.fc1(x))  # type: Variable
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        """THE REPARAMETERIZATION IDEA:

        For each training sample (we get 128 batched at a time)

        - take the current learned mu, stddev for each of the ZDIMS
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians

        Parameters
        ----------
        mu : [128, ZDIMS] mean matrix
        logvar : [128, ZDIMS] variance matrix

        Returns
        -------

        During training random sample from the learned ZDIMS-dimensional
        normal distribution; during inference its mean.

        """

        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation
            std = logvar.mul(0.5).exp_()  # type: Variable
            # - std.data is the [128,ZDIMS] tensor that is wrapped by std
            # - so eps is [128,ZDIMS] with all elements drawn from a mean 0
            #   and stddev 1 normal distribution that is 128 samples
            #   of random ZDIMS-float vectors
            eps = Variable(std.data.new(std.size()).normal_())
            # - sample from a normal distribution with standard
            #   deviation = std and mean = mu by multiplying mean 0
            #   stddev 1 sample with desired std and mu, see
            #   https://stats.stackexchange.com/a/16338
            # - so we have 128 sets (the batch) of random ZDIMS-float
            #   vectors sampled from normal distribution with learned
            #   std and mu for the current input
            return eps.mul(std).add_(mu)

        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu

    def decode(self, z: Variable) -> Variable:
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x.view(-1, DATA_SIZE))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE()

if args.load_model:
    model.load_state_dict(
        torch.load(args.load_model, map_location=lambda storage, loc: storage))
if args.cuda:
    model.cuda()


def loss_function(recon_x, x, mu, logvar) -> Variable:
    # how well do input x and output recon_x agree?
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, DATA_SIZE))

    # KLD is Kullback–Leibler divergence -- how much does one learned
    # distribution deviate from another, in this specific case the
    # learned distribution from the unit Gaussian

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # note the negative D_{KL} in appendix B of the paper
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
	## This line was/is not in the original pytorch code
    KLD /= args.batch_size * DATA_SIZE

    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    return BCE + KLD

# Dr Diederik Kingma: as if VAEs weren't enough, he also gave us Adam!
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    # toggle model to train mode
    model.train()
    train_loss = 0
    # in the case of MNIST, len(train_loader.dataset) is 60000
    # each `data` is of args.batch_size samples and has shape [128, 1, 28, 28]

    # if you have labels, do this
    # for batch_idx, (data, _) in enumerate(train_loader):
    for batch_idx, (data, _) in enumerate(train_loader):
        # print(batch_idx, data)
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()

        # push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, logvar = model(data)
        # calculate scalar loss
        loss = loss_function(recon_batch, data, mu, logvar)
        # calculate the gradient of the loss w.r.t. the graph leaves
        # i.e. input variables -- by the power of pytorch!
        loss.backward()
        train_loss += loss.item()

        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item() / len(data)))

    # print('====> Epoch: {} Average loss: {:.4f}'.format(
    #       epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    # toggle model to test / inference mode
    model.eval()
    test_loss = 0

    # each data is of args.batch_size (default 128) samples
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            # make sure this lives on the GPU
            data = data.cuda()

        # we're only going to infer, so no autograd at all required: volatile=True
        data = Variable(data) # Unneeded?
        with torch.no_grad():
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                # for the first 128 batch of the epoch, show the first 8 input digits
                # with right below them the reconstructed output digits
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, 1, DATA_W, DATA_H)[:n]])
                save_image(comparison.data.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Epoch: {} Test set loss: {:.4f}'.format(epoch, test_loss))


for epoch in range(args.start_epoch, args.epochs + 1):
    train(epoch)

    if epoch % 2000 == 0:
    	torch.save(model.state_dict(), 'models/vae-%s.pt' % (epoch))
    # load:
    # model.load_state_dict(torch.load('models/vae..', map_location=lambda storage, loc: storage))

    # 64 sets of random ZDIMS-float vectors, i.e. 64 locations / MNIST
    # digits in latent space
    sample = Variable(torch.randn(64, args.z_dims))
    if args.cuda:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()
    
    # Write out data and print loss
    if epoch % 1000 == 0:
        test(epoch)

        # save out as an 8x8 matrix of MNIST digits
        # this will give you a visual idea of how well latent space can generate things
        # that look like digits
        save_image(sample.data.view(64, 1, DATA_W, DATA_H),
               'results/sample_' + str(epoch) + '.png')
