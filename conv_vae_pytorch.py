import os, argparse
import random
import numpy as np
import SOSDataset
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='Annotated PyTorch VAE with conv layers')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--z-dims', type=int, default=20, metavar='N',
                    help='dimenstionality of the latent z variable')
parser.add_argument('--full-con-size', type=int, default=400, metavar='N',
                    help='size of the fully connected layer')
parser.add_argument('--load-model', type=str, default='', metavar='P',
                    help='load a torch model from given path (default: create new model)')
parser.add_argument('--start-epoch', type=int, default=1, metavar='N',
                    help='epoch to start at (only affects logging)')
parser.add_argument('--test-interval', type=int, default=10, metavar='N',
                    help='when to run a test epoch')
parser.add_argument('--disable-train', action='store_true', default=False, 
                    help='Disable training of model. Allows for importing this as a module.')
parser.add_argument('--grayscale', action='store_true', default=False, help='Train on grayscale data')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Width and height of a sample from the dataset
DATA_W = SOSDataset.DATA_W
DATA_H = SOSDataset.DATA_H
DATA_C = SOSDataset.DATA_C # Color component dimension size
DATA_SIZE = DATA_W * DATA_H * DATA_C

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Try with and without normalize mean
data_transform = [SOSDataset.Rescale((256, 256)), SOSDataset.RandomCrop((DATA_W, DATA_H)), 
                  SOSDataset.ToTensor(), SOSDataset.Normalize()]

# Some people recommended this type of normalisation for natural images, depedends on the input being
# a RGB torch tensor however
# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], # Hardcoded
#                                  std=[0.5, 0.5, 0.5])

# shuffle data at every epoch
# TODO: experiment with load_ram = True
pre_dir = "../Datasets/SOS/RescaleToTensorNormalize/"

# preprocessing seems slower actually
train_loader = torch.utils.data.DataLoader(
    # SOSDataset.SOSDataset(train=False, preprocessed=True, datadir=pre_dir),
    # batch_size=args.batch_size, shuffle=True, **kwargs)
    SOSDataset.SOSDataset(train=True, transform=data_transform, load_ram=False),
    batch_size=args.batch_size, shuffle=True, **kwargs)
    # datasets.MNIST('data', train=True, download=True,
    #                transform=transforms.ToTensor()),
    # batch_size=args.batch_size, shuffle=True, **kwargs)

# Same for test data
test_loader = torch.utils.data.DataLoader(
    # SOSDataset.SOSDataset(train=False, preprocessed=True, datadir=pre_dir),
    # batch_size=args.batch_size, shuffle=True, **kwargs)
    SOSDataset.SOSDataset(train=False, transform=data_transform, load_ram=False),
    batch_size=args.batch_size, shuffle=True, **kwargs)
    # datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    # batch_size=args.batch_size, shuffle=True, **kwargs)

class CONV_VAE(nn.Module):
    def __init__(self):
        super(CONV_VAE, self).__init__()

        # Question: do you use batch normalisation after every conv layer
        # (yes, given Deep Pyramidal Residual Networks?)
        # (yes, Deep Residual Learning for Image Recognition)

        # BN before activation because of 
        # www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/dgqaxu2/

        # instead of maxpooling, consider conv nets with stride 2 to do downsampling

        # ENCODER (cnn architecture based on simple vgg16)
        self.conv1 = nn.Sequential(	# input shape (3, DATA_H, DATA_W)
            nn.Conv2d(
                in_channels=3,		# RGB
                out_channels=32,        # output depth
                kernel_size=3,
                stride=1,
                padding=1
            ),				# out (64, DATA_H, DATA_W) should be same HxW as in
            nn.ReLU(),                  # inplace=True saves memory but discouraged (worth the try)
            nn.BatchNorm2d(32),         # C channel input, 4d input (NxCxHxW)
            nn.Conv2d(32, 32, 3, 1, 1), # 64 filter depth from prev layer
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),          # k=2x2,s=2 (by vgg16) out shape (16, 14, 14)(red. by 2)
        )

        # These two in the middle can maybe downsample with a conv
        self.conv2 = nn.Sequential(     # in shape (64, DATA_H/2, DATA_W/2)
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # nn.Conv2d(64, 64, 3, 1, 1),
            # nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # nn.Conv2d(64, 64, 3, 1, 1),
            # nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
        )
        
        # Good idea to make this one and the last one double
        self.conv4 = nn.Sequential(	# DATA_W/H is ~= 28
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.BatchNorm2d(128),
            # nn.Conv2d(128, 128, 3, 1, 1),
            # nn.ReLU(),
            # nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
        )

        # conv4/conv-out should be flattened

        # fc1 conv depth * (DATA_W*DATA_H / (number of pools * 2)) (with some rounding)
        self.fc1 = nn.Linear(128*14*14, args.full_con_size) # relu
        self.fc21 = nn.Linear(args.full_con_size, args.z_dims) # mean network, linear
        self.fc22 = nn.Linear(args.full_con_size, args.z_dims) # variance network, linear
        self.relu = nn.ReLU()

        # Old Encoder
        # # 28 x 28 pixels = 784 input pixels (for minst), 400 outputs
        # self.fc1 = nn.Linear(DATA_SIZE, args.full_con_size)
        # # rectified linear unit layer from 400 to 400
        # self.relu = nn.ReLU()
        # self.fc21 = nn.Linear(args.full_con_size, args.z_dims) # mu layer
        # self.fc22 = nn.Linear(args.full_con_size, args.z_dims) # logvariance layer
        # # this last layer bottlenecks through args.z_dims connections

        # DECODER
        # Should use transconv and depooling
        
        self.fc3 = nn.Linear(args.z_dims, args.full_con_size) # Relu
        # form the decoder output to a conv shape
        # should be the size of a convolution/the last conv size
        # 128*14*14 * a few (4) upsampling = the original input size
        self.fc4 = nn.Linear(args.full_con_size, 128*14*14)

        # stride in 1st covn. = 1 bc we don't wanna miss anything (interdependence) from the z layer
        # otherwhise upsample so that we learn the upsampling/scaling process (Pooling doesn't not learn
        # anything in respect to how it should scale)
        # potential source:
        # A Hybrid Convolutional Variational Autoencoder for Text Generation """
        # A deconvolutional layer (also referred to as transposed convolutions (Gulrajani, 2016) and fractionally
        # strided convolutions (Radford et al., 2015)) performs spatial up-sampling of its inputs and
        # is an integral part of latent variable genera- tive models of images (Radford et al., 2015; Gulra- jani
        # et al., 2016) """ 

        # If learning to upsample by convolution does not work out, nn.Upsample also can apply various
        # scalers (but is not learned I think)

        # z is pretty important, so set stride=1 to not miss anything first
        self.t_conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 1, 1), 
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.t_conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        self.t_conv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64), 
        )

        self.t_conv_final = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 3, 2, 1, output_padding=0), # RGB, no relu or batch norm. on output
            nn.Sigmoid() # output between 0 and 1
        )
        
        # final output (no batchnorm needed)
        # but do we need Relu?! (yeah for non-linear learning)
        # self.t_conv3 = nn.ConvTranspose2d(64, 1, 3, 1, 1)
        

        # old Decoder
        # from bottleneck to hidden 400
        # self.fc3 = nn.Linear(args.z_dims, args.full_con_size)
        # # from hidden 400 to 784 outputs
        # self.fc4 = nn.Linear(args.full_con_size, DATA_SIZE)
        # self.sigmoid = nn.Sigmoid()

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
        # I don't think these need to be seperate variables, see
        # http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        flatten_c4 = c4.view(c4.size(0), -1) # flatten conv2 to (batch_size, red_data_dim)
        h1 = self.relu(self.fc1(flatten_c4))
        return self.fc21(h1), self.fc22(h1)

        # # h1 is [128, 400] (batch, + the size of the first fully connected layer)
        # h1 = self.relu(self.fc1(x))  # type: Variable
        # return self.fc21(h1), self.fc22(h1)

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
        # another relu layers that maps h3 to a conv shape
        h4 = self.relu(self.fc4(h3))
        h4_expanded = h4.view(-1, 128, 14, 14) # 14 * (4 * 2x upsamling conv) ~= 227
        up_conv1 = self.t_conv1(h4_expanded)
        up_conv2 = self.t_conv2(up_conv1) # every layer upsamples by 2 basically
        up_conv3 = self.t_conv3(up_conv2)
        return self.t_conv_final(up_conv3) # scale up with image scaling
        # return self.sigmoid(self.fc4(h3))

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        # mu, logvar = self.encode(x.view(-1, DATA_SIZE))
        # mu and logvar are the paramters of the z distribution (after the reparameterization "trick")
        mu, logvar = self.encode(x) # conv layers work on RGB channels
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = CONV_VAE()

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
    # print(logvar)
    # print(torch.sum(logvar))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
	## This line was/is not in the original pytorch code
    KLD /= args.batch_size * DATA_SIZE

    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    return BCE + KLD

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
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()

        # push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, logvar = model(data)
        # print("decoded:", recon_batch.shape)
        # print("data:", data)
        # print("recon_batch:", recon_batch)
        # print(recon_batch.cpu().unique(sorted=True))
        # exit(0)

        # calculate scalar loss
        loss = loss_function(recon_batch, data, mu, logvar)
        # from time import sleep
        # print("Check mem!")
        # sleep(10)
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
                # the -1 is decide dim_size yourself, so could be 3 or 1 depended on color channels
                # I think we don't need the data view?
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, -1, DATA_W, DATA_H)[:n]])
                save_image(comparison.data.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)
    test_loss /= len(test_loader.dataset)
    print('====> Epoch: {} Test set loss: {:.17f}'.format(epoch, test_loss))
    return test_loss

if args.disable_train:
    args.start_epoch = 1
    args.epochs = 0

# This could/should be a dictionary
best_models = [("", 100000000000)]*3
for epoch in range(args.start_epoch, args.epochs + 1):
    train(epoch)

    # 64 sets of random ZDIMS-float vectors, i.e. 64 locations / MNIST
    # digits in latent space
    sample = Variable(torch.randn(64, args.z_dims))
    if args.cuda:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()

    # Write out data and print loss
    if epoch % args.test_interval == 0:
        test_loss = test(epoch)

        new_file = 'models/vae-%s.pt' % (epoch)
        max_idx, max_loss = max(enumerate(best_models), key = lambda x : x[1][1])
        max_loss = max_loss[1]
        if test_loss < max_loss:
            worse_model = best_models[max_idx][0]
            if not '' in [m[0] for m in best_models]: 
                os.remove(worse_model)
            best_models[max_idx] = (new_file, test_loss)

        # Save model and delete older versions
        old_file = "models/vae-%s.pt" % (epoch - 2*args.test_interval)
        found_best = old_file in [m[0] for m in best_models]
        if os.path.isfile(old_file) and not found_best:
            os.remove(old_file)
        torch.save(model.state_dict(), new_file)

        # save out as an 8x8 matrix of MNIST digits
        # this will give you a visual idea of how well latent space can generate things
        # that look like digits
        # the -1 is decide "row"/dim_size  yourself, so could be 3 or 1 depended on datasize
        # Numpy order has color channel last
        save_image(sample.data.view(64, -1, DATA_H, DATA_W),
               'results/sample_' + str(epoch) + '.png')