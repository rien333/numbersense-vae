import os, argparse
from math import ceil, floor
import random
import numpy as np
# import ColoredMNIST
# import CelebDataset
import SynDataset
import SOSDataset
import HybridEqualDataset
import torch
import torch.utils.data
import torchvision.models as models
from torch.optim import lr_scheduler
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='Annotated PyTorch VAE with conv layers')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--tune-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training with the synthetic dataset')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train with hybrid data')
parser.add_argument('--tune-epochs', type=int, default=280, metavar='N',
                    help='number of epochs to train on real data')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--z-dims', type=int, default=20, metavar='N',
                    help='dimenstionality of the latent z variable')
parser.add_argument('--alpha', type=float, default=1.0, metavar='N', 
                    help='Weight of KDL loss')
parser.add_argument('--beta', type=float, default=0.25, metavar='N', 
                    help='Weight of content loss')
parser.add_argument('--dfc', action='store_true', default=False, help="Train with deep feature consistency loss")
parser.add_argument('--full-con-size', type=int, default=400, metavar='N',
                    help='size of the fully connected layer')
parser.add_argument('--load-model', type=str, default='', metavar='P',
                    help='load a torch model from given path (default: create new model)')
parser.add_argument('--start-epoch', type=int, default=1, metavar='N',
                    help='epoch to start at (only affects logging)')
parser.add_argument('--test-interval', type=int, default=10, metavar='N',
                    help='when to run a test epoch')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Width and height of a sample from the dataset
# SOS DATAset size was changed! bc of the whole vgg/dfc thing
# DATA_W = 64
# DATA_H = 64
# DATA_C = 3

DATA_W = SOSDataset.DATA_W
DATA_H = SOSDataset.DATA_H
DATA_C = SOSDataset.DATA_C # Color component dimension size

# DATA_W = CelebDataset.DATA_W
# DATA_H = CelebDataset.DATA_H
# DATA_C = CelebDataset.DATA_C # Color component dimension size

DATA_SIZE = DATA_W * DATA_H * DATA_C

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if args.dfc:
    scale = ceil(1.137 * DATA_H) # SOS uses a 1.137 factor between cropped and original size
    # There is a normalizeMEANVGG that seem handy
    # Okay the vgg norm is now done in the net so that the original data is not as screwed up
    # data is originally 64x64, so try smaller sizes?
    # also not sure if it's between 0-1 and one per se, but maybe 0-255
    data_transform = [SOSDataset.Rescale((scale, scale)), SOSDataset.RandomCrop((DATA_W, DATA_H)),
                      SOSDataset.RandomColorShift(), SOSDataset.RandHorizontalFlip(), 
                      SOSDataset.ToTensor(),]
    syn_data_transform = list(data_transform)
    # celeb_transform = [CelebDataset.Rescale((DATA_H, DATA_W)), CelebDataset.RandomColorShift(), 
    #                    CelebDataset.RandHorizontalFlip(),  CelebDataset.ToTensor(), CelebDataset.NormalizeMean(),
    #                    CelebDataset.Normalize01()]
    # # syn_data_transform = data_transform[1:]
else:
    # ToTensor already puts everything in range 0-1
    data_transform = [SOSDataset.Rescale((256, 256)), SOSDataset.RandomCrop((DATA_W, DATA_H)),
                      SOSDataset.RandomColorShift(), SOSDataset.RandHorizontalFlip(), 
                      SOSDataset.ToTensor(), SOSDataset.NormalizeMean(), 
                      SOSDataset.Normalize01()]
    # Rescaling is not needed for synthetic data
    syn_data_transform = data_transform[1:]

with open("/etc/hostname",'r') as f:
    hostname = f.read().lower()
    lisa_check = "lisa" in hostname

if lisa_check:
    import os
    scratchdir = os.environ["TMPDIR"]
    DATA_DIR = scratchdir + "/"
    # SAVE_DIR = scratchdir + "/"
    SAVE_DIR = ""
    SORT_DIR = scratchdir + "/"
else:
    DATA_DIR = "../Datasets/"
    SAVE_DIR = "" # assume working directory
    SORT_DIR = "/tmp/"

if lisa_check and not args.cuda:
    DATA_DIR = "../Datasets/"
    SAVE_DIR = "" # assume working directory
    SORT_DIR = ""

if lisa_check:
    ngpu = torch.cuda.device_count()
elif "quva" in hostname:
    ngpu = torch.cuda.device_count()
else:
    ngpu = 1

# celeb_train_loader = torch.utils.data.DataLoader(
#     CelebDataset.CelebDataset(train=True, transform=celeb_transform,),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

# celeb_test_loader = torch.utils.data.DataLoader(
#     CelebDataset.CelebDataset(train=False, transform=celeb_transform,),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

# syn_train_loader = torch.utils.data.DataLoader(
#     SynDataset.SynDataset(train=True, transform=syn_data_transform, datadir=DATA_DIR),
#     batch_size=args.syn_batch_size, shuffle=True, **kwargs)

# syn_test_loader = torch.utils.data.DataLoader(
#     SynDataset.SynDataset(train=False, transform=syn_data_transform, datadir=DATA_DIR),
#     batch_size=args.syn_batch_size, shuffle=True, **kwargs)

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
                out_channels=64,        # output depth
                kernel_size=4,
                stride=1,
                padding=1
            ),				# out (64, DATA_H, DATA_W) should be same HxW as in
            nn.LeakyReLU(0.01),          # inplace=True saves memory but discouraged (worth the try)
            nn.BatchNorm2d(64),         # C channel input, 4d input (NxCxHxW)
            nn.Conv2d(
                in_channels=64,		# RGB
                out_channels=64,        # output depth
                kernel_size=4,
                stride=2,
                padding=1
            ),				# out (64, DATA_H, DATA_W) should be same HxW as in
            nn.LeakyReLU(0.01),          # inplace=True saves memory but discouraged (worth the try)
            nn.BatchNorm2d(64),         # C channel input, 4d input (NxCxHxW)
        )

        # These two in the middle can maybe downsample with a conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.01),		# Slight negative slope
            nn.BatchNorm2d(128),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(256),
        )

        self.conv4 = nn.Sequential(	# DATA_W/H is ~= 28
            nn.Conv2d(256, 768, 4, 2, 1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(768),
        )
        end_elems = floor(DATA_H / 16) # 16 = 2**4 downsample,
        end_shape = (end_elems**2) * 768 # eg 256*13*13 conv shape
        # conv4/conv-out should be flattened
        # fc1 conv depth * (DATA_W*DATA_H / (number of pools * 2)) (with some rounding)
        # self.fc1 = nn.Sequential(
        #     nn.Linear(end_shape, args.full_con_size),
        #     nn.LeakyReLU(0.01),
        #     nn.BatchNorm1d(args.full_con_size)
        # )

        self.fc21 = nn.Sequential(  # mean network
            # nn.Linear(args.full_con_size, args.z_dims),
            nn.Linear(end_shape, args.z_dims),
            # nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.BatchNorm1d(args.z_dims)  # This doesn't seem okay at all
        )

        # self.fc22 = nn.Linear(args.full_con_size, args.z_dims) # variance network, linear
        self.fc22 = nn.Sequential(  # variance network, linear
            # nn.Linear(args.full_con_size, args.z_dims),
            nn.Linear(end_shape, args.z_dims),
            # nn.ReLU(),
            # nn.BatchNorm1d(args.z_dims), # This doesn't seem okay at all
            # nn.ReLU(), # Gaussian std must be positive # don't think this works here
            nn.Softplus()
        )
        
        # self.fc3 = nn.Sequential(
        #     nn.Linear(args.z_dims, args.full_con_size),
        #     nn.LeakyReLU(0.01),
        #     nn.BatchNorm1d(args.full_con_size)
        # )

        self.deconv_shape = (768, end_elems+1, end_elems+1)
        # form the decoder output to a conv shape
        # should be the size of a convolution/the last conv size
        # 128*14*14 * a few (4) upsampling = the original input size
        # self.fc4 = nn.Linear(args.full_con_size, 128*15*14)
        self.fc4 = nn.Sequential(
            # nn.Linear(args.full_con_size, int(np.prod(self.deconv_shape))),
            nn.Linear(args.z_dims, int(np.prod(self.deconv_shape))),
            # nn.LeakyReLU(0.01), # Some people use normal relu here
            nn.ReLU(), # Some people use normal relu here
            nn.BatchNorm1d(int(np.prod(self.deconv_shape))) # unneeded? 
        )

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
        # so consider uncommenting the first deconv as well
        self.t_conv1 = nn.Sequential(
            nn.ConvTranspose2d(768, 768, 3, 1, 1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(768),
            nn.ConvTranspose2d(768, 256, 3, 2, 1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(256),
        )

        self.t_conv2 = nn.Sequential( # this used to be p different (or the one below idk)
            nn.ConvTranspose2d(256, 128, 3, 2, 1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(128),
        )

        self.t_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )

        self.t_conv_final = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 3, 2, 1), # RGB, no batch norm. on output
            nn.Sigmoid()  # output between 0 and 1 # Relu?
        )

        self.freeze_layers = [self.conv1, self.conv2,]

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
        # h1 = self.fc1(flatten_c4)
        # add a small epsilon for numerical stability?
        # return self.fc21(h1), self.fc22(h1) + 1e-6
        return self.fc21(flatten_c4), self.fc22(flatten_c4) + 1e-6

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
            std = torch.exp(0.5*logvar)
            # - std.data is the [128,ZDIMS] tensor that is wrapped by std
            # - so eps is [128,ZDIMS] with all elements drawn from a mean 0
            #   and stddev 1 normal distribution that is 128 samples
            #   of random ZDIMS-float vectors

            # eps = Variable(std.data.new(std.size()).normal_())
            # Updated
            eps = torch.randn_like(std)

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
        # h3 = self.fc3(z)
        # another layer that maps h3 to a conv shape
        # h4 = self.fc4(h3)
        h4 = self.fc4(z)
        h4_expanded = h4.view(-1, *self.deconv_shape) # 15 * (4 * 2x upsamling conv) ~= 225
        up_conv1 = self.t_conv1(h4_expanded)
        up_conv2 = self.t_conv2(up_conv1) # every layer upsamples by 2 basically
        up_conv3 = self.t_conv3(up_conv2)
        return self.t_conv_final(up_conv3)

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        # mu, logvar = self.encode(x.view(-1, DATA_SIZE))
        # mu and logvar are the paramters of the z distribution (after the reparameterization "trick")
        mu, logvar = self.encode(x) # conv layers work on RGB channels
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Use special vgg normalisation layer as to not make the input weird
class ImageNet_Norm_Layer_2(nn.Module):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(ImageNet_Norm_Layer_2, self).__init__()
        dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        self.mean = Variable(torch.FloatTensor(mean).type(dtype), requires_grad=0)
        self.std = Variable(torch.FloatTensor(std).type(dtype), requires_grad=0)

    def forward(self, input):
        return ((input.permute(0, 2, 3, 1) - self.mean) / self.std).permute(0, 3, 1, 2)

class _VGG(nn.Module):
    '''
    Classic pre-trained VGG19 model.
    Its forward call returns a list of the activations from
    the predefined content layers. Used for computing dfc loss.
    '''

    def __init__(self):
        super(_VGG, self).__init__()

        features = models.vgg19(pretrained=True).features
        self.norm_layer = ImageNet_Norm_Layer_2() # norm done in net to net screw the input

        # ngpu = torch.cuda.device_count()
        # self.ngpu = ngpu if not lisa_check else 1  # too much mem # assign to ngpu
        self.ngpu = ngpu
        if self.ngpu > 1:
            # Functional equivalent of below (idkkk if this is problematic? maybe it's good)
            self.gpu_func = lambda module, output: nn.parallel.data_parallel(module, output, range(self.ngpu))
            # Norm layer errors on multiple GPUs :( 
        else:
            self.gpu_func= lambda module, output: module(output)

        self.layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
               'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
               'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
        # Add one?
        content_layers = ['relu1_1', 'relu2_1', 'relu3_1',]
        self.content_layers = list(content_layers)

        self.features = nn.Sequential()
        for i, module in enumerate(features):
            name = self.layer_names[i]
            self.features.add_module(name, module)            
            if name in content_layers:
                content_layers.remove(name)
            if not content_layers:
                # Stop adding stuff
                break

    def forward(self, x):
        batch_size = x.size(0) # needed because sometimes the batch size is not exactly args.batch_size
        all_outputs = []
        x = self.norm_layer(x)
        output = x
        for name, module in self.features.named_children():
            # Forward output through one module of vgg, with or without multiple gpus
            output = self.gpu_func(module, output)
            if name in self.content_layers:
                all_outputs.append(output.view(batch_size, -1))
        return all_outputs

# this has to be a trainable module for some reason
# maybe put in the loss function
class Content_Loss(nn.Module):

    def __init__(self, alpha=1, beta=0.5):
        super(Content_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.criterion = nn.MSELoss(size_average=False)

    def forward(self, output, target, mean, logvar):
        # people use sum here instead of mean (dfc authors/versus standard pytorch sum implementation) 🌸
        # sum seems to have weird graphical glitches?
        kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) 
        # Note  detach for target
        loss_list = [self.criterion(output[layer], target[layer]) for layer in range(len(output))]
        content = sum(loss_list)
        return self.alpha * kld + self.beta * content

# Initialize layers with He initialisation
# Consider not initializing the first conv layer like Tom's code
def weights_init(m):
    classname = m.__class__.__name__
    if (isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)
        or isinstance(m, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight.data) # This is He initialization
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, std=0.015) # Small std, maybe to small?
        m.bias.data.zero_()
    # elif isinstance(m, nn.Linear): # done above
    #     nn.init.xavier_uniform(m.weight.data)
    #     m.bias.data.zero_()

model = CONV_VAE()
model.apply(weights_init)

if args.dfc:
    # The exact style seems less relevant, but try different values
    descriptor = _VGG()
    descriptor.to(device) # descriptor has it's own parallelism thingy
    descriptor.eval()
    for param in descriptor.parameters():
        param.requires_grad = False

    content_loss = Content_Loss(alpha=args.alpha, beta=args.beta)
    content_loss.to(device)

if  ngpu > 1:
    print("Using", ngpu, "GPUs!")
    model = nn.DataParallel(model)
else:
    print("Using one gpu.")
model.to(device)

if args.load_model:
    try:
        model.load_state_dict(
            torch.load(args.load_model, map_location=lambda storage, loc: storage))
    except RuntimeError as e: # trying to load a multi gpu model to a single gpu
        if "module" in str(e):
            print("Oops, converting multi gpu model to single gpu...")
            from collections import OrderedDict
            state_dict = torch.load(args.load_model, map_location=lambda storage, loc: storage)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.'
                new_state_dict[name] = v
            # load params'
            model.load_state_dict(new_state_dict)
        else:
            print(e)
            exit(1)

def loss_function_van(recon_x, x, mu, logvar):
    # how well do input x and output recon_x agree?
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    # logvar = torch.log(logvar)

    # kld is Kullback–Leibler divergence -- how much does one learned
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

    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    return BCE + KLD

def loss_function_dfc(recon_x, x, mu, logvar):
    # loss is KLD + percetupal reconstruction loss between the convs layers
    targets = descriptor(x) # vgg
    recon_features = descriptor(recon_x)
    # BCE = F.binary_cross_entropy(recon_x, x)
    # return content_loss(recon_features, targets, mu, logvar) + (20000000*BCE)
    return content_loss(recon_features, targets, mu, logvar)

# REMOVE ❗
def loss_function_dfc_split(recon_x, x, mu, logvar):
    # loss is KLD + percetupal reconstruction loss between the convs layers
    targets = descriptor(x) # vgg
    recon_features = descriptor(recon_x)
    # BCE = F.binary_cross_entropy(recon_x, x)
    # Note the mean versus sum thing also mentioned above 🌸
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # note the detach
    loss_list = [F.mse_loss(recon_features[layer], targets[layer], size_average=False) for layer in range(len(targets))]
    content = sum(loss_list)
    # return args.alpha*kld, args.beta*content,
    return kld, content # don't multiply to keep the loss in the same range at all times


# Check for dfc loss
if args.dfc:
    loss_function = loss_function_dfc
else:
    loss_function = loss_function_van


def train(epoch, loader, optimizer):
    model.train()
    # the enum thingy can be removed I guess
    for data, _ in loader:
    # for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, logvar = model(data)
        # calculate scalar loss
        loss = loss_function(recon_batch, data, mu, logvar)
        # calculate the gradient of the loss w.r.t. the graph leaves
        # i.e. input variables -- by the power of pytorch!
        loss.backward()
        optimizer.step()

def test(epoch, loader):
    model.eval()
    test_loss = 0
    kld_loss, content_loss  = 0, 0,
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
        # for i, data in enumerate(loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            # test_loss += loss_function_dfc(recon_batch, data, mu, logvar).item()
            kld, content = [l.item() for l in loss_function_dfc_split(recon_batch, data, mu, logvar)]
            kld_loss += kld
            content_loss += content
            # bce_loss += bce
            test_loss += kld + content
            if i == 0:
                n = min(data.size(0), 7)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(loader.batch_size, -1, DATA_W, DATA_H)[:n]])
                save_image(comparison,
                           SAVE_DIR + 'results/reconstruction_' + str(epoch) + '.png', nrow=n)

        # ~50 sets of random ZDIMS-float vectors to images
        # Weird hack bc this is drawn from ~ N(0, 1), and our distribution looks different
        sample = torch.randn(49, args.z_dims).to(device) * 5.2

        if ngpu > 1:
            sample = model.module.decode(sample)
        else:
            sample = model.decode(sample)
        # this will give you a visual idea of how well latent space can generate new things
        save_image(sample.data.view(49, -1, DATA_H, DATA_W),
               SAVE_DIR + 'results/sample_' + str(epoch) + '.png', nrow=n)
        # print(torch.unique(sample.cpu(), sorted=True))

    test_loss /= len(loader.dataset)
    print('====> Epoch: {} Test set loss: {:.10f} Content loss: {:.4f} KLD loss: {:.4f}'.format(epoch, test_loss, content_loss/len(loader.dataset), kld_loss/len(loader.dataset)))
    return test_loss

def train_routine(epochs, train_loader, test_loader, optimizer, scheduler, reset=120, start_epoch=0):
    # This could/should be a dictionary
    best_models = [("", 100000000000)]*5

    for epoch in range(start_epoch, epochs + 1):
        train(epoch, train_loader, optimizer)

        # Write out data and print loss
        if epoch % args.test_interval == 0:
            test_loss = test(epoch, test_loader)
            scheduler.step(test_loss)

            new_file = SAVE_DIR + 'models/vae-%s.pt' % (epoch)
            max_idx, max_loss = max(enumerate(best_models), key = lambda x : x[1][1])
            max_loss = max_loss[1]
            if test_loss < max_loss:
                worse_model = best_models[max_idx][0]
                if not '' in [m[0] for m in best_models]: 
                    os.remove(worse_model)
                best_models[max_idx] = (new_file, test_loss)

            # Save model and delete older versions
            old_file = SAVE_DIR + "models/vae-%s.pt" % (epoch - 2*args.test_interval)
            found_best = old_file in [m[0] for m in best_models]
            if os.path.isfile(old_file) and not found_best:
                os.remove(old_file)
            torch.save(model.state_dict(), new_file)
        
        if ((epoch - start_epoch) % reset == 0) and (epoch != start_epoch):
            print("Resetting learning rate")
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.33, patience=4, cooldown=1, 
                                                       verbose=True)

if __name__ == "__main__":

    grow_f=6.2952 # Lisa size
    # grow_f=6.2952/4 # Lisa size
    # grow_f=3.5032
    hybrid_train_loader = torch.utils.data.DataLoader(
        HybridEqualDataset.HybridEqualDataset(epochs=args.epochs-6, train=True, transform=data_transform, 
                                              t=0.605,grow_f=grow_f, datadir=DATA_DIR, sorted_loc=SORT_DIR),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    hybrid_test_loader = torch.utils.data.DataLoader(
        HybridEqualDataset.HybridEqualDataset(epochs=args.epochs-6, train=False, transform=data_transform, 
                                              t=0.605,grow_f=2.0, datadir=DATA_DIR, sorted_loc=SORT_DIR),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # # optimizer = optim.Adam(model.parameters(), lr=1e-3) # = 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.00135)
    # Decay lr if nothing happens after 4 epochs (try 3?)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.23, patience=4, cooldown=1, 
                                               verbose=True)

    train_routine(args.epochs, train_loader=hybrid_train_loader, test_loader=hybrid_test_loader, 
                  optimizer=optimizer, scheduler=scheduler, reset=102)

    # Freeze early layers
    model_access = model.module if ngpu > 1 else model
    for l in model.modules():
        if l in model_access.freeze_layers:
            for p in l.parameters():
                p.requires_grad = False

    hybrid_train_loader = None

    # Fine tune on real data
    SOS_train_loader = torch.utils.data.DataLoader(
        SOSDataset.SOSDataset(train=True, transform=data_transform, extended=True, datadir=DATA_DIR),
        batch_size=args.tune_batch_size, shuffle=True, **kwargs)
    SOS_test_loader = torch.utils.data.DataLoader(
        SOSDataset.SOSDataset(train=False, transform=data_transform, extended=True, datadir=DATA_DIR),
        batch_size=args.tune_batch_size, shuffle=True, **kwargs)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.33, patience=4, cooldown=2, 
                                               verbose=True)

    train_routine(args.tune_epochs, train_loader=SOS_train_loader, test_loader=SOS_test_loader, 
                  start_epoch=args.epochs, optimizer=optimizer, scheduler=scheduler, 
                  reset=100)
