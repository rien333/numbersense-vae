import os
import torch
from torch import nn, optim
import SOSDataset
import SynDataset
import HybridEqualDataset
import conv_vae_pytorch as vae_pytorch

Z_DIMS = vae_pytorch.args.z_dims # input size
FC1_SIZE = 276 # try some different values as well
FC2_SIZE = 250 # To small to support all outputs?

class Classifier(nn.Module):
    
    def __init__(self):
        super(Classifier, self).__init__()
        # Try dropout for classification with pre training with synthetic data
        self.fc1 = nn.Sequential(
            nn.Linear(Z_DIMS, FC1_SIZE),
            nn.Dropout(),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(FC1_SIZE),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(FC1_SIZE, FC2_SIZE),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(FC2_SIZE),
        )
        self.fc3 = nn.Linear(FC2_SIZE, 5) # output 5 labels
        self.sigmoid = nn.Sigmoid()

    # Input: z activation of an image
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.sigmoid(self.fc3(x))

classifier = Classifier()
# print("Classifier: loading model 56")
# classifier.load_state_dict(
#     torch.load("classifier-models/vae-56.pt", map_location=lambda storage, loc: storage))


model = vae_pytorch.model
# toggle model to test / inference mode
model.eval()
# if not training the VAE will select the zs with highest probability
model.training = False

if vae_pytorch.args.cuda:
    classifier.cuda()
    model.cuda() # need to call this here again 

def train(epoch, loader, optimizer, criterion):
    classifier.train()
    # running_loss = 0.0
    # Test set is fairly small, also consider training on a larger set
    for i, (ims, labels) in enumerate(loader, 1): # unseen data
        # convert ims to z vector
        # You should reparameterize these z's, and make sure to set the model in testing/evalution mode when
        # sampling with model.reparameterize(zs), as that will draw zs with the highest means
        # I guess the output will be a batch of z vectors with Z_DIM

        optimizer.zero_grad()
        # Assume more than one gpu!!!
        mu, logvar = model.module.encode(ims.cuda()) # Might need .cuda
        zs = model.module.reparameterize(mu, logvar)
        
        # zero the parameter gradients
        # forward + backward + optimize
        outputs = classifier(zs)
        # target ("labels") should be 1D
        labels = labels.long().cuda().view(-1)  # Might need .cuda
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        # if i % 15 == 0:
        #     print('[Epoch %d, it.: %5d] loss: %.17f' %
                  # (epoch + 1, i + 1, running_loss / 2000)) # average by datasize?

def test(epoch, loader):
    classifier.eval()
    # How well does the classifier (that now has seen the test data) perform on unseen data?
    # i.e. the train data?
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (ims, labels) in enumerate(loader): # unseen data
            mu, logvar = model.module.encode(ims.cuda())
            zs = model.module.reparameterize(mu, logvar)
            outputs = classifier(zs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            labels = labels.long().cuda().view(-1)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Epoch %d -> Test Accuracy: %d %%' % (epoch+1, accuracy))
    
    # Test per class score
    classes = list(range(5))
    class_correct = list(0. for i in range(10))
    class_total = list(0.0000000001 for i in range(10))
    with torch.no_grad():
        for im, labels in loader:
            mu, logvar = model.module.encode(im.cuda()) # Might need .cuda
            zs = model.module.reparameterize(mu, logvar)
            outputs = classifier(zs)
            labels = labels.long().cuda().view(-1) # Might need .cuda
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(5):
        print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

    return accuracy

def train_routine(epochs, train_loader, test_loader, optimizer, criterion, start_epoch=0,):
    best_models = [("", -100000000000)]*4
    test_interval = 7
    # Save models according to loss instead of acc?
    for epoch in range(start_epoch, start_epoch+epochs):
        train(epoch, train_loader, optimizer, criterion)

        if epoch % test_interval == 0:
            test_acc = test(epoch, test_loader)
            # Save best performing models
            new_file = 'classifier-models/vae-%s.pt' % (epoch)
            min_idx, min_acc = min(enumerate(best_models), key = lambda x : x[1][1])
            min_acc = min_acc[1]
            if test_acc > min_acc:
                worse_model = best_models[min_idx][0]
                if not '' in [m[0] for m in best_models]: 
                    os.remove(worse_model)
                best_models[min_idx] = (new_file, test_acc)

            # Save model and delete older versions
            old_file = "classifier-models/vae-%s.pt" % (epoch - 2 *  test_interval)
            found_best = old_file in [m[0] for m in best_models]
            if os.path.isfile(old_file) and not found_best:
                os.remove(old_file)
            torch.save(classifier.state_dict(), new_file)

if __name__ == "__main__":
    scale = vae_pytorch.scale
    DATA_W = SOSDataset.DATA_W
    DATA_H = SOSDataset.DATA_H
    DATA_C = SOSDataset.DATA_C # Color component dimension size
    DATA_DIR = vae_pytorch.DATA_DIR

    kwargs = {'num_workers': 1, 'pin_memory': True} if vae_pytorch.args.cuda else {}
    data_transform = [SOSDataset.Rescale((scale, scale)), SOSDataset.RandomCrop((DATA_W, DATA_H)),
                      SOSDataset.RandomColorShift(), SOSDataset.RandHorizontalFlip(), 
                      SOSDataset.ToTensor(), SOSDataset.NormalizeMean(), SOSDataset.Normalize01()]

    class_weights = torch.cuda.FloatTensor([0.232, 0.4, 1.0, 1.0, 0.955])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
    # syn_samples = [1500] * 5
    grow_f=3.481
    # syn_samples = [1900, 1000, 8923, 8900, 7100] # These work quite well I believe
    # real_samples = [1800, 3402, 1604, 1058, 853]
    # real_samples = [0] * 5
    # real_samples = [853] * 5
    hybrid_train_loader = torch.utils.data.DataLoader(
        HybridEqualDataset.HybridEqualDataset(epochs=30-5, train=True, t=1.1, transform=data_transform, 
                                              grow_f=grow_f, datadir=DATA_DIR,),
        batch_size=vae_pytorch.args.batch_size, shuffle=True, **kwargs)


    # syn_samples = [750] * 5 
    # real_samples = [1] * 5
    # hybrid_test_loader = torch.utils.data.DataLoader(
    #     HybridEqualDataset.HybridEqualDataset(epochs=30-5, train=False, t=0.5, transform=data_transform, 
    #                                           grow_f=1.5, datadir=DATA_DIR, syn_samples=syn_samples, 
    #     real_samples=real_samples),
    #     batch_size=vae_pytorch.args.batch_size, shuffle=True, **kwargs)

    # syn_train_loader = torch.utils.data.DataLoader(
    #     SynDataset.SynDataset(train=True, transform=data_transform,),
    #     batch_size=vae_pytorch.args.batch_size, shuffle=True, **kwargs)

    # syn_test_loader = torch.utils.data.DataLoader(
    #     SynDataset.SynDataset(train=False, transform=data_transform, ),
    #     batch_size=vae_pytorch.args.batch_size, shuffle=True, **kwargs)

    SOS_test_loader = torch.utils.data.DataLoader(
        SOSDataset.SOSDataset(train=False, transform=data_transform, extended=True, datadir=DATA_DIR),
        batch_size=vae_pytorch.args.batch_size, shuffle=True, **kwargs)

    # Evalaute on synthetic data first when fine tuning
    # train_routine(30, train_loader=syn_train_loader, test_loader=syn_test_loader, optimizer=optimizer)
    train_routine(72, train_loader=hybrid_train_loader, test_loader=SOS_test_loader, optimizer=optimizer, 
                  criterion=criterion)
    # train_routine(30, train_loader=hybrid_train_loader, test_loader=hybrid_test_loader, optimizer=optimizer)


    # Fine tune later (load in all synthetic data at first step, and then mostly sos)
    for p in classifier.fc1.parameters():
        p.requires_grad = False
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, classifier.parameters()),  lr=0.00015, momentum=0.85)

    # real_samples = [1597, 1595, 1604, 1058, 853]
    grow_f=0.22 # How small can you make this?
    # balance classes a little
        # Fine tune on real data
    SOS_train_loader = torch.utils.data.DataLoader(
        SOSDataset.SOSDataset(train=True, transform=data_transform, extended=True, datadir=DATA_DIR),
        batch_size=vae_pytorch.args.batch_size, shuffle=True, **kwargs)

    # hybrid_train_loader = torch.utils.data.DataLoader(
    #     HybridEqualDataset.HybridEqualDataset(epochs=30, train=True, t=1.1, transform=data_transform,
    #                                           grow_f=grow_f, datadir=DATA_DIR, real_samples=real_samples),
    #     batch_size=vae_pytorch.args.batch_size, shuffle=True, **kwargs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    class_weights = torch.cuda.FloatTensor([0.24, 0.65, 0.97, 1.0, 0.95]) # 1 and 0.8 are reversed
    train_routine(73, train_loader=SOS_train_loader, test_loader=SOS_test_loader, optimizer=optimizer, criterion=criterion)


# classifier.load_state_dict(
#         torch.load("classifier-models/vae-180.pt", map_location=lambda storage, loc: storage)
# classifier.eval()

# Test on which classes the model performs well

# classes = list(range(5))
# class_correct = list(0. for i in range(10))
# class_total = list(0.0000000001 for i in range(10))
# with torch.no_grad():
#     for im, labels in train_loader:
#         mu, logvar = model.encode(im) # Might need .cuda
#         zs = model.reparameterize(mu, logvar)
#         outputs = classifier(zs)
#         labels = labels.long().view(-1) # Might need .cuda
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(labels.shape[0]):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1

# for i in range(5):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))

# from torchvision import transforms
# from torchvision.utils import save_image
# import cv2
# import os
# from random import choice

# icat = lambda x: os.system("/home/rwever/local/bin/imgcat " + x)

# data_t = transforms.Compose(vae_pytorch.data_transform)

# b_dir = "../Datasets/SUN397"
# b_classes_txt = b_dir + "/ClassName.txt"
# with open(b_classes_txt, "r") as f:
#     b_classes = f.read().splitlines()

# for f in range(1000):
#     rnd_class = choice(b_classes)
#     rnd_class_p = b_dir + rnd_class + "/"
#     b_ims = choice(os.listdir(rnd_class_p))
#     fname = rnd_class_p + b_ims
#     background = cv2.imread(fname)
#     im = data_t((background, 0))[0].view(1, 3, vae_pytorch.DATA_H, vae_pytorch.DATA_W)
#     mu, logvar = model.encode(im.cuda())
#     zs = model.reparameterize(mu, logvar)
#     outputs = classifier(zs)
#     if outputs[0][0] >= 0.96:
#         icat(fname)
#         print(outputs)
