import os
import torch
from torch import nn, optim
import SOSDataset
import SynDataset
import HybridEqualDataset
import conv_vae_pytorch as vae_pytorch
from torch.nn import functional as F
import numpy as np
from torch.optim import lr_scheduler
import random

Z_DIMS = vae_pytorch.args.z_dims # input size
FC1_SIZE = 150 # try some different values as well
FC2_SIZE = 140 # To small to support all outputs?

class Classifier(nn.Module):
    
    # Ordening https://stackoverflow.com/a/40295999/1657933
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(Z_DIMS, FC1_SIZE),
            nn.ReLU(), # The original code had functional relu's
            nn.BatchNorm1d(FC1_SIZE),
            nn.Dropout(0.5), # probability of a zeroing out an element of the input to prevent co-adaptation
        )
        self.fc2 = nn.Sequential(
            nn.Linear(FC1_SIZE, FC2_SIZE),
            nn.ReLU(),
            nn.BatchNorm1d(FC2_SIZE),
            nn.Dropout(0.6),
        )
        self.fc3 = nn.Linear(FC2_SIZE, 5) # output 5 labels
        # self.sigmoid = nn.Sigmoid()
        # I think dim=1 is the default for 2d tensor, but explictly don't 
        # self.softmax = nn.Softmax(dim=1) # Generalized sigmoid over n dimensions

    # Input: z activation of an image
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        # The Crossentropyloss function already incoperates softmax apparently
        return self.fc3(x) # try softmax?

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
    running_loss = 0
    for ims, labels in loader: # unseen data
        optimizer.zero_grad()
        mu, logvar = model.encode(ims.cuda())
        zs = model.reparameterize(mu, logvar)
        outputs = classifier(zs)
        # target ("labels") should be 1D
        labels = labels.long().cuda().view(-1)  # Might need .cuda
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print('[Epoch %d]  Train set loss: %.17f' %
          (epoch, running_loss / len(loader))) # average by datasize?

def test(epoch, loader, criterion):
    classifier.eval()
    # How well does the classifier (that now has seen the test data) perform on unseen data?
    # i.e. the train data?
    correct = 0
    total = 0
    classes = list(range(5))
    class_correct = list(0. for i in range(5))
    class_total = list(0.0000000001 for i in range(5))
    running_loss = 0
    with torch.no_grad():
        for i, (ims, labels) in enumerate(loader): # unseen data
            mu, logvar = model.encode(ims.cuda())
            zs = model.reparameterize(mu, logvar)
            outputs = classifier(zs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            labels = labels.long().cuda().view(-1)
            loss = criterion(outputs, labels)
            correct += (predicted == labels).sum().item() # calculate mean accuracy
            running_loss += loss.item()
            c = (predicted == labels).squeeze()
            # Calculate correct instances per class
            for i in range(labels.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Epoch %d -> Test set loss: %.17f ' % (epoch, running_loss/len(loader)))
    
    accuracy = 100 * correct / total
    print('Mean Accuracy %3s : %2d %%' % ("", accuracy))
    for i in range(5):
        print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    class_scores = np.array(class_correct) / np.array(class_total)
    return running_loss, class_scores

def train_routine(epochs, train_loader, test_loader, optimizer, criterion, scheduler, start_epoch=0,):
    best_models = [("", -100000000000)]*4
    test_interval = vae_pytorch.args.test_interval
    chance_scores = np.array([0.275, 0.465, 0.186, 0.117, 0.097]) * 1.65
    # Save models according to loss instead of acc?
    for epoch in range(start_epoch+1, start_epoch+epochs):
        train(epoch, train_loader, optimizer, criterion)

        if epoch % test_interval == 0:
            test_loss, class_scores = test(epoch, test_loader, criterion)
            # Calculate the loss per class weight by their relative importance?
            # so use the class scores I guess
            scheduler.step(test_loss)
            # Save best performing models
            new_file = 'classifier-models/vae-%s.pt' % (epoch)
            max_idx, max_loss = max(enumerate(best_models), key = lambda x : x[1][1])
            max_loss = max_loss[1]
            if test_loss < max_loss:
                worse_model = best_models[max_idx][0]
                if not '' in [m[0] for m in best_models]: 
                    os.remove(worse_model)
                best_models[max_idx] = (new_file, test_loss)

            # Save model and delete older versions
            old_file = "classifier-models/vae-%s.pt" % (epoch - 2 *  test_interval)
            found_best = old_file in [m[0] for m in best_models]
            if os.path.isfile(old_file) and not found_best:
                os.remove(old_file)
            torch.save(classifier.state_dict(), new_file)
   
            # shift_alpha = random.uniform(0.009, 0.016)
            shift_alpha = 0
            updt_idxs = (class_scores < chance_scores) * shift_alpha
            print(updt_idxs)
            weights = criterion.weight.cpu().numpy() + updt_idxs
            print("new weights", weights)
            # normalize
            # criterion.weight = torch.cuda.FloatTensor(weights / np.sum(weights))
            

if __name__ == "__main__":
    
    scale = vae_pytorch.scale
    DATA_W = SOSDataset.DATA_W
    DATA_H = SOSDataset.DATA_H
    DATA_C = SOSDataset.DATA_C # Color component dimension size
    DATA_DIR = vae_pytorch.DATA_DIR

    kwargs = {'num_workers': 1, 'pin_memory': True} if vae_pytorch.args.cuda else {}
    # 🐝🐝🐝 Readd old transforms
    data_transform = [SOSDataset.Rescale((DATA_W, DATA_H)),
                      SOSDataset.ToTensor(),]

    # class weights with imbalance ratio
    # these work okay in that they can lead to a model that performs above chance for all classes
    # and scores comparable to baseline methods
    # class_weights = torch.cuda.FloatTensor([0.763, 0.557, 0.854, 0.903, 0.922]) ** 4
    # class_weights = torch.cuda.FloatTensor([1, 1, 1, 1, 1,])
    # class_weights = torch.cuda.FloatTensor([0.19, 0.495, 0.951, 1.0, 0.94])
    # real_samples = [2597, 4854, 1604, 1058, 853]
    # syn_samples = [2596, 4853, 1604, 1058, 853] # should equal 5
    real_samples = np.array([2596, 4854, 1604, 1058, 853]) # undersample ❗
    # real_samples = np.array([853] * 5)
    # syn_samples = np.array([500] * 5)
    # syn_samples = np.array([2000, 4000, 1604, 1058, 853]) * 1
    syn_samples = np.array([0, 0, 0, 220, 220]) * 1
    total_samples = real_samples + syn_samples
    n_samples = np.sum(total_samples)
    # class_weights = torch.cuda.FloatTensor(1-(total_samples/n_samples))**3.5
    # class_weights = torche.cuda.FloatTensor([1]*5)
    class_weights = torch.cuda.FloatTensor([0.1, 0.2, 0.4, 0.5, 0.45])
    print("Weights", class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights, size_average=False)
    optimizer = optim.SGD(classifier.parameters(), lr=0.0005, momentum=0.9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.42, patience=4, cooldown=1, 
                                               verbose=True)
    grow_f=4.5006
    hybrid_train_loader = torch.utils.data.DataLoader(
        HybridEqualDataset.HybridEqualDataset(epochs=30-5, train=True, t=1.1, transform=data_transform,
                                              grow_f=grow_f, datadir=DATA_DIR, real_samples=real_samples,
                                              syn_samples=syn_samples),
        batch_size=vae_pytorch.args.batch_size, shuffle=True, **kwargs)

    SOS_test_loader = torch.utils.data.DataLoader(
        SOSDataset.SOSDataset(train=False, transform=data_transform, extended=True, datadir=DATA_DIR),
        batch_size=vae_pytorch.args.batch_size, shuffle=True, **kwargs)

    # Evalaute on synthetic data first when fine tuning
    train_routine(vae_pytorch.args.epochs, train_loader=hybrid_train_loader, test_loader=SOS_test_loader, 
                  optimizer=optimizer, criterion=criterion, scheduler=scheduler)

    # classifier.load_state_dict(
    #     torch.load("classifier-models/vae-60.pt", map_location=lambda storage, loc: storage))
    # classifier.eval()

    for p in classifier.fc1.parameters():
        p.requires_grad = False
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, classifier.parameters()),  lr=0.001, momentum=0.85)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.33, patience=4, cooldown=2, 
                                               verbose=True)
    # grow_f=0.22 # How small can you make this?
    # balance classes a little
        # Fine tune on real data
    SOS_train_loader = torch.utils.data.DataLoader(
        SOSDataset.SOSDataset(train=True, transform=data_transform, extended=True, datadir=DATA_DIR),
        batch_size=vae_pytorch.args.tune_batch_size, shuffle=True, **kwargs)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_routine(vae_pytorch.args.tune_epochs, start_epoch=vae_pytorch.args.epochs, train_loader=SOS_train_loader, 
                  test_loader=SOS_test_loader, optimizer=optimizer, criterion=criterion, scheduler=scheduler)


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
