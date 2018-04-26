import torch
from torch import nn, optim
import SOSDataset
import conv_vae_pytorch as vae_pytorch

model = vae_pytorch.model
# toggle model to test / inference mode
model.eval()

Z_DIMS = vae_pytorch.args.z_dims # input size
FC1_SIZE = 1024 # try some different values as well
FC2_SIZE = 256

if not torch.cuda.is_available():
    print("Cuda unavailable!")
    exit(0)

torch.cuda.manual_seed(1)
kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = vae_pytorch.train_loader
test_loader = vae_pytorch.test_loader

class Classifier(nn.Module):
    
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(Z_DIMS, FC1_SIZE),
            nn.ReLU(),
            nn.BatchNorm1d(FC1_SIZE)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(FC1_SIZE, FC2_SIZE),
            nn.ReLU(),
            nn.BatchNorm1d(FC2_SIZE)
        )
        self.fc3 = nn.Linear(FC2_SIZE, 5) # output 5 labels
        self.sigmoid = nn.Sigmoid()

    # Input: z activation of an image
    def forward(self, x): 
        x = self.fc1(x)
        x = self.fc2(x)
        return self.sigmoid(self.fc3(x))

classifier = Classifier()
classifier.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

# Need to call this here again
model.cuda()
# if not training the VAE will select the zs with highest probability
model.train = False

def train(epoch):
    classifier.train()
    running_loss = 0.0
    # Test set is fairly small, also consider training on a larger set
    for i, (ims, labels) in enumerate(test_loader, 1): # unseen data
        # convert ims to z vector
        # You should reparameterize these z's, and make sure to set the model in testing/evalution mode when
        # sampling with model.reparameterize(zs), as that will draw zs with the highest means
        # I guess the output will be a batch of z vectors with Z_DIM
        mu, logvar = model.encode(ims.cuda())
        zs = model.reparameterize(mu, logvar)
        
        # print(zs)
        # print(zs.shape)
        # exit(0)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = classifier(zs)
        # target ("labels") should be 1D
        labels = labels.long().cuda().view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 15 == 0:
            print('[Epoch %d, it.: %5d] loss: %.17f' %
                  (epoch + 1, i + 1, running_loss / 2000)) # average by datasize?

def test(epoch):
    classifier.eval()
    # How well does the classifier (that now has seen the test data) perform on unseen data?
    # i.e. the train data?
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (ims, labels) in enumerate(train_loader): # unseen data
            mu, logvar = model.encode(ims.cuda())
            zs = model.reparameterize(mu, logvar)
            outputs = classifier(zs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            labels = labels.long().cuda().view(-1)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Epoch %d -> Test Accuracy: %d %%' % (epoch+1, accuracy))
    return accuracy


best_models = [("", 100000000000)]*3
for epoch in range(1, 1501):
    train(epoch)

    if epoch % 10 == 0:
        test_acc = test(epoch)

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
        old_file = "classifier-models/vae-%s.pt" % (epoch - 2*args.test_interval)
        found_best = old_file in [m[0] for m in best_models]
        if os.path.isfile(old_file) and not found_best:
            os.remove(old_file)
        torch.save(model.state_dict(), new_file)
