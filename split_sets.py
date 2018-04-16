import os, sys
import fnmatch
import random

# Split a folder in a training and a test set.
if len(sys.argv) > 1:
    datadir = sys.argv[1]
else:
    datadir = "/Users/rw/Desktop/Counting/Datasets/SOS"


files = fnmatch.filter(os.listdir(datadir), '*.jpg')

os.system("mkdir -p '%s/train'" % (datadir))
os.system("mkdir -p '%s/test'" % (datadir))
# Get 80% from the train set, see original SOS paper
train_idxs = set(random.sample(range(0, len(files)), int(0.8 * len(files))))
test_idxs = set(range(len(files))) - train_idxs
cwd = os.getcwd()

for idx, i in enumerate(train_idxs):
    os.system("ln -s %s/%s %s/train/%d.jpg" % (datadir, files[idx], datadir, idx))

for idx, i in enumerate(test_idxs, start=idx):
    os.system("ln -s %s/%s %s/test/%d.jpg" % (datadir, files[idx], datadir, idx))
