import os
import zclassifier
import random
import cv2
from torchvision import transforms
import numpy as np
import torch
import conv_vae_pytorch as vae_pytorch
from torchvision.utils import save_image
import SOSDataset
import random

# Generates synthetic images for the subitizing task, as described in Salient Object Subitizing (2016)

# 256 cus they need to be transformed still
DATA_W = SOSDataset.DATA_W
DATA_H = SOSDataset.DATA_H
DATA_C = SOSDataset.DATA_C
data_t = transforms.Compose([SOSDataset.Rescale((DATA_H, DATA_W)), SOSDataset.ToTensor()])
classifier = zclassifier.classifier
if vae_pytorch.args.cuda:
    zclassifier.classifier.cuda()
# some relevant settings are done in z-classifier for conv_vae
model = zclassifier.model
classifier.load_state_dict(
        torch.load("classifier-models/65-37d7.pt", map_location=lambda storage, loc: storage))
classifier.eval()

obj_types = 32
MSRA10K_dir = "../Datasets/MSRA10K_Imgs_GT/"
files_txt = MSRA10K_dir + "files_jpg.txt"
with open(files_txt, "r") as f:
    files = f.read().splitlines()
random.shuffle(files)
files=files[:obj_types]

# data_out = "../Datasets/synthetic2-small"
data_out = "../Datasets/syn-new"

# Only generate from a few background classes to reduce noise
b_types = 7
b_dir = "../Datasets/SUN397"
b_classes_txt = b_dir + "/ClassName.txt"
with open(b_classes_txt, "r") as f:
    b_classes = f.read().splitlines()
    b_classes = random.sample(b_classes, b_types)

b_classes = ["/f/forest/needleleaf/"]

# Linear definitley works better for the fitting task
mask_resizing = cv2.INTER_LINEAR
obj_resizing = cv2.INTER_LINEAR

im_ref_low = 0.4 # was 0.4
im_ref_high = 0.8 # was 0.8

im_low = 0.85 # was 0.85 
im_high = 1.15 # was 1.15

print("Generating images with size factor probability between %s and %s 🌸" % (im_low, im_high))
print("Using %s background classes for generation" % (len(b_classes)))
print("Using %s objects for generation" % (len(files)))

# Rotate and expand
def rotate(mat, angle):
  # angle in degrees

  height, width = mat.shape[:2]
  image_center = (width/2, height/2)

  rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

  abs_cos = abs(rotation_mat[0,0])
  abs_sin = abs(rotation_mat[0,1])

  bound_w = int(height * abs_sin + width * abs_cos)
  bound_h = int(height * abs_cos + width * abs_sin)

  rotation_mat[0, 2] += bound_w/2 - image_center[0]
  rotation_mat[1, 2] += bound_h/2 - image_center[1]

  rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
  return rotated_mat

# returns transform at location with corresponding mask for pasting in new objecs
# maybe could just be written as a series of list comphresions
def generate_transforms(n, ref_size, ref_mask, thresh=0.5, total_size=False):
    transforms = []
    # record amount of pixels each object takes up originally
    total_pixels = []
    label_idxs = []
    r_s = np.random.uniform(im_low, im_high, size=n)
    if total_size:
        s=np.sum(r_s)
        n_paste_sizes = np.uint32((r_s/s) * total_size)
    else:
        n_paste_sizes = np.uint32(r_s * ref_size)

    # maybe the whole could just be written as a series of list comphresions
    for nps in n_paste_sizes:
        transform = []
        resize = lambda i, s=nps: cv2.resize(i, s, obj_resizing)
        n_mask = resize(ref_mask)
        transform.append(resize)
        # Rotate
        deg = random.uniform(-10, 10)
        # Maybe use transparent pixels for the expand?
        rot = lambda i, d=deg: rotate(i, d)
        n_mask = rot(n_mask)
        transform.append(rot)

        if random.randint(0,1):
            # trans = lambda i: i.transpose(Image.FLIP_LEFT_RIGHT)
            trans = lambda i: cv2.flip(i, 1) # flip vertically
            n_mask = trans(n_mask)
            transform.append(trans)

        # Experiment with width and height start
        valid_im = False
        tries = 4 # Maybe avioid death locks like so
        # tries are handeld else where
        # consider moving the scaling inside to find the right size
        while not valid_im and tries > 0:
        # while not valid_im:
            # find something better
            valid_im = True
            try:
                ploc = (random.randint(0, DATA_H-n_mask.shape[0]), 
                        random.randint(0, DATA_W-n_mask.shape[1]))
            except:
                valid_im = False
                tries = 0
                break
            # paste the mask in a background array holding all the occupied pixels
            # and check if their area is still 50% after the other pastes
            background_arr = np.zeros((DATA_H, DATA_W), dtype=np.float)

            # After the clip? (idk if clipping is occlussion)
            # Def before bc you want objects to be somewhat visible, so take into account what
            # of their oriignal appearance is still there
            new_total_pixels = total_pixels + [np.sum(n_mask.astype(bool))]
            end_h = ploc[0]+n_mask.shape[0]
            end_w = ploc[1]+n_mask.shape[1]
            clip_h = background_arr.shape[0] - (end_h)
            clip_w = background_arr.shape[1] - (end_w)

            # Cut off m where it doesn't fit
            # Is this okay
            c_n_mask = n_mask.copy()
            if clip_w < 0:
                c_n_mask = c_n_mask[:, :clip_w]
            if clip_h < 0:
                c_n_mask = c_n_mask[:clip_h]

            background_arr[ploc[0]:end_h, ploc[1]:end_w] = c_n_mask
            n_idxs = set(np.where(background_arr.reshape(DATA_W*DATA_H))[0])
            # Make these numpy arrays and substract them from each other

            # Update indices with new indices
            new_label_idxs = [idxs - n_idxs for idxs in label_idxs]
            new_label_idxs += [n_idxs]

            # See how much remains
            for pixel_idxs, t_pixels in zip(new_label_idxs, new_total_pixels):
                if len(pixel_idxs) < thresh * t_pixels:
                    tries -= 1
                    valid_im = False
                    break
            if not valid_im:
                continue
            label_idxs = list(new_label_idxs) # copy?
            total_pixels = list(new_total_pixels)
            # print(("t "+'{:>8} '*len(total_pixels)).format(*total_pixels))
            # print(("n "+'{:>8} '*len(label_idxs)).format(*[len(labels) for labels in label_idxs]))
            # print("-"*20)

        if tries <= 0:
            return False # break out
        # else append
        transforms.append((transform, ploc, c_n_mask))
    return transforms

# Returns: Obj array and Obj filename
def single_obj():
    single_obj = False
    while not single_obj:
        # Load random object
        fname = MSRA10K_dir+random.choice(files) # files is like a global
        # Do this as a preprocessing thing?
        # Object images need to be filtered for containing only one salient object
        obj = cv2.imread(fname, 1) # RGB

        # Check for only one object
        im = data_t((cv2.cvtColor(obj, cv2.COLOR_BGR2RGB), 0))[0].view(1, DATA_C, DATA_H, DATA_W)
        mu, logvar = model.module.encode(im.cuda())
        zs = model.module.reparameterize(mu, logvar)
        outputs = classifier(zs)
        # get max index
        if torch.argmax(outputs).item() != 1 or outputs[0][1] < 0.92:
            continue
        single_obj = True
    return obj, fname
    

# Returns:  background array
def background_im():
    valid_background = False
    while not valid_background:
        rnd_class = random.choice(b_classes)
        rnd_class_p = b_dir + rnd_class + "/"
        b_ims = random.choice(os.listdir(rnd_class_p))
        background = cv2.imread(rnd_class_p + b_ims, 1) # RGB
        try:
            background = cv2.resize(background, (DATA_H, DATA_W))
        except:
            continue

        # Check if the background is valid, i.e. contains no clear salient object
        im = data_t((cv2.cvtColor(background, cv2.COLOR_BGR2RGB), 0))[0].view(1, DATA_C, DATA_H, DATA_W)
        mu, logvar = model.module.encode(im.cuda())
        zs = model.module.reparameterize(mu, logvar)
        outputs = classifier(zs)

        if outputs[0][0] < 0.96:
            continue

        valid_background = True
    return background

# Returns: scale and image array
# obj_f is a file path refering to an object/mask combo
# background is a numpy image
def generate_image(cat, thresh=0.5, obj_f=None, background=None, size=False):
    if obj_f:
        obj = cv2.imread(obj_f, 1) # RGB
    else:
        obj, obj_f = single_obj()
    mask = cv2.imread(obj_f.replace(".jpg", ".png"), 0).astype(np.float) / 255.0 # Grayscale

    if background is None:
        background = background_im() 

    background = background.astype(np.float)
    # Get bounding box of white region
    coords = np.argwhere(mask.astype(bool))
    # Bounding box of non-black pixels.
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1 # slices are exclusive at the top

    mask_crop = mask[y0:y1, x0:x1]
    obj_ref = obj[y0:y1, x0:x1]
    # size for new basis object 
    l_dim = 0 if mask_crop.shape[0] > mask_crop.shape[1] else 1

    # The basis which will be pasted and transformed further

    # Paste N e [0,4] objects
    valid_t = False if cat > 0 else True
    transforms = []
    while not valid_t:
        scale = random.uniform(im_ref_low, im_ref_high) * DATA_H
        scale_f = scale / mask_crop.shape[l_dim]
        ref_size = tuple(s * scale_f for s in obj_ref.shape[:2][::-1])
        transforms = generate_transforms(cat, ref_size, mask_crop, thresh=thresh, total_size=size)
        if transforms:
            valid_t = True
    
    # This is handy for some applications
    cum_area = 0
    for t in transforms:
        transform, ploc, nmask = t
        paste_obj = obj_ref.copy()
        for t in transform:
            paste_obj = t(paste_obj)            
        cum_area += len(np.where(np.ravel(nmask) > 0.9)[0]) # should give considerable color information
        np.set_printoptions(precision=1, linewidth=238, suppress=True, edgeitems=9)

        # should already be an array
        nmask = nmask.reshape(*nmask.shape, 1)
        rio = background[ploc[0]:ploc[0]+nmask.shape[0], ploc[1]:ploc[1]+nmask.shape[1]]
        background_rio = rio * (1.0 - nmask)
        paste_obj = paste_obj * nmask
        bg_paste = cv2.add(paste_obj, background_rio)
        # Add region back into the original image
        background[ploc[0]:ploc[0]+nmask.shape[0], ploc[1]:ploc[1]+nmask.shape[1]] = bg_paste

    return background, cum_area

def generate_set(fidx, nfiles, cat, thresh=0.5):
    # for fidx in range(nfiles):
    nfiles += fidx
    while fidx < nfiles:
        im = generate_image(cat=cat, thresh=thresh)
        cv2.imwrite("%s/%d-%d.jpg" % (data_out, fidx, cat), im)
        fidx += 1

# _, obj_f = single_obj()
if __name__ == "__main__":
    b = background_im()
    for i in range(4):
        im, _ = generate_image(i, thresh=1.0, background=b, size=5002)
        cv2.imwrite("/tmp/test%s.png" % (i), im)

# generate(4000, 4000, cat=4, thresh=0.7)
# generate(12000, 14000, 3, thresh=0.66)
# generate(20000, 24000, 1, thresh=0.56)
# generate(25000, 30000, 1, thresh=0.5)
# generate_set(25000, 30000, 2, thresh=0.65)
# generate(3342, 4700, cat=0, thresh=0.5)
