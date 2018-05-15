import os
import zclassifier
import random
import cv2
from torchvision import transforms
import numpy as np
import torch
import conv_vae_pytorch as vae_pytorch
from torchvision.utils import save_image

# Generates synthetic images for the subitizing task, as described in Salient Object Subitizing (2016)

# Setup inline iterm display
# plt.axis("off")

# 256 cus they need to be transformed still
DATA_W = 256
DATA_H = 256
data_t = transforms.Compose(vae_pytorch.data_transform)
classifier = zclassifier.classifier
if vae_pytorch.args.cuda:
    zclassifier.cuda()
# some relev`ant settings are done in z-classifier for conv_vae
model = zclassifier.model
classifier.load_state_dict(
        torch.load("classifier-models/vae-224.pt", map_location=lambda storage, loc: storage))
classifier.eval()

MSRA10K_dir = "../Datasets/MSRA10K_Imgs_GT/"
files_txt = MSRA10K_dir + "files_jpg.txt"
with open(files_txt, "r") as f:
    files = f.read().splitlines()

# data_out = "../Datasets/synthetic2-small"
data_out = "../Datasets/syn-new"
b_dir = "../Datasets/SUN397"
b_classes_txt = b_dir + "/ClassName.txt"
with open(b_classes_txt, "r") as f:
    b_classes = f.read().splitlines()

mask_resizing = cv2.INTER_LINEAR
obj_resizing = cv2.INTER_LINEAR

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
def generate_transforms(n, ref_size, ref_mask, thresh=0.5):
    transforms = []
    # record amount of pixels each object takes up originally
    total_pixels = []
    label_idxs = []
    for im in range(n):
        transform = []
        scale_f = random.uniform(0.85,1.15)
        n_paste_size = tuple(int(s*scale_f) for s in ref_size)
        resize = lambda i, s=n_paste_size: cv2.resize(i, s, obj_resizing)
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
        tries = 5 # Maybe avioid death locks like so
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

def generate(fidx, nfiles, cat, thresh=0.5):
    # for fidx in range(nfiles):
    nfiles += fidx
    while fidx < nfiles:
        # Load random object
        fname = MSRA10K_dir+random.choice(files)
        # Do this as a preprocessing thing?
        # Object images need to be filtered for containing only one salient object
        # obj = Image.open(fname)
        obj = cv2.imread(fname, 1) # RGB
        # mask = Image.open(fname.replace(".jpg", ".png")).convert("L")
        mask = cv2.imread(fname.replace(".jpg", ".png"), 0).astype(np.float) / 255.0 # Grayscale

        # Check for only one object
        im = data_t((cv2.cvtColor(obj, cv2.COLOR_BGR2RGB), 0))[0].view(1, 3, vae_pytorch.DATA_H, vae_pytorch.DATA_W)
        save_image(im, "test.png")
        mu, logvar = model.encode(im)
        zs = model.reparameterize(mu, logvar)
        outputs = classifier(zs)
        # This always fails for some reason!? ❗❗❗
        # get max index
        if torch.argmax(outputs).item() != 1 or outputs[0][1] < 0.90:
            continue

        rnd_class = random.choice(b_classes)
        rnd_class_p = b_dir + rnd_class + "/"
        b_ims = random.choice(os.listdir(rnd_class_p))
        background = cv2.imread(rnd_class_p + b_ims, 1) # RGB
        background = cv2.resize(background, (DATA_H, DATA_W))

        # Check if the background is valid, i.e. contains no clear salient object
        # This always fails for some reason!? ❗❗❗
        im = data_t((cv2.cvtColor(background, cv2.COLOR_BGR2RGB), 0))[0].view(1, vae_pytorch.DATA_C, vae_pytorch.DATA_H, vae_pytorch.DATA_W)
        mu, logvar = model.encode(im)
        zs = model.reparameterize(mu, logvar)
        outputs = classifier(zs)
        
        if outputs[0][0] < 0.96:
            continue

        background = background.astype(np.float)
        # Get boundix box of white region
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
        valid_t = False
        while not valid_t:
            scale = random.uniform(0.4,0.8) * DATA_H
            scale_f = scale / mask_crop.shape[l_dim]
            new_size = tuple(s * scale_f for s in obj_ref.shape[:2][::-1])
            transforms = generate_transforms(cat, new_size, mask_crop, thresh=thresh)
            if transforms:
                valid_t = True

        for t in transforms:
            transform, ploc, nmask = t
            paste_obj = obj_ref.copy()
            for t in transform:
                paste_obj = t(paste_obj)            
            # should already be an array
            nmask = nmask.reshape(*nmask.shape, 1)
            rio = background[ploc[0]:ploc[0]+nmask.shape[0], ploc[1]:ploc[1]+nmask.shape[1]]
            background_rio = rio * (1.0 - nmask)
            paste_obj = paste_obj * nmask
            bg_paste = cv2.add(paste_obj, background_rio)
            # Add region back into the original image
            background[ploc[0]:ploc[0]+nmask.shape[0], ploc[1]:ploc[1]+nmask.shape[1]] = bg_paste

        cv2.imwrite("%s/%d-%d.jpg" % (data_out, fidx, cat), background)
        fidx += 1

# Amount per category
# prev=0
# nfiles = [1350, 1020, 1550, 1750, 1950]
# thresh = 0.5
# for cat, n in enumerate(nfiles):
#     if cat == 3:
#         thresh = 0.6
#     if cat ==4:
#         thresh = 0.7
#     generate(prev, n, cat, thresh=thresh)
#     prev += n


generate(0, 4000, cat=4, thresh=0.7)
generate(0, 4000, 3, thresh=0.6)
generate(0, 4000, 2, thresh=0.5)
generate(0, 4000, 1, thresh=0.5)
