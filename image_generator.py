import os
import zclassifier
import random
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import conv_vae_pytorch as vae_pytorch

# Generates synthetic images for the subitizing task, as described in Salient Object Subitizing (2016)

# Setup inline iterm display
# plt.axis("off")

# 256 cus they need to be transformed still
DATA_W = 256
DATA_H = 256

data_t = transforms.Compose(vae_pytorch.data_transform)
classifier = zclassifier.classifier
# some relevant settings are done in z-classifier for conv_vae
model = zclassifier.model
classifier.load_state_dict(
        torch.load("classifier-models/vae-180.pt", map_location=lambda storage, loc: storage))
classifier.eval()

MSRA10K_dir = "../Datasets/MSRA10K_Imgs_GT/"
files_txt = MSRA10K_dir + "files_jpg.txt"
with open(files_txt, "r") as f:
    files = f.read().splitlines()

# data_out = "../Datasets/synthetic2-small"
data_out = "../Datasets/test"
b_dir = "../Datasets/SUN397"
b_classes_txt = b_dir + "/ClassName.txt"
with open(b_classes_txt, "r") as f:
    b_classes = f.read().splitlines()

mask_resizing = Image.BILINEAR
obj_resizing = Image.BILINEAR

def generate(fidx, nfiles, cat):
    # for fidx in range(nfiles):
    nfiles += fidx
    while fidx < nfiles:
        # Load random object
        fname = MSRA10K_dir+random.choice(files)
        # Do this as a preprocessing thing?
        # Object images need to be filtered for containing only one salient object
        obj = Image.open(fname)
        # Does capture values between 0-1
        mask = Image.open(fname.replace(".jpg", ".png")).convert("L")

        # Check for only one object
        im = data_t((np.array(obj), 0))[0].view(1, 3, vae_pytorch.DATA_H, vae_pytorch.DATA_W)
        mu, logvar = model.encode(im.cuda())
        zs = model.reparameterize(mu, logvar)
        outputs = classifier(zs)
        # get max index
        if torch.argmax(outputs).item() != 1 or outputs[0][1] < 0.90:
            continue

        rnd_class = random.choice(b_classes)
        rnd_class_p = b_dir + rnd_class + "/"
        b_ims = random.choice(os.listdir(rnd_class_p))
        background = Image.open(rnd_class_p + b_ims).convert("RGB")
        background = background.resize((DATA_W, DATA_H), Image.BILINEAR)

        # Check if the background is valid, i.e. contains no clear salient object
        im = data_t((np.array(background), 0))[0].view(1, 3, vae_pytorch.DATA_H, vae_pytorch.DATA_W)
        mu, logvar = model.encode(im.cuda())
        zs = model.reparameterize(mu, logvar)
        outputs = classifier(zs)
        if outputs[0][0] < 0.96:
            continue

        box = mask.getbbox()
        mask_crop = mask.crop(box)
        obj_ref = obj.crop(box)
        # size for new basis object 
        scale = random.uniform(0.4,0.8) * DATA_H
        l_dim  = 0 if obj_ref.size[0] > obj_ref.size[1] else 1
        scale_f = scale / obj_ref.size[l_dim]
        new_size = tuple(s * scale_f for s in obj_ref.size)
        # The basis which will be pasted and transformed further
        # Maybe we will could skip the resizing and just use the sizes

        # Paste N e [0,4] objects
        valid_im = True
        total_pixels = []
        label_idxs = [] # list of sets with the indexes of a label L
        ims = [] # These are transforms with speedup

        for i in range(cat):
            transforms = []
            scale_f = random.uniform(0.85,1.15)
            n_paste_size = tuple(int(s*scale_f) for s in new_size)
            resize = lambda i, s=n_paste_size: i.resize(s, obj_resizing)
            n_mask = resize(mask_crop)
            transforms.append(resize)
            # Rotate
            deg = random.uniform(-10, 10)
            # Maybe use transparent pixels for the expand?
            rot = lambda i, d=deg: i.rotate(d, obj_resizing, expand=True)
            n_mask = rot(n_mask)
            transforms.append(rot)

            if random.randint(0,1):
                trans = lambda i: i.transpose(Image.FLIP_LEFT_RIGHT)
                n_mask = trans(n_mask)
                transforms.append(trans)

            # Experiment with width and height start
            try: # idk
                box = (random.randint(0, DATA_H-n_mask.size[1]+1), 
                       random.randint(0, DATA_W-n_mask.size[0]+1))
            except:
                valid_im = False
                break
            # paste the mask in a background array holding all the occupied pixels
            # and check if their area is still 50% after the other pastes
            m = np.array(n_mask, dtype=bool)
            background_arr = np.zeros((DATA_H, DATA_W), dtype=np.bool)

            # After the clip? (idk if clipping is occlussion)
            # Def before bc you want objects to be somewhat visible, so take into account what
            # of their oriignal appearance is still there
            total_pixels += [m.sum()]

            end_h = box[0]+m.shape[0]
            end_w = box[1]+m.shape[1]
            clip_h = background_arr.shape[0] - (end_h)
            clip_w = background_arr.shape[1] - (end_w)

            # Cut off m where it doesn't fit
            if clip_w < 0:
                m = m[:, :clip_w]
            if clip_h < 0:
                m = m[:clip_h]

            background_arr[box[0]:end_h, box[1]:end_w] = m
            n_idxs = set(np.where(background_arr.reshape(DATA_W*DATA_H))[0])
            label_idxs = [idxs - n_idxs for idxs in label_idxs]
            # Update old idxs with new
            label_idxs.append(n_idxs)
            # See how much remains
            for pixel_idxs, t_pixels in zip(label_idxs, total_pixels):
                if len(pixel_idxs) < 0.5 * t_pixels:
                    valid_im = False
                    break
                
            if not valid_im:
                break
            ims.append((transforms, box, n_mask))

        if not valid_im:
            continue
        
        for im in ims:
            transform, box, mask = im
            paste_obj = obj_ref.copy()
            for t in transform:
                paste_obj = t(paste_obj)
            # reverse the box due to PIL/numpy differences
            background.paste(paste_obj, box[::-1], mask)
        background.save("%s/%d-%d.jpg" % (data_out, fidx, cat))
        
        fidx += 1

# Amount per category
# nfiles = [250, 100, 400, 500, 550]
# prev=0
# for cat, n in enumerate(nfiles):
#     generate(prev, n, cat)
#     prev += n

generate(0, 5, 4)
