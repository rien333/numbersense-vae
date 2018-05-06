import os
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

MSRA10K_dir = "../Datasets/MSRA10K_Imgs_GT/"
files_txt = MSRA10K_dir + "files_jpg.txt"
with open(files_txt, "r") as f:
    files = f.read().splitlines()

data_out = "../Datasets/synthetic"
b_dir = "../Datasets/SUN397"
b_classes_txt = b_dir + "/ClassName.txt"
with open(b_classes_txt, "r") as f:
    b_classes = f.read().splitlines()

mask_resizing = Image.BILINEAR
obj_resizing = Image.BILINEAR
nfiles = 1000
# for fidx in range(0, len(files), 2):
for fidx in range(nfiles):
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
    if torch.argmax(outputs).item() != 1 and outputs[1] < 0.9:
        continue
    # Formula to see if a thus1000 object contains one object:
    #z[1] should be max, and higher than 0.9

    rnd_class = random.choice(b_classes)
    rnd_class_p = b_dir + rnd_class + "/"
    b_ims = random.choice(os.listdir(rnd_class_p))
    background = Image.open(rnd_class_p + b_ims)
    background = background.resize((DATA_W, DATA_H), Image.BILINEAR)

    
    box = mask.getbbox()
    mask_crop = mask.crop(box)
    obj_ref = obj.crop(box)
    # size for new basis object 
    scale = random.uniform(0.4,0.8) * DATA_H
    l_dim  = 0 if obj_ref.size[0] > obj_ref.size[1] else 1
    scale_f = scale / obj_ref.size[l_dim] 
    new_size = tuple(i * scale_f for i in obj_ref.size)
    # The basis which will be pasted and transformed further
    # Maybe we will could skip the resizing and just use the sizes
    
    # Paste N e [0,4] objects
    background_arr = np.zeros((DATA_H, DATA_W))
    valid_im = True
    total_pixels = []
    ims = [] # These are transforms with speedup
    # Huge speed up:
    # Save the image transforms as lambda's in a list, and apply those only after it has
    # been found that background_arr is valid
    # Then calculate the paste masks from the background_arr 
    
    for i in range(random.randint(1,4)):
        transforms = []
        scale_f = random.uniform(0.85,1.15)
        n_paste_size = tuple(int(i*scale_f) for i in new_size)
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
        box = (random.randint(0, DATA_H-n_mask.size[1]), 
               random.randint(0, DATA_W-n_mask.size[0]))
        # paste the mask in a background array holding all the occupied pixels
        # and check if their area is still 50% after the other pastes
        m = np.array(n_mask, dtype=bool)

        end_h = box[0]+m.shape[0]
        end_w = box[1]+m.shape[1]
        clip_h = background_arr.shape[0] - (end_h)
        clip_w = background_arr.shape[1] - (end_w)

        # Cut off m where it doesn't fit
        if clip_w < 0:
            m = m[:, :clip_w]
        if clip_h < 0:
            m = m[:clip_h]
        # After the clip? (idk if clipping is occlussion)
        total_pixels += [m.sum()]
        background_arr[box[0]:end_h, box[1]:end_w] = m.astype(int)*(i+1)
        # plt.imshow(np.array(background))
        # plt.show()
        for idx, pixels in enumerate(total_pixels):
            new_total = background_arr[background_arr == (idx+1)].sum()
            if new_total < 0.5 * pixels:
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
        background.paste(paste_obj, box, mask)
    background.save("%s/%d.jpg" % (data_out, fidx))

    if fidx > 100:
        break

