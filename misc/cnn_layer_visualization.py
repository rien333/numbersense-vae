"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak

❗ Okay but what if you feed this an z vector that is likely go generate a certain classlab
and then use the decoder part (doesn't work)
❗ Optimize a random sampled z value instead of an image with the deconv layer

"""
import os
import numpy as np

import conv_vae_pytorch as vaepytorch
import torch
from torch.autograd import Variable
from torch.optim import SGD
from torchvision import transforms
from torchvision.utils import save_image

# Input: numpy image
def preprocess_image(im):
    t = vaepytorch.data_transform
    t = transforms.Compose(t)
    t1, _ = t((im, 0))
    return t1

class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Generate a random image
        # Okay but why with these values?
        # just go with it I guess
        self.created_image = np.uint8(np.random.uniform(150, 180, (227, 227, 3)))
        # Create the folder to export images if not exists
        if not os.path.exists('generated'):
            os.makedirs('generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        # self.model[self.selected_layer].register_forward_hook(hook_function)
        self.model.conv1.register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Process image and return variable
        p_im = preprocess_image(self.created_image)
        self.processed_image = Variable(p_im, requires_grad=True)
        # Define optimizer for the image
        # Earlier layers need higher learning rates to visualize whereas later layers need less
        optimizer = SGD([self.processed_image], lr=5, weight_decay=1e-6)
        for i in range(1, 151):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            layers = [self.model.conv1, self.model.conv2, self.model.conv3, self.model.conv4,]
            for index, layer in enumerate(layers):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Save image
            if i % 50 == 0:
                save_image(self.processed_image.cpu(), 'generated/layer_vis_l' + str(self.selected_layer) +
                           '_f' + str(self.selected_filter) + '_iter'+str(i)+'.jpg') # set nrow?
                # cv2.imwrite('../generated/layer_vis_l' + str(self.selected_layer) +
                #             '_f' + str(self.selected_filter) + '_iter'+str(i)+'.jpg',
                #             self.created_image)

    def visualise_layer_without_hooks(self):
        p_im = preprocess_image(self.created_image).view(1,3,227, 227)
        self.processed_image = Variable(p_im, requires_grad=True)
        # Define optimizer for the image
        # Earlier layers need higher learning rates to visualize whereas later layers need less
        optimizer = SGD([self.processed_image], lr=6, weight_decay=1e-6)
        for i in range(1, 451):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            # x = x.cuda()
            layers = [self.model.conv1, self.model.conv2, self.model.conv3, self.model.conv4,]
            for index, layer in enumerate(layers):
                # Forward pass layer by layer
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            # self.created_image = recreate_image(self.processed_image)
            # Save image
            if i % 50 == 0:
                save_image(self.processed_image.cpu(), 'generated/layer_vis_l' + str(self.selected_layer) +
                           '_f' + str(self.selected_filter) + '_iter'+str(i)+'.jpg') # set nrow?

if __name__ == '__main__':
    cnn_layer = 1
    filter_pos = 5 # which filter?
    # Fully connected layer is not needed
    # pretrained_model = models.vgg16(pretrained=True).features
    
    layer_vis = CNNLayerVisualization(vaepytorch.model, cnn_layer, filter_pos)

    # What's the difference with the two hooks?

    # Layer visualization with pytorch hooks
    # layer_vis.visualise_layer_with_hooks()

    # Layer visualization without pytorch hooks
    layer_vis.visualise_layer_without_hooks()
