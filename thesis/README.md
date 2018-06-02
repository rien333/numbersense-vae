# Introduction #

Although various machine learning approaches to numerical determination and estimation of objects in images already exist, some research seems impartial to broader cognitive debate [see @Stoianov2012], resulting in models that are somewhat unhelpful for progressing the understanding of numerical cognition.
<p align="center"><img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/baac3f1faea174b7f9a59fa8e83073cb.svg?invert_in_darkmode" align=middle width=143.26649039999998pt height=39.452455349999994pt/></p>

(**Note:** the quality and the content of the text is yet reflective of whatever i have in mind for my thesis)


# Related Work #

# Method #

## Variational Autoencoder ##

Optimization objectives
1. <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/4a5d495a4a9a1140380700f4affe6e59.svg?invert_in_darkmode" align=middle width=208.26828659999998pt height=24.65753399999998pt/>
2. Visual reconstruction loss (e.g. BCE)

## Deep Feature Consistent ##

To make the reconstructions made by the VAE percetually closer to whatever humans deem important characteristics of images, @hou2017deep propose optimizing the reconstructions with help of the hidden layers of a pretrained network. This can be done by predefining a set of layers <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/bb29cf3d0decad4c2df62b08fbcb2d23.svg?invert_in_darkmode" align=middle width=9.55577369999999pt height=22.831056599999986pt/> from a pretrained network (in their case vgg-19), and for every <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/bb29cf3d0decad4c2df62b08fbcb2d23.svg?invert_in_darkmode" align=middle width=9.55577369999999pt height=22.831056599999986pt/> matching the hidden representation of the input <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> to the hidden representation of the reconstruction <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/33717a96ef162d4ca3780ca7d161f7ad.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=18.666631500000015pt/>:

<p align="center"><img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/5bc2442c0525b19f82e624180207553e.svg?invert_in_darkmode" align=middle width=186.32252594999997pt height=18.88772655pt/></p>

The more mathematical intituition behind this loss is that whatever some hidden layer <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/bb29cf3d0decad4c2df62b08fbcb2d23.svg?invert_in_darkmode" align=middle width=9.55577369999999pt height=22.831056599999986pt/> of the vgg-19 network encodes should be retained in the reconstructed output, as the vgg-19 has proven to model important visual characteristics of a large variety of image types (vgg-19 having been trained on imagenet).

# Experiments #

# Results & Discussion #

## Qualitive Analysis ##

# Conclusion #
