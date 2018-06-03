# Introduction #

Although various machine learning approaches to numerical determination and estimation of objects in images already exist, some research seems impartial to broader cognitive debate [see @Stoianov2012], resulting in models that are somewhat unhelpful for progressing the understanding of numerical cognition.

$$
\frac{n!}{m!(n-m)!} = {n \choose m}
$$

(**Note:** the quality and the content of the text is not yet reflective of whatever I have in mind for my thesis)


# Related Work #

## Visual Number Sense ##


## Subitizing ##

<!-- First also talk about how subitizing is a type of visual number sense, what it is, what the subitizing range is etc. -->

As constructing a dataset fit for visual numerosity estimation is a difficult task given the lack of other datasets made out of natural images containing a variety of labeled, large object groups, we set out to model the phenomenon of _subitizing_,  a type of visual number sense which had a dataset catered to this phenomenon readily avialable. As seen in the figure below, the goal of the _Salient Object Subitizing_ (SOS) dataset as defined by @zhang2016salient is to clearly show a number of salient objects that lies within the subitizing range. ![sos_example](https://github.com/rien333/numbersense-vae/blob/master/thesis/subitizing.png "Example images from the SOS dataset")

# Methods #

## Variational Autoencoder ##

Optimization objectives

1. $\mathcal{KL}\lbrack\mathcal{N}(\mu(X), \Sigma(X)) \vert\vert \mathcal{N}(0, I)\rbrack$
2. Visual reconstruction loss (e.g. BCE)

## Deep Feature Consistent Perceptual Loss ##

To make the reconstructions made by the VAE perceptually closer to whatever humans deem important characteristics of images, @hou2017deep propose optimizing the reconstructions with help of the hidden layers of a pretrained network. This can be done by predefining a set of layers $l_i$ from a pretrained network (@hou2017deep and present research use VGG-19), and for every $l_i$ matching the hidden representation of the input $x$ to the hidden representation of the reconstruction $\bar{x}$ made by the VAE:

$$\[\mathcal{L}^{l_{i}}_{rec} = \textrm{MSE}(\Phi(x)^{l_{i}}, \Phi(\bar{x}̄)^{l_{i}})\]$$

The more mathematical intuition behind this loss is that whatever some hidden layer $l_i$ of the VGG-19 network encodes should be retained in the reconstructed output, as the VGG-19 has proven to model important visual characteristics of a large variety of image types (VGG-19 having been trained on ImageNet).

# Experiments #

# Results & Discussion #

## Qualitive Analysis ##

# Conclusion #

# References
