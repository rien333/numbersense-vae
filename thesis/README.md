# Introduction

Although various machine learning approaches to numerical determination
and estimation of objects in images already exist, some research seems
impartial to broader cognitive debate (see Stoianov and Zorzi 2012),
resulting in models that are somewhat unhelpful for progressing the
understanding of numerical cognition.

\[
\frac{n!}{m!(n-m)!} = {n \choose m}
\]

(**Note:** the quality and the content of the text is not yet reflective
of whatever I have in mind for my thesis)

# Related Work

# Method

## Variational Autoencoder

Optimization
    objectives

1.  \(\mathcal{KL}\lbrack\mathcal{N}(\mu(X), \Sigma(X)) \vert\vert \mathcal{N}(0, I)\rbrack\)
2.  Visual reconstruction loss (e.g. BCE)

## Deep Feature Consistent Perceptual Loss

To make the reconstructions made by the VAE perceptually closer to
whatever humans deem important characteristics of images, Hou et al.
(2017) propose optimizing the reconstructions with help of the hidden
layers of a pretrained network. This can be done by predefining a set of
layers \(l_i\) from a pretrained network (in Hou et al. (2017)’s case
VGG-19), and for every \(l_i\) matching the hidden representation of the
input \(x\) to the hidden representation of the reconstruction
\(\bar{x}\):

\[\[\mathcal{L}^{l_{i}}_{rec} = \textrm{MSE}(\Phi(x)^{l_{i}}, \Phi(\bar{x}̄)^{l_{i}})\]\]

The more mathematical intuition behind this loss is that whatever some
hidden layer \(l_i\) of the VGG-19 network encodes should be retained in
the reconstructed output, as the VGG-19 has proven to model important
visual characteristics of a large variety of image types (VGG-19 having
been trained on imagenet).

# Experiments

# Results & Discussion

## Qualitive Analysis

# Conclusion

# References

<div id="refs" class="references">

<div id="ref-hou2017deep">

Hou, Xianxu, Linlin Shen, Ke Sun, and Guoping Qiu. 2017. “Deep Feature
Consistent Variational Autoencoder.” In *Applications of Computer Vision
(Wacv), 2017 Ieee Winter Conference on*, 1133–41. IEEE.

</div>

<div id="ref-Stoianov2012">

Stoianov, Ivilin, and Marco Zorzi. 2012. “Emergence of a’visual Number
Sense’in Hierarchical Generative Models.” *Nature Neuroscience* 15 (2).
Nature Publishing Group: 194.

</div>

</div>
