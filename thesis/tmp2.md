[1]{.header-section-number} Introduction
========================================

(**Note:** the quality and the content of the text is not yet reflective
of whatever I have in mind for my thesis)

Although various machine learning approaches to numerical determination
and estimation of objects in images already exist, some research seems
impartial to broader cognitive debate (see Stoianov and Zorzi 2012),
resulting in models that are somewhat unhelpful for progressing the
understanding of numerical cognition.

Visual numerical cognition functions differently to for example
mathematical/arithmetical numerical cognition in regard to three closely
related principles (properties). These properties consequently guide the
general approach of this research (source? maybe note that I can discern
three, lakoff might be an source)

1.  Firstly, ...
2.  Fusce commodo.
3.  Visual sense of number is an emergent property of neurons embedded
    in generative hierarchical learning processes/models.
    (biological/animal stuff) (Artificial networks) Modeling visual
    number therefore necessitates non-researcher
    depended(/i.e. handcrafted (you could I guess cite dreyfus that this
    is an unrealistic way of learning that AI suffers from)) features,
    restricting the choice of an algorithm to be unsupervised as such an
    algorithm will learn . Given their ability to construct the
    underlying (stochastic?) representation of the data
    distribution(they learn their own features?) (...in an unsupervised
    manner), *Varitional Autoencoders* (VAEs) seem fit to tackle this
    problem.

\
[\$\$ \\frac{n!}{m!(n-m)!} = {n \\choose m} \$\$]{.math .display}\

[2]{.header-section-number} Related Work
========================================

[2.1]{.header-section-number} Visual Number Sense
-------------------------------------------------

[2.2]{.header-section-number} Subitizing
----------------------------------------

As constructing a dataset fit for visual numerosity estimation is a
difficult task given the lack of other datasets made out of natural
images containing a variety of labeled, large object groups, we set out
to model the phenomenon of *subitizing*, a type of visual number sense
which had a dataset catered to this phenomenon readily avialable. As
seen in the [figure](#sub) below, the goal of the *Salient Object
Subitizing* (SOS) dataset as defined by Zhang et al. (2016) is to
clearly show a number of salient objects that lies within the subitizing
range.

![sos\_example](https://github.com/rien333/numbersense-vae/blob/master/thesis/subitizing.png "Example images from the SOS dataset"){#fig:sub}

[3]{.header-section-number} Methods
===================================

[3.1]{.header-section-number} Variational Autoencoder
-----------------------------------------------------

Optimization objectives

1.  [𝒦ℒ\[𝒩(*μ*(*X*), *Σ*(*X*))\|\|𝒩(0, *I*)\]]{.math .inline}
2.  Visual reconstruction loss (e.g. BCE)

[3.2]{.header-section-number} Deep Feature Consistent Perceptual Loss
---------------------------------------------------------------------

To make the reconstructions made by the VAE perceptually closer to
whatever humans deem important characteristics of images, Hou et al.
(2017) propose optimizing the reconstructions with help of the hidden
layers of a pretrained network. This can be done by predefining a set of
layers [*l*~*i*~]{.math .inline} from a pretrained network (Hou et al.
(2017) and present research use VGG-19), and for every [*l*~*i*~]{.math
.inline} matching the hidden representation of the input [*x*]{.math
.inline} to the hidden representation of the reconstruction [*x̄*]{.math
.inline} made by the VAE:

\
[ℒ~*rec*~^*l*~*i*~^ = MSE(*Φ*(*x*)^*l*~*i*~^, *Φ*(*x̄*̄)^*l*~*i*~^)]{.math
.display}\

The more mathematical intuition behind this loss is that whatever some
hidden layer [*l*~*i*~]{.math .inline} of the VGG-19 network encodes
should be retained in the reconstructed output, as the VGG-19 has proven
to model important visual characteristics of a large variety of image
types (VGG-19 having been trained on ImageNet).

[4]{.header-section-number} Experiments
=======================================

[4.1]{.header-section-number} Hidden Representation Classifier
--------------------------------------------------------------

### [4.1.1]{.header-section-number} Classifier architecture {#classifierarch}

To asses wether the learned latent space of the VAE network supports
subitizing/(the VAE) showcases the (emergent) ability to perform in
subitizing, a two layer fully-connected net is fed with latent
activation vectors [*z*~*X*~*i*~~]{.math .inline} created by the encoder
module of the VAE from an image [*X*~*i*~]{.math .inline}, and a
corresponding subitizing class label [*Y*~*i*~]{.math .inline}, where
[*X*~*i*~]{.math .inline} and [*Y*~*i*~]{.math .inline} are respectively
an image and class label from the SOS training set. Both fully-connected
layers contain 160 neurons. Each of the linear layers is followed by
batchnorm layer ([**???**]{.citeproc-not-found
data-reference-id="batchnorm-ref"}), a ReLU activation function and a
dropout layer, respectively. A fully-connected net was choosen because
using another connectionist module for read-outs of the hidden
representation heightens the biological plausibility of the final
approach (Zorzi, Testolin, and Stoianov 2013). Zorzi, Testolin, and
Stoianov (2013) note that the appended connectionist classifier module
can for example be concieved of as cognitive response module (?),
although the main reason behind training this classifier is to asses
it's performance against other algorithmic and human data.

### [4.1.2]{.header-section-number} Class imbalance

Class imbalance is a phenomenon sometimes encountered in datasets
whereby the number of instances belonging to one or more of the classes
is significantly higher than the amount of instances belonging to any of
the other classes. Although exact definitions on what constitues class
imbalance are highly depended on the problem and method, we follow
([**???**]{.citeproc-not-found data-reference-id="ref"}) in that the
most represented classes should at least account for 40% of the dataset
size. For the SOS dataset, this implies that class 0 and 1 will
*majority classes*, while the others should be considered *minority
classes*. Notably, class imbalance can not only be concieved of in terms
of quantitative difference, but also in qualitive difference, whereby
the relative importance of some class is higher than others (e.g. in
classification relating to malignent tumors, misclassifying malignent
examples as unmalignent could be weighted more strongly, see
([**???**]{.citeproc-not-found data-reference-id="ref"})). Most
literature makes a distinction between three kinds of aproaches to
taclkle class imbalance (for a discussion see
[**???**]{.citeproc-not-found data-reference-id="imbalance"},
[@ADASyn]{.citation data-cites="ADASyn"} [@imbalance2]{.citation
data-cites="imbalance2"}).

1.  *Oversampling techniques*.
2.  *Undersampling techniques*.
3.  *Mauris mollis tincidunt felis.*

An ensemble of techniques was used to tackle the class imbalance in the
SOS dataset. First, slight random undersampling of the two majority
classes (0 and 1) is performed, reducing their size by \~10%
([**???**]{.citeproc-not-found
data-reference-id="ref-random-undersample"}, sckikit maybe).
Furthermore, as in practice many common sophisticated under- and
oversampling techniques (e.g. data augmentation or outlier removal, for
an overview see ([**???**]{.citeproc-not-found
data-reference-id="imbalance"})) proved non-effective, another so called
\"\" algorith was used, namely cost-sensitive class weighting.
Cost-senstive ... consists of .... The uneffectiveness of quantive
sampling techniques is likely to be caused by that in addition to the
quantitative difference in class examples, there is also a slight
difficulty factor whereby assesing the class of latent vector
[*z*]{.math .inline} is significantly if belongs to class 2 or 3 versus
any other, for these two classes require rather precise contours to
discern the invidual objects, in case of them for example overlapping,
which remains hard for VAEs given their tendency to produce blurred
reconstructions. The classifier network therefore seems inclined to put
all of its representational power towards the easier classes, as this
will result in a lower total cost, whereby this inclination will become
even stronger as the quantitative class imbalance grows. The class
weights for cost sensitive learning are set according to the
quantitative class imbalance ratio ([**???**]{.citeproc-not-found
data-reference-id="ref"}), but better accuracy was obtained by slightly
altering the relative difference between the weight by raising all of
them to some power [*n*]{.math .inline}. In our experiments,
[*n* = 3]{.math .inline} resulted in a balance between high per class
accuray scores and aforementioned scores roughly following the same
shape as in other algorithms, which hopefully implies that the
classifier is able to generalize in a manner comparable to previous
approaches. For the SOS dataset with random majority class
undersampling, if [*n* ≫ 3]{.math .inline} the classifier accuracy for
the majority classes shrinks towards chance, and, interestingly,
accuracy for the minority classes becomes comparable to the state of the
art.

[5]{.header-section-number} Results & Discussion
================================================

[5.1]{.header-section-number} Subitizing Read-Out
-------------------------------------------------

Accuray of the `zclassifier` (i.e. the classifier as described in
[**section x.x**](#classifierarch) that learns to classify latent
activation patterns to subitizing labels) is reported over the witheld
SOS test set. Accuracy scores of other algorithms were copied over from
Zhang et al. (2016).

|            | 0    | 1    | 2    | 3    | 4+   | mean |
|-----------:|------|------|------|------|------|------|
|      Chance| 27.5 | 46.5 | 18.6 | 11.7 | 9.7  | 22.8 |
|      SalPry| 46.1 | 65.4 | 32.6 | 15.0 | 10.7 | 34.0 |
|        GIST| 67.4 | 65.0 | 32.3 | 17.5 | 24.7 | 41.4 |
|    SIFT+IVF| 83.0 | 68.1 | 35.1 | 26.6 | 38.1 | 50.1 |
|  zclasifier| 76   | 49   | 40   | 27   | 30   | 44.4 |
|     CNN\_FT| 93.6 | 93.8 | 75.2 | 58.6 | 71.6 | 78.6 |

The subitizing performance of the VAE is comparable to highest scoring
non-machine learning algorithm, and performs worse overall than the CNNs
trained by Zhang et al. (2016). This can be explained by a number of
factors. First of all, the `CNN_ft` algorithm used by Zhang et al.
(2016) has been pretrained on the large, well tested databese of images
it is trained on (i.e. ImageNet, which contains N images, while our
procedure uses M syntethic and N2 natural images). Additionaly, their
model is capable of more complex representations due its depth and the
amount of modules it contains (). Moreover, all their alogirhtms are
trained in a supervised manner, which can sometimes be a lot easier than
unsupervised training ([**???**]{.citeproc-not-found
data-reference-id="ref"})

[5.2]{.header-section-number} Qualitive Analysis
------------------------------------------------

[6]{.header-section-number} Conclusion
======================================

References {#references .unnumbered}
==========

::: {#refs .references}
::: {#ref-hou2017deep}
Hou, Xianxu, Linlin Shen, Ke Sun, and Guoping Qiu. 2017. "Deep Feature
Consistent Variational Autoencoder." In *Applications of Computer Vision
(Wacv), 2017 Ieee Winter Conference on*, 1133--41. IEEE.
:::

::: {#ref-Stoianov2012}
Stoianov, Ivilin, and Marco Zorzi. 2012. "Emergence of a'visual Number
Sense'in Hierarchical Generative Models." *Nature Neuroscience* 15 (2).
Nature Publishing Group: 194.
:::

::: {#ref-zhang2016salient}
Zhang, Jianming, Shuga Ma, Mehrnoosh Sameki, Stan Sclaroff, Margrit
Betke, Zhe Lin, Xiaohui Shen, Brian Price, and Radomír Měch. 2016.
"Salient Object Subitizing." *arXiv Preprint arXiv:1607.07525*.
:::

::: {#ref-zorzi2013modeling}
Zorzi, Marco, Alberto Testolin, and Ivilin Peev Stoianov. 2013.
"Modeling Language and Cognition with Deep Unsupervised Learning: A
Tutorial Overview." *Frontiers in Psychology* 4. Frontiers: 515.
:::
:::
