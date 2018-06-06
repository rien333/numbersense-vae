Introduction
============

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

<p align="center"><img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/baac3f1faea174b7f9a59fa8e83073cb.svg?invert_in_darkmode" align=middle width=143.26649039999998pt height=39.452455349999994pt/></p>

Related Work
============

Visual Number Sense
-------------------

Subitizing
----------

<!-- First also talk about how subitizing is a type of visual number sense, what it is, what the subitizing range is etc. -->
<!-- Also mention something about how this dataset is concustructed -->
<!-- Do you also explain synthetic data here? (see original ordering) -->
As constructing a dataset fit for visual numerosity estimation is a
difficult task given the lack of other datasets made out of natural
images containing a variety of labeled, large object groups, we set out
to model the phenomenon of *subitizing*, a type of visual number sense
which had a dataset catered to this phenomenon readily avialable. As
seen in the [figure](#sub) below, the goal of the *Salient Object
Subitizing* (SOS) dataset as defined by Zhang et al. (2016) is to
clearly show a number of salient objects that lies within the subitizing
range.

![sos\_example](https://github.com/rien333/numbersense-vae/blob/master/thesis/subitizing.png "Example images from the SOS dataset")

Methods
=======

Variational Autoencoder
-----------------------

Optimization objectives

1.  <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/4a5d495a4a9a1140380700f4affe6e59.svg?invert_in_darkmode" align=middle width=208.26828659999998pt height=24.65753399999998pt/>
2.  Visual reconstruction loss (e.g. BCE)

Deep Feature Consistent Perceptual Loss
---------------------------------------

To make the reconstructions made by the VAE perceptually closer to
whatever humans deem important characteristics of images, Hou et al.
(2017) propose optimizing the reconstructions with help of the hidden
layers of a pretrained network. This can be done by predefining a set of
layers <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/bb29cf3d0decad4c2df62b08fbcb2d23.svg?invert_in_darkmode" align=middle width=9.55577369999999pt height=22.831056599999986pt/> from a pretrained network (Hou et al. (2017) and present
research use VGG-19), and for every <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/bb29cf3d0decad4c2df62b08fbcb2d23.svg?invert_in_darkmode" align=middle width=9.55577369999999pt height=22.831056599999986pt/> matching the hidden
representation of the input <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> to the hidden representation of the
reconstruction <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/33717a96ef162d4ca3780ca7d161f7ad.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=18.666631500000015pt/> made by the VAE:

<p align="center"><img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/ef0c0a57404ef77119b5924d13572cc5.svg?invert_in_darkmode" align=middle width=196.6041231pt height=18.88772655pt/></p>

The more mathematical intuition behind this loss is that whatever some
hidden layer <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/bb29cf3d0decad4c2df62b08fbcb2d23.svg?invert_in_darkmode" align=middle width=9.55577369999999pt height=22.831056599999986pt/> of the VGG-19 network encodes should be retained in
the reconstructed output, as the VGG-19 has proven to model important
visual characteristics of a large variety of image types (VGG-19 having
been trained on ImageNet).

Experiments
===========

Hidden Representation Classifier
--------------------------------

### Classifier architecture

To asses wether the learned latent space of the VAE network supports
subitizing/(the VAE) showcases the (emergent) ability to perform in
subitizing, a two layer fully-connected net is fed with latent
activation vectors <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/8ec2ef7749d65d10a01bcd8e313835d4.svg?invert_in_darkmode" align=middle width=22.81049759999999pt height=14.15524440000002pt/> created by the encoder module of the VAE
from an image <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/1338d1e5163ba5bc872f1411dd30b36a.svg?invert_in_darkmode" align=middle width=18.269651399999987pt height=22.465723500000017pt/>, and a corresponding subitizing class label <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/c503cd3cc90b9dc8646daa73c42365ae.svg?invert_in_darkmode" align=middle width=14.19429989999999pt height=22.465723500000017pt/>,
where <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/1338d1e5163ba5bc872f1411dd30b36a.svg?invert_in_darkmode" align=middle width=18.269651399999987pt height=22.465723500000017pt/> and <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/c503cd3cc90b9dc8646daa73c42365ae.svg?invert_in_darkmode" align=middle width=14.19429989999999pt height=22.465723500000017pt/> are respectively an image and class label from the
SOS training set. Both fully-connected layers contain 160 neurons. Each
of the linear layers is followed by batchnorm layer
([**???**]{.citeproc-not-found data-reference-id="batchnorm-ref"}), a
ReLU activation function and a dropout layer, respectively. A
fully-connected net was choosen because using another connectionist
module for read-outs of the hidden representation heightens the
biological plausibility of the final approach (Zorzi, Testolin, and
Stoianov 2013). Zorzi, Testolin, and Stoianov (2013) note that the
appended connectionist classifier module can for example be concieved of
as cognitive response module (?), although the main reason behind
training this classifier is to asses it's performance against other
algorithmic and human data.

### Class imbalance

Class imbalance is a phenomenon encountered in datasets whereby the
number of instances belonging to at least one class is significantly
higher than the amount of instances belonging to any of the other
classes. Although there is no consensus on an exact definition of what
constitutes a dataset with class imbalance, we follow Fernández et al.
(2013) in that given over-represented class <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/e710976777475b48fa8cd4acca538748.svg?invert_in_darkmode" align=middle width=18.778654949999993pt height=14.15524440000002pt/> the number of
instances <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/a20a9547a59ab0b2b1981cdbae565bd3.svg?invert_in_darkmode" align=middle width=23.46794504999999pt height=22.465723500000017pt/> of class <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/3bc6fc8b86b6c61889f4e572c7546b8e.svg?invert_in_darkmode" align=middle width=11.76470294999999pt height=14.15524440000002pt/> should satisfy
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/4bc2a2d82a13d9f7cd3aceb96521c510.svg?invert_in_darkmode" align=middle width=112.73481449999998pt height=22.465723500000017pt/> for a dataset to be considered imbalanced. For
the SOS dataset, \$N\_{c\_0} = 2596, <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/ccf450d9f9b37675e9737077577750e4.svg?invert_in_darkmode" align=middle width=81.11441909999999pt height=22.465723500000017pt/>, <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/a5df93fc55d423f6f51cfbb403f3d6a7.svg?invert_in_darkmode" align=middle width=81.11441909999999pt height=22.465723500000017pt/>
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f39adc9940c0f9d49d251fce913d1a49.svg?invert_in_darkmode" align=middle width=81.11441909999999pt height=22.465723500000017pt/> and <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/c815afb96a24431ab4cdb4f3d4e2dd6e.svg?invert_in_darkmode" align=middle width=72.89520974999999pt height=22.465723500000017pt/>, which implies that <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/09d819a43c6e2990856e40dbda09f893.svg?invert_in_darkmode" align=middle width=13.666351049999989pt height=14.15524440000002pt/> and <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/988584bba6844388f07ea45b7132f61c.svg?invert_in_darkmode" align=middle width=13.666351049999989pt height=14.15524440000002pt/>
are *majority classes*, while the others should be considered *minority
classes*. Most literature makes a distinction between three general
algorithm-agnostic aproaches that tackle class imbalance (for a
discussion, see Fernández et al. 2013). The first two rebalance the
class distribution by altering the amount of examples per class.
However, class imbalance can not only be concieved of in terms of
quantitative difference, but also as qualitive difference, whereby the
relative importance of some class is (weighted?) higher than others
(e.g. in classification relating to malignent tumors, misclassifying
malignent examples as unmalignent could be weighted more strongly, see
([**???**]{.citeproc-not-found data-reference-id="ref"})).
<!-- mention the relevance of qualitive difference to SOS? -->

1.  *Oversampling techniques* are a particulary well performing set of
    solutions to class imbalance are . Oversampling alters the class
    distribution by producing more examples of the minority class, for
    example generating syntethic data that resembles minority examples
    (e.g. [**???**]{.citeproc-not-found data-reference-id="ADAsyn"};
    [**???**]{.citeproc-not-found data-reference-id="SMOTE"}), resulting
    in a more balanced class distribituon.
2.  *Undersampling techniques*. Undersampling balances the class
    distribution by discarding examples from the majority class.
    Elementation of majority class instances can for example ensue by
    removing those instances that are highly similair (e.g. Tomek 1976)
3.  *Cost sensitive techniques.*

An ensemble of techniques was used to tackle the class imbalance in the
SOS dataset. First, slight random undersampling of the two majority
classes (<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/09d819a43c6e2990856e40dbda09f893.svg?invert_in_darkmode" align=middle width=13.666351049999989pt height=14.15524440000002pt/> and <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/988584bba6844388f07ea45b7132f61c.svg?invert_in_darkmode" align=middle width=13.666351049999989pt height=14.15524440000002pt/>) is performed, reducing their size by \~10%
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
difficulty factor whereby assesing the class of latent vector <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.367621899999993pt height=14.15524440000002pt/> is
significantly if belongs to <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/e355414b8774603011922d600510b1df.svg?invert_in_darkmode" align=middle width=13.666351049999989pt height=14.15524440000002pt/> or <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/c2411b9678590eb9d75c90bfebb7c5c1.svg?invert_in_darkmode" align=middle width=13.666351049999989pt height=14.15524440000002pt/> versus any other class, for
these two classes require rather precise contours to discern the
invidual objects, in case they for example overlapping, which remains
hard for VAEs given their tendency to produce blurred reconstructions.
The classifier network therefore seems inclined to put all of its
representational power towards the easier classes, as this will result
in a lower total cost, whereby this inclination will become even
stronger as the quantitative class imbalance grows. The class weights
for cost sensitive learning are set according to the quantitative class
imbalance ratio ([**???**]{.citeproc-not-found
data-reference-id="ref"}), but better accuracy was obtained by slightly
altering the relative difference between the weight by raising all of
them to some power <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/>. In our experiments, <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/aa6905d780872f0007f642420d7a2d9c.svg?invert_in_darkmode" align=middle width=40.00371704999999pt height=21.18721440000001pt/> resulted in a balance
between high per class accuray scores and aforementioned scores roughly
following the same shape as in other algorithms, which hopefully implies
that the classifier is able to generalize in a manner comparable to
previous approaches. For the SOS dataset with random majority class
undersampling, if <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/4df892aafd70851e6daaeca903b9acf1.svg?invert_in_darkmode" align=middle width=43.65668669999999pt height=21.18721440000001pt/> the classifier accuracy for the majority
classes shrinks towards chance, and, interestingly, accuracy for the
minority classes becomes comparable to the state of the art.

Results & Discussion
====================

Subitizing Read-Out
-------------------

<!-- TODO
    - [ ] Mention something about what other alogithms do -->
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

<!-- # | zclasifier | 73   | 49   | 37   | 31   | 26   | 49   | -->
<!-- Epoch 67 -> Test Accuracy: 51 % -->
<!-- Accuracy of     0 : 78 % -->
<!-- Accuracy of     1 : 53 % -->
<!-- Accuracy of     2 : 35 % -->
<!-- Accuracy of     3 : 25 % -->
<!-- Accuracy of     4 : 28 % -->
<!-- Epoch 45 -> Test set loss: 34.28524211883544837 -->
<!-- Mean Accuracy     : 49 % -->
<!-- Accuracy of     0 : 68 % -->
<!-- Accuracy of     1 : 52 % -->
<!-- Accuracy of     2 : 35 % -->
<!-- Accuracy of     3 : 23 % -->
<!-- Accuracy of     4 : 31 % -->
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

Qualitive Analysis
------------------

Conclusion
==========

References
==========

Fernández, Alberto, Victoria López, Mikel Galar, María José del Jesus,
and Francisco Herrera. 2013. "Analysing the Classification of Imbalanced
Data-Sets with Multiple Classes: Binarization Techniques and Ad-Hoc
Approaches." *Knowledge-Based Systems* 42 (April). Elsevier BV: 97--110.
<https://doi.org/10.1016/j.knosys.2013.01.018>.

Hou, Xianxu, Linlin Shen, Ke Sun, and Guoping Qiu. 2017. "Deep Feature
Consistent Variational Autoencoder." In *Applications of Computer Vision
(Wacv), 2017 Ieee Winter Conference on*, 1133--41. IEEE.

Stoianov, Ivilin, and Marco Zorzi. 2012. "Emergence of a'visual Number
Sense'in Hierarchical Generative Models." *Nature Neuroscience* 15 (2).
Nature Publishing Group: 194.

Tomek, Ivan. 1976. "Two Modifications of Cnn." *IEEE Trans. Systems, Man
and Cybernetics* 6: 769--72.

Zhang, Jianming, Shuga Ma, Mehrnoosh Sameki, Stan Sclaroff, Margrit
Betke, Zhe Lin, Xiaohui Shen, Brian Price, and Radomír Měch. 2016.
"Salient Object Subitizing." *arXiv Preprint arXiv:1607.07525*.

Zorzi, Marco, Alberto Testolin, and Ivilin Peev Stoianov. 2013.
"Modeling Language and Cognition with Deep Unsupervised Learning: A
Tutorial Overview." *Frontiers in Psychology* 4. Frontiers: 515.
