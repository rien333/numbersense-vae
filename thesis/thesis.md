% Modeling Subitizing with Variational Autoencoders

---
number_sections: true
numbersections: true
output: 
  html_document:
    number_sections: true
---

# Introduction #

(**Note:** the quality and the content of the text is not yet reflective of whatever I have in mind for my thesis)

Although various machine learning approaches to numerical determination and estimation of objects in images already exist, some research seems impartial to broader cognitive debate [see @Stoianov2012], resulting in models that are somewhat unhelpful for progressing the understanding of numerical cognition.

Visual numerical cognition functions differently to for example mathematical/arithmetical numerical cognition in regard to three closely related principles (properties). These properties consequently guide the general approach of this research (source? maybe note that I can discern three, lakoff might be an source)

1. Firstly, ... 
2. Fusce commodo.
3. Visual sense of number is an emergent property of neurons embedded in generative hierarchical learning processes/models. (biological/animal stuff) (Artificial networks) Modeling visual number therefore necessitates non-researcher depended(/i.e. handcrafted (you could I guess cite dreyfus that this is an unrealistic way of learning that AI suffers from)) features, restricting the choice of an algorithm to be unsupervised as such an algorithm will learn . Given their ability to construct the underlying (stochastic?) representation of the data distribution(they learn their own features?) (...in an unsupervised manner), _Varitional Autoencoders_ (VAEs) seem fit to tackle this problem. 

$$
\frac{n!}{m!(n-m)!} = {n \choose m}
$$


# Related Work #

## Visual Number Sense ##

## Subitizing ##

<!-- First also talk about how subitizing is a type of visual number sense, what it is, what the subitizing range is etc. -->
<!-- Also mention something about how this dataset is concustructed -->
<!-- Do you also explain synthetic data here? (see original ordering) -->

As constructing a dataset fit for visual numerosity estimation is a difficult task given the lack of other datasets made out of natural images containing a variety of labeled, large object groups, we set out to model the phenomenon of _subitizing_,  a type of visual number sense which had a dataset catered to this phenomenon readily avialable. As seen in the [figure](#sub) below, the goal of the _Salient Object Subitizing_ (SOS) dataset as defined by @zhang2016salient is to clearly show a number of salient objects that lies within the subitizing range.

![sos_example](https://github.com/rien333/numbersense-vae/blob/master/thesis/subitizing.png "Example images from the SOS dataset")

# Methods #

## Variational Autoencoder ##

Optimization objectives

1. $\mathcal{KL}\lbrack\mathcal{N}(\mu(X), \Sigma(X)) \vert\vert \mathcal{N}(0, I)\rbrack$
2. Visual reconstruction loss (e.g. BCE)

## Deep Feature Consistent Perceptual Loss ##

To make the reconstructions made by the VAE perceptually closer to whatever humans deem important characteristics of images, @hou2017deep propose optimizing the reconstructions with help of the hidden layers of a pretrained network. This can be done by predefining a set of layers $l_i$ from a pretrained network (@hou2017deep and present research use VGG-19), and for every $l_i$ matching the hidden representation of the input $x$ to the hidden representation of the reconstruction $\bar{x}$ made by the VAE:

$$\mathcal{L}^{l_{i}}_{rec} = \textrm{MSE}(\Phi(x)^{l_{i}}, \Phi(\bar{x}̄)^{l_{i}})$$

The more mathematical intuition behind this loss is that whatever some hidden layer $l_i$ of the VGG-19 network encodes should be retained in the reconstructed output, as the VGG-19 has proven to model important visual characteristics of a large variety of image types (VGG-19 having been trained on ImageNet).

# Experiments #

## Hidden Representation Classifier ##

### Classifier architecture ### {#classifierarch}

To asses wether the learned latent space of the VAE network supports subitizing/(the VAE) showcases the (emergent) ability to perform in subitizing, a two layer fully-connected net is fed with latent activation vectors $z_{X_i}$ created by the encoder module of the VAE from an image $X_i$, and a corresponding subitizing class label $Y_i$, where $X_i$ and $Y_i$ are respectively an image and class label from the SOS training set. Both fully-connected layers contain 160 neurons. Each of the linear layers is followed by batchnorm layer [@batchnorm-ref], a ReLU activation function and a dropout layer, respectively. A fully-connected net was choosen because using another connectionist module for read-outs of the hidden representation heightens the biological plausibility of the final approach [@zorzi2013modeling]. @zorzi2013modeling note that the appended connectionist classifier module can for example be concieved of as cognitive response module (?), although the main reason behind training this classifier is to asses it's performance against other algorithmic and human data. 

### Class imbalance ###

Class imbalance is a phenomenon encountered in datasets whereby the number of instances belonging to at least one class is significantly higher than the amount of instances belonging to any of the other classes. Although there is no consensus on an exact definition of what constitutes a dataset with class imbalance, we follow @Fernandez2013 in that given over-represented class $c_m$ the number of instances $N_{c_i}$ of class $c_i$ should satisfy $N_{c_i} < 0.4 * N_{c_m}$ for a dataset to be considered imbalanced. For the SOS dataset, $N_{c_0} = 2596$, $N_{c_1} = 4854$,  $N_{c_2} = 1604$ $N_{c_3} = 1058$ and $N_{c_4} = 853$, which implies that $c_0$ and $c_1$ are _majority classes_, while the others should be considered _minority classes_. Most literature makes a distinction between three general algorithm-agnostic aproaches that tackle class imbalance [for a discussion, see @Fernandez2013]. The first two rebalance the class distribution by altering the amount of examples per class. However, class imbalance can not only be concieved of in terms of quantitative difference, but also as qualitive difference, whereby the relative importance of some class is (weighted?) higher than others (e.g. in classification relating to malignent tumors, misclassifying malignent examples as unmalignent could be weighted more strongly, see @ref). 
<!-- mention the relevance of qualitive difference to SOS? -->

1. *Oversampling techniques* are a particulary well performing set of solutions to class imbalance are . Oversampling alters the class distribution by producing more examples of the minority class, for example generating syntethic data that resembles minority examples [e.g. @ADAsyn; @SMOTE], resulting in a more balanced class distribituon. 
2. *Undersampling techniques*. Undersampling balances the class distribution by discarding examples from the majority class. Elementation of majority class instances can for example ensue by removing those instances that are highly similair [e.g. @tomek1976two]
3. *Cost sensitive techniques.* 

An ensemble of techniques was used to tackle the class imbalance in the SOS dataset. First, slight  random undersampling of the two majority classes ($c_0$ and $c_1$) is performed, reducing their size by ~10% [@ref-random-undersample, sckikit maybe]. Furthermore, as in practice many common sophisticated under- and oversampling techniques (e.g. data augmentation or outlier removal, for an overview see [@imbalance]) proved non-effective, another so called "" algorith was used, namely cost-sensitive class weighting. Cost-senstive ... consists of .... The uneffectiveness of quantive sampling techniques is likely to be caused by that in addition to the quantitative difference in class examples, there is also a slight difficulty factor whereby assesing the class of latent vector $z$ is significantly if belongs to $c_2$ or $c_3$ versus any other class, for these two classes require rather precise contours to discern the invidual objects, in case they for example overlapping, which remains hard for VAEs given their tendency to produce blurred reconstructions. The classifier network therefore seems inclined to put all of its representational power towards the easier classes, as this will result in a lower total cost, whereby this inclination will become even stronger as the quantitative class imbalance grows. The class weights for cost sensitive learning are set according to the quantitative class imbalance ratio [@ref], but better accuracy was obtained by slightly altering the relative difference between the weight by raising all of them to some power $n$. In our experiments, $n=3$ resulted in a balance between high per class accuray scores and aforementioned scores roughly following the same shape as in other algorithms, which hopefully implies that the classifier is able to generalize in a manner comparable to previous approaches. For the SOS dataset with random majority class undersampling, if $n \gg 3$ the classifier accuracy for the majority classes shrinks towards chance, and, interestingly, accuracy for the minority classes becomes comparable to the state of the art.

# Results & Discussion #

## Subitizing Read-Out ##

<!-- TODO
    - [ ] Mention something about what other alogithms do -->

Accuray of the `zclassifier` (i.e. the classifier as described in [**section x.x**](#classifierarch) that learns to classify latent activation patterns to subitizing labels) is reported over the witheld SOS test set. Accuracy scores of other algorithms were copied over from @zhang2016salient.

|            | 0    | 1    | 2    | 3    | 4+   | mean |
|-----------:|------|------|------|------|------|------|
|     Chance | 27.5 | 46.5 | 18.6 | 11.7 | 9.7  | 22.8 |
|     SalPry | 46.1 | 65.4 | 32.6 | 15.0 | 10.7 | 34.0 |
|       GIST | 67.4 | 65.0 | 32.3 | 17.5 | 24.7 | 41.4 |
|   SIFT+IVF | 83.0 | 68.1 | 35.1 | 26.6 | 38.1 | 50.1 |
| zclasifier | 76   | 49   | 40   | 27   | 30   | 44.4 |
|     CNN_FT | 93.6 | 93.8 | 75.2 | 58.6 | 71.6 | 78.6 |

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

The subitizing performance of the VAE is comparable to highest scoring non-machine learning algorithm, and performs worse overall than the CNNs trained by @zhang2016salient. This can be explained by a number of factors. First of all, the `CNN_ft` algorithm used by @zhang2016salient has been pretrained on the large, well tested databese of images it is trained on (i.e. ImageNet, which contains N images, while our procedure uses M syntethic and N2 natural images). Additionaly, their model is capable of more complex representations due its depth and  the amount of modules it contains (). Moreover, all their alogirhtms are trained in a supervised manner, which can sometimes be a lot easier than unsupervised training [@ref]

## Qualitive Analysis ##


# Conclusion #

# References
