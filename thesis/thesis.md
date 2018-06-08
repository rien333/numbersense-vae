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

Although various machine learning approaches dealing with the numerical determination of the amount of objects in images already exist, some research seems impartial to broader cognitive debate [see @stoianov2012], resulting in models that are somewhat unhelpful for progressing the understanding of numerical cognition. Any approach to computational modeling of numerical cognition that aims to maintain biological plausibility should adhere to neurological findings about the general characteristics of cognitive processes related to numerical tasks. Essential to understanding the cognitive processes behind number sense are their perceptual origins, for so called  _visual numerosity_ has been posed as the fundamental basis for developmentally later kinds of number sense, such as that required for arithmetical thinking and other more rigorous concepts of number found in mathematics. Visual numerosity is the perceptual capability of many organisms to perceive a group of items as having either a distinct or approximate cardinality. Some specific characteristics of visual numerosity can be derived from it's neural basis. @nieder2016neuronal and @harvey2013topographic present research where topologically organized neural populations exhibiting particular response profiles to visual changes in quantity were discovered, whose response profiles furthermore remained invariant to changes of visual features other than quantity. In particular, their topological ordering was such that aside from populations responding to their own preferred numerosity, they also showed progressively diminishing activation to preferred numerosities of adjacent populations, in a somewhat bell-shaped fashion [@nieder2016neuronal]. This correspondence between visual quantity and topological network structure leads them to conclude that neural populations directly code (i.e. without interposition of higher cognitive processes) specific visual numerosities. This coding property of neurons participating in visual numoristy was found in humans and other animals alike [see @nieder2016neuronal; @harvey2013topographic]. 

<!-- move to another section maybe? related work? -->
Notwithstanding the success of previous biologically informed approaches to modeling numerical cognition with artificial neural networks [@stoianov2012], more work is to be done in applying such models to natural images. The main reason behind pursuing natural images is improving the biological plausibility over previous approaches using binary images containing only simple geometric shapes [for examples, see @stoianov2012; @wu2018two], given that natural images are closer to everyday sensations than binary images. Furthermore, any dataset with uniform object categories does not capture how visual number sense in animals is abstract in regard to the perceived objects [@nieder2016neuronal], meaning that a model should be able show that it performs equally well between object of different visual complexities. Another way in which we will improve biological plausibility is by looking at the discovery of the direct involvement of certain neural populations' with numerosity perception (as described previously), which will guide various algorithmic decisions. Moreover, the properties of these neurons' encoding scheme provide interesting comparison data to the visual numerosity encoding scheme used by our final model. Some important and closely related characteristics of visual numerosity that constrain our approach, but hopefully improve the biological plausibility of the final model are:

1. Visual number sense is a purely automatic appreciation of the sensory world. It can be characterized as "sudden", or as visible at a glance. _Convolutional neural networks_ (CNNs) not only showcase excellent performance in extracting visual features from complicated natural images [@GoogleDeepMind; @krizhevsky2012imagenet; for visual number sense and CNNs see @zhang2016salient], but are furthermore based on the neural functioning of the perceptual system of animals [specifically cats, see @lecun1995convolutional]. CNNs mimicking aspects of the animal visual cortex thus make them an excellent candidate for modeling automatic neural coding by means of specific numerosity percepts. 
2. The directness of visual number sense leads us to conclude that no interposition of external process is required for numerosity perception, at least no other than the immediate appreciation of sensory data by means of low-level neural processes. More appropriately, the sudden character of visual number sense could be explained by it omitting higher cognitive processes, such as conscious representations (@dehaene2011number indeed points to types of visual number sense being preattentive) or symbolic processing (visual numerosity is completely non-verbal, @nieder2016). Moreover, the existence of visual sense of number in human newborns [@lakoff], animals [@animalsnumericalcognition] and cultures without exact counting systems [@dehaene2011number, p261; everett?] further strengthen the idea that specific kinds of sense of number do not require much mediation and functions as an interplay between perceptual capabilities and their neural encodings, given the aforementioned groups lack of requirements for reasoning about number. In line with cognitive research, earlier mentioned neural populations indeed did not display their characteristic response profile when confronted with Arabic numerals (i.e. symbolic representations of number), and thus is able to function without for example linguistic facilities. Visual number sense being a purely perceptual process, implies that our model should not use external computational processes often used in computer vision research on numerical determination task such as counting-by-detection (which requires both arithmetic and iterative attention to all objects in the group) or segmenting techniques [e.g. @chattopadhyay2016counting].
3. Visual sense of number is an emergent property of neurons embedded in generative hierarchical learning processes, which even holds for artificial models [@zhang2016salient]. The fact that it also exist in animals suggests that it is an implicitly learned skill learned at a neural level, for animals do not exhibit a lot of vertical learning, let alone human newborns having received much numerical training. Modeling visual number therefore necessitates non-researcher depended features (i.e.  should avoid commonly used handcrafted features such as SIFT or HOG features), a generally unrealistic trope of artificial learning according to AI critics [@dreyfus2007heideggerian] and research into the human learning process [@Zorzi2013], restricting the choice of an algorithm to be unsupervised as such an algorithm will learn its own distribution of the data. Given their ability to construct the underlying stochastic representation of the data, i.e. autonomous feature determination they learn their own features,  _Varitional Autoencoders_ (VAEs) seem fit to tackle this problem. Moreover, VAEs are an unsupervised algorithm, just as learning visual number sense does not rely on supervision.

Most of the research above was conducted on approximate numerical cognition, but these characteristics hold equally well for number systems that offer a more distinct sense of number (e.g. subitizing, see below). Likewise, subitizing is suggested to be implemented as parallel preattentive process in the visual system [@dehaene2011number, p57], whereby the visual system might rely on it's ability to recognize holistic patterns to arrive at a final subitizing count [@jansen2014role; dehaene2011number, p57; @piazza2002subitizing]. This means that the "sudden" character of subitizing (unsurprisingly, sudden is it's etymological stem) is arrived our visual systems ability to process simple geometric configuration of objects in parallel, whereby an increase in complexity deprives perceiving a group with 4+ items of it's sudden distinct numerical perpetual character for this would strain our parallelization capabilities too much. This strain would then lead us to resort to counting, which has a perceptual character from suddenness (enumeration is a very strict and patterned activity). 

Present research therefore asks: How can we apply artificial neural networks to learning the emergent cognitive/neural skill of subitizing in a manner comparable to their biological equivalents?


# Related Work #

## Visual Number Sense ##

## Subitizing ##

<!-- First also talk about how subitizing is a type of visual number sense, what it is, what the subitizing range is etc. -->
<!-- Also mention something about how this dataset is concustructed -->
<!-- Do you also explain synthetic data here? (see original ordering) -->

As constructing a dataset fit for visual numerosity estimation is a difficult task given the lack of other datasets made out of natural images containing a variety of labeled, large object groups, we set out to model the phenomenon of _subitizing_,  a type of visual number sense which had a dataset catered to this phenomenon readily available. As seen in the [figure](#sub) below, the goal of the _Salient Object Subitizing_ (SOS) dataset as defined by @zhang2016salient is to clearly show a number of salient objects that lies within the subitizing range.

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

To asses whether the learned latent space of the VAE network supports subitizing/(the VAE) showcases the (emergent) ability to perform in subitizing, a two layer fully-connected net is fed with latent activation vectors $z_{X_i}$ created by the encoder module of the VAE from an image $X_i$, and a corresponding subitizing class label $Y_i$, where $X_i$ and $Y_i$ are respectively an image and class label from the SOS training set. Both fully-connected layers contain 160 neurons. Each of the linear layers is followed by batch normalization layer [@ioffe2015batch], a ReLU activation function and a dropout layer, respectively. A fully-connected net was choosen because using another connectionist module for read-outs of the hidden representation heightens the biological plausibility of the final approach [@zorzi2013modeling]. @zorzi2013modeling note that the appended connectionist classifier module can for example be concieved of as cognitive response module (?), although the main reason behind training this classifier is to asses it's performance against other algorithmic and human data. 

### Class imbalance ###

Class imbalance is a phenomenon encountered in datasets whereby the number of instances belonging to at least one of the class is significantly higher than the amount of instances belonging to any of the other classes. Although there is no consensus on an exact definition of what constitutes a dataset with class imbalance, we follow @fernandez2013 in that given over-represented class $c_m$ the number of instances $N_{c_i}$ of one the classes $c_i$ should satisfy $N_{c_i} < 0.4 * N_{c_m}$ for a dataset to be considered imbalanced. For the SOS dataset, $N_{c_0} = 2596$, $N_{c_1} = 4854$,  $N_{c_2} = 1604$ $N_{c_3} = 1058$ and $N_{c_4} = 853$, which implies that $c_0$ and $c_1$ are _majority classes_, while the others should be considered _minority classes_. Most literature makes a distinction between three general algorithm-agnostic approaches that tackle class imbalance [for a discussion, see @fernandez2013]. The first two rebalance the class distribution by altering the amount of examples per class. However, class imbalance can not only be conceived of in terms of quantitative difference, but also as qualitative difference, whereby the relative importance of some class is (weighted?) higher than others (e.g. in classification relating to malignant tumors, misclassifying malignant examples as nonmalignant could be weighted more strongly, see @ref). 
<!-- mention the relevance of qualitative difference to SOS? -->

1. *Oversampling techniques* are a particularly well performing set of solutions to class imbalance are . Oversampling alters the class distribution by producing more examples of the minority class, for example generating synthetic data that resembles minority examples [e.g. @ADAsyn; @SMOTE], resulting in a more balanced class distribution. 
2. *Undersampling techniques*. Undersampling balances the class distribution by discarding examples from the majority class. Elimination of majority class instances can for example ensue by removing those instances that are highly similar [e.g. @tomek1976two]
3. *Cost sensitive techniques.* 

An ensemble of techniques was used to tackle the class imbalance in the SOS dataset. First, slight  random undersampling of the two majority classes ($c_0$ and $c_1$) is performed, reducing their size by ~10% [@ref-random-undersample, sckikit maybe]. Furthermore, as in practice many common sophisticated under- and oversampling techniques (e.g. data augmentation or outlier removal, for an overview see [@imbalance]) proved largelt non-effective, a cost-sensitive class weighting was applied. Cost-senstive ... consists of .... The uneffectiveness of quantive sampling techniques is likely to be caused by that in addition to the quantitative difference in class examples, there is also a slight difficulty factor whereby assesing the class of latent vector $z$ is significantly if belongs to $c_2$ or $c_3$ versus any other class, for these two classes require rather precise contours to discern the invidual objects, in case they for example overlapping, which remains hard for VAEs given their tendency to produce blurred reconstructions. The classifier network therefore seems inclined to put all of its representational power towards the easier classes, as this will result in a lower total cost, whereby this inclination will become even stronger as the quantitative class imbalance grows. The class weights for cost sensitive learning are set according to the quantitative class imbalance ratio [@ref], but better accuracy was obtained by slightly altering the relative difference between the weight by raising all of them to some power $n$. In our experiments, $n=3$ resulted in a balance between high per class accuray scores and aforementioned scores roughly following the same shape as in other algorithms, which hopefully implies that the classifier is able to generalize in a manner comparable to previous approaches. For the SOS dataset with random majority class undersampling, if $n \gg 3$ the classifier accuracy for the majority classes shrinks towards chance, and, interestingly, accuracy for the minority classes becomes comparable to the state of the art.

## Synthetic images ##

Subitizing has been noted to become harder when objects are superimposed, thus resorting to external process as counting by object enumeration [@dehaene2011number, p57.]. We therefore increase the overlap threshold for subitizing classes with more objects compared to @zhang2016salient.

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

<!--  LocalWords:  numbersections
 -->
