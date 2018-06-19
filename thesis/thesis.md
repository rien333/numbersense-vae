% Modeling Subitizing with Variational Autoencoders

---
number_sections: true
numbersections: true
header-includes:
   - \usepackage{amssymb}
---


# Introduction #

(**Note:** the quality and the content of the text is not yet reflective of whatever I have in mind for my thesis)

Although various machine learning approaches dealing with the numerical determination of the amount of objects in images already exist, some research can be impartial to broader cognitive debate, resulting in models that are somewhat unhelpful for progressing the understanding of numerical cognition. Any approach to computational modeling of numerical cognition that aims to maintain biological plausibility should adhere to neurological findings about the general characteristics of cognitive processes related to numerical tasks. Essential to understanding the cognitive processes behind number sense is their perceptual origin, for so called  _visual numerosity_ has been posed as the fundamental basis for developmentally later kinds of number sense, such as that required for arithmetical thinking and other more rigorous concepts of number found in mathematics [@lakoff chap. 2; @numerosity-basis]. Visual numerosity is the perceptual capability of many organisms to perceive a group of items as having either a distinct or approximate cardinality. Some specific characteristics of visual numerosity can be derived from it's neural basis. @nieder2016neuronal and @harvey2013topographic present research where topologically organized neural populations exhibiting response profiles that remained largely invariant to all sensory modalities except quantity were discovered. Specifically, the found topological ordering was such that aside from populations responding to their own preferred numerosity, they also showed progressively diminishing activation to preferred numerosities of adjacent populations, in a somewhat bell-shaped fashion [@nieder2016neuronal]. This correspondence between visual quantity and topological network structure leads them to conclude that neural populations can directly  (i.e. without interposition of higher cognitive processes) encode specific visual numerosities. This coding property of neurons participating in visual numoristy was found in humans and other animals alike [see @nieder2016neuronal; @harvey2013topographic].

Notwithstanding the success of previous biologically informed approaches to modeling numerical cognition with artificial neural networks [@stoianov2012], more work is to be done in applying such models to natural images. The main reason behind pursuing natural images is improving the biological plausibility over previous approaches relying on binary images containing only simple geometric shapes [for examples, see @stoianov2012; @wu2018two], given that natural images are closer to everyday sensations than binary images. Furthermore, any dataset with uniform object categories does not capture how visual number sense in animals is abstract in regard to the perceived objects [@nieder2016neuronal], implying that a model should be able show that it performs equally well between objects of different visual complexities. Another way in which biological plausibility will be improved is by guiding some algorithmic decisions on the previously described discovery of the direct involvement of certain neural populations with numerosity perception. Moreover, the properties of these neurons' encoding scheme provide interesting evaluation data to the visual numerosity encoding scheme used by our final model. Some important and closely related characteristics of visual numerosity that constrain our approach, but hopefully improve the biological plausibility of the final model are:

1. Visual number sense is a purely _automatic_ appreciation of the sensory world. It can be characterized as "sudden", or as visible at a glance [@dehaene2011number, p57; @zhang2016salient]. _Convolutional neural networks_ (CNNs) not only showcase excellent performance in extracting visual features from complicated natural images [@GoogleDeepMind; @krizhevsky2012imagenet; for visual number sense and CNNs see @zhang2016salient], but are furthermore functionally inspired by the visual cortex of animals [specifically cats, see @lecun1995convolutional]. CNNs mimicking aspects of the animal visual cortex thus make them an excellent candidate for modeling automatic neural coding by means of numerosity percepts. 

2. The directness of visual number entails that no interposition of external processes is required for numerosity perception, at least no other than lower-level sensory neural processes. More appropriately, the sudden character of visual number sense could be explained by it omitting higher cognitive processes, such as conscious representations [@dehaene2011number indeed points to types of visual number sense being preattentive] or symbolic processing [visual numerosity is completely non-verbal, @nieder2016neuronal]. Furthermore, the existence of visual sense of number in human newborns [@lakoff], animals [@animalsnumericalcognition] and cultures without exact counting systems [@dehaene2011number, p261; @franka2008number] further strengthens the idea that specific kinds of sense of number do not require much mediation and can purely function as an interplay between perceptual capabilities and neural encoding schemes, given the aforementioned groups lack of facilities for abstract reasoning about number [see @Everett2005, p. 626; @lakoff, chap. 3, for a discussion on how cultural facilitaties such as fixed symbols and linguistic practices can facilitate the existence of discrete number in humans]. Indeed, @harvey2013topographic show that earlier mentioned neural populations did not display their characteristic response profile when confronted with Arabic numerals, that is, symbolic representations of number. These populations thus show the ability to function seperately from higer-order representational facilities. Visual number sense being a immediate and purely perceptual process implies that our model should not apply external computational techniques often used in computer vision research on numerical determination task such as counting-by-detection (which requires both arithmetic and iterative attention to all objects in the group) or segmenting techniques [e.g. @chattopadhyay2016counting]. Instead, we want to our model to operate in an autonomous and purely sensory fashion. 

3. Relatedly, visual sense of number is an emergent property of neurons embedded in generative hierarchical learning models, either artificial or biological [@stoianov2012]. The fact that visual number sense exist in animals and human newborns suggests that it is an implicitly learned skill learned at the neural level, for animals do not exhibit a lot of vertical learning, let alone human newborns having received much numerical training. Deemed as a generally unrealistic trope of artificial learning by AI critics [@dreyfus2007heideggerian] and research into the human learning process [@Zorzi2013], modeling visual number necessitates non-researcher depended features. This will restrict the choice of algorithm to so called _unsupervised_ learning algorithms, as such an algorithm will learn its own particular representation of the data distribution. Given their ability to infer the underlying stochastic representation of the data, thus performing autonomous feature determination,  _Varitional Autoencoders_ (VAEs) seem fit to tackle this problem (see [**section x.x**](#vae) for more detail). Moreover, VAEs are trained in an unsupervised manner, similair to how learning visual number sense, given appropriate circumstances, does not require labeled data due it being emergent. Another intersesting aspect of VAEs is their relatively interpretable and overseeable learned feature space, which might tell us something about how it deals with visual numerosity, and thus allow us to evaluate the properties of the VAE's encoding against biological data. 

Unfortunately, no dataset fit for visual numerosity estimation satisfied above requirements (sizable collections of natural image with large and varied, precisely labeled objects groups are hard to construct), forcing present research towards _subitizing_,  a type of visual number sense which had a catered dataset readily available. Subitizing is the ability of many animals to immediately perceive the number of items in a group without resorting to counting or enumeration, given that the number of items falls within the subitizing range of 1-4 [@kaufman1949; @animalsnumericalcognition] Most of the research above was conducted on approximate numerical cognition, but the aforementioned characterisics of visual sense of number hold equally well for a more distinct sense of number such as subitizing. Similairily, subitizing is suggested to be a parallel preattentive process in the visual system [@dehaene2011number, p57], the visual system likely relying on it's ability to recognize holistic patterns for a final subitizing count [@jansen2014role; @dehaene2011number, p57; @piazza2002subitizing]. This means that the "sudden" character of subitizing is caused by the visual system's ability to process simple geometric configurations of objects in parallel, whereby increasing the size of a group behind the subitizing range deprives perceiving this group of it's sudden and distinct numerical perceptual character for this would strain our parallelization capabilities too much. The difference in perceptual character is due to a recourse to enumeration techniques (and possibly others) whenever the subizting parallelization threshold is exceeded, which differ from suddenness in being a consciously guided (i.e. attentive) patterned type of activity. 

Present research therefore asks: how can artificial neural networks be applied to learning the emergent neural skill of subitizing in a manner comparable to their biological equivalents? To answer this, we will first highlight the details of our training procedure by describing a dataset constructed for modeling subitizing and how we implementented our VAE algorithm to learn a representation of this dataset. Next, as the subitizing task is essentialy an image classification task, a methodology for evaluating the unsupervised VAE model on subitizing classification is described. We demonstrate that the performance of our unsupervised approach is comparable with supervised approaches using handcrafted features, although performance is still behind state of the art supervised machine learning due to problems inherent to the particulair VAE implementation. Finally, testing the final models robustness to changes in visual features shows the emergence of a property similair to biological neuron, that is to say, the encoding scheme specifically supports particulair numerosity percepts invariant to visual features other than quantity.

# Related Work #

## Visual Number Sense ##

<!-- Probably skip, but the arxiv paper and stoianov2012 can be reharsed -->
<!-- Investigate the goodfellow2016deep reference as to why it is somewhat computationally expensive -->

As previously described @Stoianov2012 applied artificial neural netwoks to visual numerosity estimation, although without using natural images. They discoverd that some resultant neural populations concerned with numerosity estimation shared multiple properties with biological populations participating in similair tasks, most prominently an encoding scheme that was invariant to the cumaltative surface area of the objects present in the provided images. Present research hopes to discover a similair kind of invarience to surface area. Likewise, we will employ the same scale invarience test, although a succesfull application to natural images already shows a fairly abstract representation of number, as the objects therein already contain varied visual features. 

Some simplicity of the dataset used by @stoianov2012 is due their use of the relatively computationally expensive restricted boltzmann machine [@goodfellow2016deep, chap. 20].  Given developments in generative algorithms and the availablity of more computational power, we will therefore opt for a different algorithmic approach  (as discussed in the introduction) that will hopefully scale better to natural images. 


## Salient Object Subitizing Dataset ##

<!-- Also mention something about how this dataset is concustructed -->
<!-- Do you also explain synthetic data here? (yes seems alright) -->
<!-- Mention the results of their approach in some section -->

As seen in the [figure](#sub) below, the goal of the _Salient Object Subitizing_ (SOS) dataset as defined by @zhang2016salient is to clearly show a number of salient objects that lies within the subitizing range. As other approaches (mention something about 0 and 4)  The dataset was constructed from an ensemble of other datasets to avoid potential dataset bias, and contains approximately 14K natural images [@zhang2016salient]. 

![sos_example](https://github.com/rien333/numbersense-vae/blob/master/thesis/subitizing.png "Example images from the SOS dataset")

# Methods #

## Variational Autoencoder ## {#vae}
                                              ┌───┐
                                              │ z⁰│
                                              │ . │
               ┌─────────────────┐            │ . │            ┌─────────────────┐
               │                 │            │ . │            │                 │
               │                 │            │ . │            │                 │
┏━━━━━━━━┓     │                 │            │ . │            │                 │     ┏━━━━━━━━┓
┃●●●●●●●●┃     │                 │            │ . │            │                 │     ┃■■■■■■■■┃
┃●●●●●●●●┃────▶│ Encoder Network │───────────▶│ . │───────────▶│ Decoder Network │────▶┃■■■■■■■■┃
┃●●●●●●●●┃     │                 │            │ . │            │                 │     ┃■■■■■■■■┃
┗━━━━━━━━┛     │                 │            │ . │            │                 │     ┗━━━━━━━━┛
    X          │                 │            │ . │            │                 │         X'
               │                 │            │ . │            │                 │
               └─────────────────┘            │ . │            └─────────────────┘
                                              │ zⁿ│
                                              └───┘

                                  Latent vector z, with zⁱ ∈ R

VAEs are part of the family of autoencoder algorithms, owing this title to the majority of their structure consisting of an encoder and a decoder module [@doersch2016tutorial] (see figure X for the schematics of an autoencoder). In an regular autoencoder, the encoder module learns to map features from data samples $X \in \mathbb{R}^{n}$ into latent variables $z \in \mathbb{R}^{m}$ often so that $m \ll n$ and thus performs in dimenstionality reduction, while the decoder function learns to reconstruct latent variables $z$ into $\hat{X} \in \mathbb{R}^{n}$ such that $\hat{X}$ matches $X$ according to some predefined similarity measure [@liou2014autoencoder]. Reducing the input to be of much lower dimenstionality forces the autoencoder to learn only the most emblematic regularities of the data, as these will minimize the reconstrution error. The latent space can thus be seen as an inferred hidden feature representation of the data.

Where VAEs primarily differ from regular autoencoders is that rather than directly coding data samples into a feature space, they learn the parameters of a distribution that represents that feature space. Therefore, VAEs perform stochastic inference of the underlying distribution of the input data, instead of only creating some efficient mappping to a lower dimenstionality that simultaneously facilitates accurate reconstruction. Provided with statistical knowledge of the characteristics of the input, VAEs can not only reconstruct exisiting examples, but also generate novel examples that are similair to the input data based on the inferred statistics. The ability to generate novel examples makes VAEs a *generative* algorithm. 

The task of the encoder network in a VAE is to infer the mean and variance parameters of a probability distribution of the latent space $\mathcal{N}(z \mathbin{\vert} \mu{X}, \Sigma{X})$ such that samples $z$ drawn from this distribution facilitate reconstruction of $X$ [@doersch2016tutorial]. Novel sampled $z$ vectors can then be fed into the decoder network as usual. $\mu(X)$ and $\Sigma(X)$ are constrained to roughly follow a unit gaussian by minizing the Kullback-Leibler divergence (denoted as $\mathcal{KL}$) between $\mathcal{N}(0, I)$ and $\mathcal{N}(z \mathbin{\vert} \mu{X}, \Sigma{X})$, where $\mathcal{KL}$ measures the distance between probability distribitions. Normally distributed latent variables capture the intuition behind generative algorithms that they should support sampling latent variables that produce reconstructions that are merely similair to the input, and not necessarily accurate copies. Furthermore, optimizing an abritary distribution would be intractable, thus we need to rely on the fact that given a set $S$ of normaly distributed variales in $\mathbb{R}^{n}$ and any sufficiently complicated function $F$ (such as a neural network), there exist an (mappping) $F(S)$ that can generate any distribution in $\mathbb{R}^{n}$[@doersch2016tutorial]. Therefore, the two optimization objectives of an VAE become [see figure 4 of @doersch2016tutorial]: 

1. $\mathcal{KL}\lbrack\mathcal{N}(\mu(X), \Sigma(X)) \vert\vert \mathcal{N}(0, I)\rbrack$
2. Some reconstruction loss. Within visual problems, plain VAEs can for example minimize the binary cross entropy between $X$ and $\hat{X}$.

The first objective of generating an appropate distribution (research into optimizing distribitions differently exists) function of the latent spance makes VAEs generative, partly satisfying the third constraint outlined in the introduction. To fully satisfy this constraint, the final architecture uses deep neural networks for both the encoder and decoder module (see Experiments VAE architecture), making the implementation an hierarchical model. For a full overview on the implementation of a VAE, refer to @kingma2013auto and @doersch2016tutorial.

## Deep Feature Consistent Perceptual Loss ##

To make the reconstructions made by the VAE perceptually closer to whatever humans deem important characteristics of images, @hou2017deep propose optimizing the reconstructions with help of the hidden layers of a pretrained network. This can be done by predefining a set of layers $l_i$ from a pretrained network (@hou2017deep and present research use VGG-19), and for every $l_i$ matching the hidden representation of the input $x$ to the hidden representation of the reconstruction $\bar{x}$ made by the VAE:

$$\mathcal{L}^{l_{i}}_{rec} = \textrm{MSE}(\Phi(x)^{l_{i}}, \Phi(\bar{x}̄)^{l_{i}})$$

The more mathematical intuition behind this loss is that whatever some hidden layer $l_i$ of the VGG-19 network encodes should be retained in the reconstructed output, as the VGG-19 has proven to model important visual characteristics of a large variety of image types (VGG-19 having been trained on ImageNet). (One notable downside of this approach is that although layers from the VGG-19 represent important visual information , it is well known fact that the first few layers (which seem to work best in our own test, as well as in their original application ) only encode simple features such as edges and lines (i.e countours) which are combined into more complex features in deeper layers. This means that the optimization is somewhat unambitious, in that it will never try to learn any other visual features aside from what the set of predfined layers $l_i$ encode, such as detailed object textures. Indeed, although countour reconstruction has greatly improved with this loss function, reconstructed detail such as facial features show less improvement.

## Extending The Dataset With Syntethic Images ##


# Experiments #

## Hidden Representation Classifier ##

### Classifier architecture ### {#classifierarch}

To asses whether the learned latent space of the VAE network supports subitizing/(the VAE) showcases the (emergent) ability to perform in subitizing, a two layer fully-connected net is fed with latent activation vectors $z_{X_i}$ created by the encoder module of the VAE from an image $X_i$, and a corresponding subitizing class label $Y_i$, where $X_i$ and $Y_i$ are respectively an image and class label from the SOS training set. Both fully-connected layers contain 160 neurons. Each of the linear layers is followed by a batch normalization layer [@ioffe2015batch], a ReLU activation function and a dropout layer, respectively. A fully-connected net was choosen because using another connectionist module for read-outs of the hidden representation heightens the biological plausibility of the final approach [@zorzi2013modeling]. @zorzi2013modeling note that the appended connectionist classifier module can for example be concieved of as cognitive response module (?), although the main reason behind training this classifier is to asses it's performance against other algorithmic and human data. 

### Class imbalance ###

Class imbalance is a phenomenon encountered in datasets whereby the number of instances belonging to at least one of the class is significantly higher than the amount of instances belonging to any of the other classes. Although there is no consensus on an exact definition of what constitutes a dataset with class imbalance, we follow @fernandez2013 in that given over-represented class $c_m$ the number of instances $N_{c_i}$ of one the classes $c_i$ should satisfy $N_{c_i} < 0.4 * N_{c_m}$ for a dataset to be considered imbalanced. For the SOS dataset, $N_{c_0} = 2596$, $N_{c_1} = 4854$,  $N_{c_2} = 1604$ $N_{c_3} = 1058$ and $N_{c_4} = 853$, which implies that $c_0$ and $c_1$ are _majority classes_, while the others should be considered _minority classes_. Most literature makes a distinction between three general algorithm-agnostic approaches that tackle class imbalance [for a discussion, see @fernandez2013]. The first two rebalance the class distribution by altering the amount of examples per class. However, class imbalance can not only be conceived of in terms of quantitative difference, but also as qualitative difference, whereby the relative importance of some class is (weighted?) higher than others (e.g. in classification relating to malignant tumors, misclassifying malignant examples as nonmalignant could be weighted more strongly, see @ref). 
<!-- mention the relevance of qualitative difference to SOS? -->

1. *Oversampling techniques* are a particularly well performing set of solutions to class imbalance are . Oversampling alters the class distribution by producing more examples of the minority class, for example generating synthetic data that resembles minority examples [e.g. @he2008adasyn; @chawla2002smote], resulting in a more balanced class distribution. 
2. *Undersampling techniques*. Undersampling balances the class distribution by discarding examples from the majority class. Elimination of majority class instances can for example ensue by removing those instances that are highly similar [e.g. @tomek1976two]
3. *Cost sensitive techniques.* 

An ensemble of techniques was used to tackle the class imbalance in the SOS dataset. First, slight  random under-sampling with replacement of the two majority classes ($c_0$ and $c_1$) is performed [see [@JMLR:v18:16-365], reducing their size by ~10%. Furthermore, as in practice many common sophisticated under- and oversampling techniques (e.g. data augmentation or outlier removal, for an overview see @fernandez2013) proved largely non-effective, a cost-sensitive class weighting was applied. Cost-senstive ... consists of .... The uneffectiveness of quantive sampling techniques is likely to be caused by that in addition to the quantitative difference in class examples, there is also a slight difficulty factor whereby assesing the class of latent vector $z$ is significantly if belongs to $c_2$ or $c_3$ versus any other class, for these two classes require rather precise contours to discern the invidual objects, in case they for example overlapping, which remains hard for VAEs given their tendency to produce blurred reconstructions. The classifier network therefore seems inclined to put all of its representational power towards the easier classes, as this will result in a lower total cost, whereby this inclination will become even stronger as the quantitative class imbalance grows. The class weights for cost sensitive learning are set according to the quantitative class imbalance ratio [@ref], but better accuracy was obtained by slightly altering the relative difference between the weight by raising all of them to some power $n$. In our experiments, $n=3$ resulted in a balance between high per class accuray scores and aforementioned scores roughly following the same shape as in other algorithms, which hopefully implies that the classifier is able to generalize in a manner comparable to previous approaches. For the SOS dataset with random majority class undersampling, if $n \gg 3$ the classifier accuracy for the majority classes shrinks towards chance, and, interestingly, accuracy for the minority classes becomes comparable to the state of the art.

## Hybrid Dataset ##

Subitizing has been noted to become harder when objects are superimposed, forcing recource to external processes as counting by object enumeration [@dehaene2011number, p57.]. Therefore, the object overlap threshold is increased by $0.1$ starting from $0.5$ for every object added to an image, compared to @zhang2016salient's the static value of $0.5$, as VAEs have been noted to produce blurry reconstructions, indicating a poor ability to code object edges, so distorting class labels. 

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
