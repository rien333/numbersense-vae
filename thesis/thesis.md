% Modeling Subitizing with Variational Autoencoders

---
number_sections: true
numbersections: true
header-includes:
   - \usepackage{amssymb}
   - \usepackage{amsmath}
   - \usepackage{ tipa }
---

<!-- TODO
    - [ ] Write abstract
-->

# Introduction #

(**Note:** the quality and the content of the text is not yet reflective of whatever I have in mind for my thesis)

Although various machine learning approaches dealing with the numerical determination of the amount of objects in images already exist, some research can be impartial to broader cognitive debate, resulting in models that are somewhat unhelpful for progressing the understanding of numerical cognition. Any approach to computational modeling of numerical cognition that aims to maintain biological plausibility should adhere to neurological findings about the general characteristics of cognitive processes related to numerical tasks. Essential to understanding the cognitive processes behind number sense is their perceptual origin, for so called  _visual numerosity_ has been posed as the fundamental basis for developmentally later kinds of number sense, such as that required for arithmetical thinking and other more rigorous concepts of number found in mathematics [@lakoff chap. 2; @numerosity-basis]. Visual numerosity is the perceptual capability of many organisms to perceive a group of items as having either a distinct or approximate cardinality. Some specific characteristics of visual numerosity can be derived from it's neural basis. @nieder2016neuronal and @harvey2013topographic present research where topologically organized neural populations exhibiting response profiles that remained largely invariant to all sensory modalities except quantity were discovered. Specifically, the found topological ordering was such that aside from populations responding to their own preferred numerosity, they also showed progressively diminishing activation to preferred numerosities of adjacent populations, in a somewhat bell-shaped fashion [@nieder2016neuronal]. This correspondence between visual quantity and topological network structure leads them to conclude that neural populations can directly  (i.e. without interposition of higher cognitive processes) encode specific visual numerosities. This coding property of neurons participating in visual numerosity was found in humans and other animals alike [see @nieder2016neuronal; @harvey2013topographic].

Notwithstanding the success of previous biologically informed approaches to modeling numerical cognition with artificial neural networks [@stoianov2012], more work is to be done in applying such models to natural images. The main reason behind pursuing natural images is improving the biological plausibility over previous approaches relying on binary images containing only simple geometric shapes [for examples, see @stoianov2012; @wu2018two], given that natural images are closer to everyday sensations than binary images. Furthermore, any dataset with uniform object categories does not capture how visual number sense in animals is abstract in regard to the perceived objects [@nieder2016neuronal], implying that a model should be able show that it performs equally well between objects of different visual complexities. Another way in which biological plausibility will be improved is by guiding some algorithmic decisions on the previously described discovery of the direct involvement of certain neural populations with numerosity perception. Moreover, the properties of these neurons' encoding scheme provide interesting evaluation data to the visual numerosity encoding scheme used by our final model. Some important and closely related characteristics of visual numerosity that constrain our approach, but hopefully improve the biological plausibility of the final model are:

1. Visual number sense is a purely _automatic_ appreciation of the sensory world. It can be characterized as "sudden", or as visible at a glance [@dehaene2011number, p. 57; @zhang2016salient]. _Convolutional neural networks_ (CNNs) not only showcase excellent performance in extracting visual features from complicated natural images [@GoogleDeepMind; @krizhevsky2012imagenet; for visual number sense and CNNs see @zhang2016salient], but are furthermore functionally inspired by the visual cortex of animals [specifically cats, see @lecun1995convolutional]. CNNs mimicking aspects of the animal visual cortex thus make them an excellent candidate for modeling automatic neural coding by means of numerosity percepts. 

2. The directness of visual number entails that no interposition of external processes is required for numerosity perception, at least no other than lower-level sensory neural processes. More appropriately, the sudden character of visual number sense could be explained by it omitting higher cognitive processes, such as conscious representations [@dehaene2011number, p. 58 indeed points to types of visual number sense being pre-attentive] or symbolic processing [visual numerosity percepts are understood non-verbally, @nieder2016neuronal]. Furthermore, the existence of visual sense of number in human newborns [@lakoff, chap. 1], animals [@animalsnumericalcognition] and cultures without exact counting systems [@dehaene2011number, p. 261; @franka2008number] further strengthens the idea that specific kinds of sense of number do not require much mediation and can purely function as an interplay between perceptual capabilities and neural encoding schemes, given the aforementioned groups lack of facilities for abstract reasoning about number [see @Everett2005, p. 626; @lakoff, chap. 3, for a discussion on how cultural facilities such as fixed symbols and linguistic practices can facilitate the existence of discrete number in humans]. Indeed, @harvey2013topographic show that earlier mentioned neural populations did not display their characteristic response profile when confronted with Arabic numerals, that is, symbolic representations of number. These populations thus show the ability to function separately from higher-order representational facilities. Visual number sense being a immediate and purely perceptual process implies that our model should not apply external computational techniques often used in computer vision research on numerical determination task such as counting-by-detection [which requires both arithmetic and iterative attention to all group members, see @zhang2016salient; @detection2016unconstrained] or segmenting techniques [e.g. @chattopadhyay2016counting]. Instead, we want to our model to operate in an autonomous and purely sensory fashion. 

<!-- I'm not sure if this holds in the biological case, but you could argue that "Comparable to how some research describes the brain as Bayesian machine [for a discussion see @bayesianbrain; @bayes2], VAEs learn a stochastic model of sensory input data by optimizing the parameters of a probability distribution such that the probability distribution maximizes the likelihood of the input (or "training") data." 🌸 Maybe after the blosssom: although not specifically true for numerosity learning, the brain is suggestively organized in a similair fashion as some research suggests ... 

Another solution would be ommiting or replacing the two constraints, by for example stating about possibility for complex organisation, and in the case of artificial neurons those embedded in hierarchically structured, generative learning models

"The generative basis of natural number concepts" vaguely supports the biological case for generative models and number sense (generation being some kind of innate skill or something)

(after biological:) under the probable assumption that the brain is also a preditive modeller/stochastic inferational device
-->

3. Relatedly, visual sense of number is an emergent property of hierarchically organized neurons embedded in generative learning models, either artificial or biological [@stoianov2012; the brain can be characterized as building a predictive modeler, or a "Bayesian machine", @bayesianbrain; @bayes2]. The fact that visual number sense exist in animals and human newborns suggests that it is an implicitly learned skill learned at the neural level, for animals do not exhibit a lot of vertical learning, let alone human newborns having received much numerical training. Deemed as a generally unrealistic trope of artificial learning by AI critics [@dreyfus2007heideggerian] and research into the human learning process [@Zorzi2013], modeling visual number necessitates non-researcher depended features. This will restrict the choice of algorithm to so called _unsupervised_ learning algorithms, as such an algorithm will learn its own particular representation of the data distribution. Given their ability to infer the underlying stochastic representation of the data, i.e. perform in autonomous feature determination,  _Variational Autoencoders_ (VAEs) seem fit to tackle this problem ([**section x.x**](#vae) details their precise working). Moreover, VAEs are trained in an unsupervised manner similar to how, given appropriate circumstances, visual numerosity abilities are implicitly learned skills that emerge without "labeled data". Another interesting aspect of VAEs is their relatively interpretable and overseeable learned feature space, which might tell something about how it deals with visual numerosity, and thus allows us to evaluate the properties of the VAE's encoding against biological data. 

Unfortunately, no dataset fit for visual numerosity estimation task similar to @stoianov2012 satisfied above requirements (sizable collections of natural image with large and varied, precisely labeled objects groups are hard to construct), forcing present research towards _subitizing_,  a type of visual number sense which had a catered dataset readily available. Subitizing is the ability of many animals to immediately perceive the number of items in a group without resorting to counting or enumeration, given that the number of items falls within the subitizing range of 1-4 [@kaufman1949; @animalsnumericalcognition] Most of the research above was conducted on approximate numerical cognition, but the aforementioned characteristics of visual sense of number hold equally well for a more distinct sense of number such as subitizing. Similarly, subitizing is suggested to be a parallel pre-attentive process in the visual system [@dehaene2011number, p. 57], the visual system likely relying on it's ability to recognize holistic patterns for a final subitizing count [@jansen2014role; @dehaene2011number, p. 57; @piazza2002subitizing]. This means that the "sudden" character of subitizing is caused by the visual system's ability to process simple geometric configurations of objects in parallel, whereby increasing the size of a group behind the subitizing range deprives perceiving this group of it's sudden and distinct numerical perceptual character for this would strain our parallelization capabilities too much. The difference in perceptual character is due to a recourse to enumeration techniques (and possibly others) whenever the subitizing parallelization threshold is exceeded, which differ from suddenness in being a consciously guided (i.e. attentive), patterned type of activity. 

Present research therefore asks: how can artificial neural networks be applied to learning the emergent neural skill of subitizing in a manner comparable to their biological equivalents? To answer this, we will first highlight the details of our training procedure by describing a dataset constructed for modeling subitizing and how we implemented our VAE algorithm to learn a representation of this dataset. Next, as the subitizing task is essentially an image classification task, a methodology for evaluating the unsupervised VAE model's performance on the subitizing classification task is described. We demonstrate that the performance of our unsupervised approach is comparable with supervised approaches using handcrafted features, although performance is still behind state of the art supervised machine learning approaches due to problems inherent to the particular VAE implementation. Finally, measuring the final models robustness to changes in visual features shows the emergence of a property similar to biological neurons, that is to say, the VAE's encoding scheme specifically supports particular numerosity percepts invariant to visual features other than quantity.

# Related Work #

<!-- TODO
    - [ ] Related work is often structured with line subsection, do that as it also saves space
-->

## Visual Number Sense ##

<!-- Probably skip, but the arxiv paper and stoianov2012 can be reharsed -->
<!-- TODO
    - [x] Investigate the goodfellow2016deep reference as to why it is somewhat computationally expensive
-->

As previously described, @stoianov2012 applied artificial neural networks to visual numerosity estimation, although without using natural images. They discovered neural populations concerned with numerosity estimation that shared multiple properties with biological populations participating in similar tasks, most prominently an encoding scheme that was invariant to the cumulative surface area of the objects present in the provided images. Present research hopes to discover a similar kind of invariance to surface area. Likewise, we will employ the same scale invariance test, although a successful application to natural images already shows a fairly abstract representation of number, as the objects therein already contain varied visual features. 

Some simplicity of the dataset used by @stoianov2012 is due their use of the relatively computationally expensive Restricted Boltzmann Machine (RBM) [with the exception of exploiting prior knowledge of regularities in the probability distribution over the observed data, equation $\tag{20.6}$ from @goodfellow2016deep shows that computational cost in RBMs grows as a multiple of the size of it's hidden and observed units]. Given developments in generative algorithms and the availability of more computational power, we will therefore opt for a different algorithmic approach (see [section X.X](vae)) that will hopefully scale better to natural images. 


## Salient Object Subitizing Dataset ##

<!-- Do you also explain synthetic data here? (yes seems alright) -->
<!-- Mention the results of their approach in some section -->

As seen in the [figure](#sub) below, the goal of the _Salient Object Subitizing_ (SOS) dataset as defined by @zhang2016salient is to clearly show a number of salient objects that lies within the subitizing range. As other approaches often perform poor on images with complex backgrounds or with a large number of objects, @zhang2016salient also introduce images with no salient objects, as well as images where the number of salient objects lies outside of the subitizing range (labeled as "4+"). The dataset was constructed from an ensemble of other datasets to avoid potential dataset bias, and contains approximately 14K natural images [@zhang2016salient]. 

![sos_example](https://github.com/rien333/numbersense-vae/blob/master/thesis/subitizing.png "Example images from the SOS dataset")

# Methods #

## Variational Autoencoder ## {#vae}

<!--
    TODO 
    - [ ] Image of complete vae architecture
-->

<!--
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
--> 

![autoencoder](https://github.com/rien333/numbersense-vae/blob/master/thesis/ae-small.svg "Schematic architecture of an autoencoder")


VAEs are part of the family of autoencoder algorithms, owing this title to the majority of their structure consisting of an encoder and a decoder module [@doersch2016tutorial] (see figure X for the schematics of an autoencoder). In an regular autoencoder, the encoder module learns to map features from data samples $X \in \mathbb{R}^{n}$ into latent variables $z \in \mathbb{R}^{m}$ often so that $m \ll n$ and thus performs in dimensionality reduction, while the decoder function learns to reconstruct latent variables $z$ into $X' \in \mathbb{R}^{n}$ such that $X'$ matches $X$ according to some predefined similarity measure [@liou2014autoencoder]. Reducing the input to be of much lower dimensionality forces the autoencoder to learn only the most emblematic regularities of the data, as these will minimize the reconstruction error. The latent space can thus be seen as an inferred hidden feature representation of the data.

Where VAEs primarily differ from regular autoencoders is that rather than directly coding data samples into some feature space, they learn the parameters of a distribution that represents that feature space. Therefore, VAEs perform stochastic inference of the underlying distribution of the input data, instead of only creating some efficient mapping to a lower dimensionality that simultaneously facilitates accurate reconstruction. Now provided with statistical knowledge of the characteristics of the input, VAEs can not only perform reconstruction, but also generate novel examples that are similar to the input data based on the inferred statistics. The ability to generate novel examples makes VAEs a *generative* algorithm. 

The task of the encoder network in a VAE is to infer the mean and variance parameters of a probability distribution of the latent space $\mathcal{N}(z \mathbin{\vert} \mu(X), \Sigma(X))$ such that samples $z$ drawn from this distribution facilitate reconstruction of $X$ [@doersch2016tutorial]. Novel sampled $z$ vectors can then be fed into the decoder network as usual. $\mu(X)$ and $\Sigma(X)$ are constrained to roughly follow a unit Gaussian by minimizing the Kullback-Leibler divergence (denoted as $\mathcal{KL}$) between $\mathcal{N}(0, I)$ and $\mathcal{N}(z \mathbin{\vert} \mu(X), \Sigma(X))$, where $\mathcal{KL}$ measures the distance between probability distributions. Normally distributed latent variables capture the intuition behind generative algorithms that they should support sampling latent variables that produce reconstructions that are merely _similar_ to the input, and not necessarily accurate copies. Furthermore, optimizing an arbitrary distribution would be intractable, thus VAEs need to rely on the fact that given a set of normally distributed variables $S = \{s_1, ... s_\mathbf{n}\}$ with $S \in \mathbb{R}^{\mathbf{n}}$ and any sufficiently complicated function $f(s_i)$ (such as a neural network), there exists a mapping $f: S \mapsto S'$ from which we can generate any arbitrary distribution $P(X) \in \mathbb{R}^{\mathbf{n}}$ with $S' \sim P(X)$ [@doersch2016tutorial].

Therefore, the optimization objectives of a VAE become [also see figure 4 of @doersch2016tutorial]: 

1. $\mathcal{KL}\lbrack\mathcal{N}(\mu(X), \Sigma(X)) \vert\vert \mathcal{N}(0, I)\rbrack$
2. Some reconstruction loss. Within visual problems, plain VAEs can for example minimize the binary cross entropy (BCE) between $X$ and $X'$.

Objective **(1)** grants VAEs the ability to generate new samples from the learned distribution, partly satisfying the constraint outlined in the introduction whereby visual numerosity skills emerge in generative learning models. To fully satisfy this constraint, the final architecture uses deep neural networks for both the encoder and decoder module (see figure X for the VAE architecture), making the implementation an hierarchical model as well. As an VAEs latent space encodes the most important features of the data, it is hoped the samples drawn from the encoder provide information regarding it's subitizing performance (see [**section x.x**](#readout)). For a complete overview of implementing a VAE, refer to @kingma2013auto and @doersch2016tutorial.

## Deep Feature Consistent Perceptual Loss ##

<!-- TODO
    - [ ] Note something about alpha and beta? Our dataset worked better with alpha ~= 0.001 and beta, based on reported final reconstrution error, although more work in optimizing this ratio is to be done.
    - [x] Comparison on the sos dataset (see compare_rnd.py)
    - [x] Find a reference for the properties of the first few layers of a deep CNN
-->

<!-- This reason is a little strange. I think you need to focus more on how CNNs or say something about CNNs capturing more visual characterisics/human (visual) similarity judgemets in general -->

Because the frequently used pixel-by-pixel reconstruction loss measures in VAEs do not necessarily comply with human perceptual similarity judgements, @hou2017deep propose optimizing the reconstructions with help of the hidden layers of a pretrained deep CNN network, because these models are particularly better at capturing spatial correlation compared to pixel-by-pixel measurements [@hou2017deep]. Additionally, CNNs haven proven to model visual characteristics of images deemed important by humans, for example by being able to perform complex classification tasks [@krizhevsky2012imagenet]. The ability of the proposed _Feature Perceptual Loss_ (FPL) to retain spatial correlation should reduce the noted blurriness [@larsen2015autoencoding] of the VAE's reconstructions, which is especially problematic in subitizing tasks since blurring merges objects which in turn distorts subitizing labels. @hou2017deep and present research employ VGG-19 [@simonyan2014very] as the pretrained network  $\Phi$, trained on the large and varied ImageNet [@russakovsky2015imagenet] dataset. FPL requires predefining a set of layers $l_i \in L$ from pretrained network $\Phi$, and works by minimizing the mean squared error (MSE) between the hidden representations of input $x$ and VAE reconstruction $\bar{x}$ at every layer $l_i$.  Aside from the $\mathcal{KL}$-divergence, the VAE's second optimization objective is now as follows:

<!-- 
This but as a sum.
You could also just note two objectives, and so also introduce alpha and beta 
-->

$$ \sum_{l \in L}^{} = \textrm{MSE}(\Phi(x)^{l}, \Phi(\bar{x}̄)^{l})$$

The intuition behind FPL is that whatever some hidden layer $l_i$ of the VGG-19 network encodes should be retained in the reconstruction $\bar{x}$, as the VGG-19 has proven to model important visual characteristics of a large variety of image types. In @hou2017deep's and our experiments $L = \{\texttt{relu1\_1, relu2\_1, relu3\_1\}}$ resulted in the best reconstructions. One notable shortcoming of FPL is that although the layers from VGG-19 represent important visual information, it is a known fact that the first few layers of deep CNNs only encode simple features such as edges and lines (i.e they support contours), which are only combined into more complex features deeper into the network [@liu2017towards; FPL's authors @hou2017deep note something similar]. This means that the optimization objective is somewhat unambitious, in that it will never try to learn any other visual features [for examples, refer to @liu2017towards, Fig. 6.] aside from what the set of predefined layers $L$ represents. Indeed, although contour reconstruction has greatly improved with FPL, the reconstruction of detail such as facial features shows less improvement. Although @hou2017deep show a successful application of FPL, they might have been unaware of this shortcoming due to using a more uniform dataset consisting only of centered faces. For a comparison between FPL and BCE-based reconstruction measures on the SOS dataset, refer to figure Y.Y. 

## Hybrid Dataset ## {#hybrid}

<!-- TODO
    - [x] Hybrid dataset (intuition, settings graph etc, pretrain,)
    - [x] Synthetic image settings
    - [ ] Generate bezier graph
-->

We follow @zhang2016salient in pre-training our model with synthetic images and later fine-tuning on the SOS dataset. However, some small chances to their synthetic image pre-training setup are proposed. First, the synthetic dataset is extended with natural images from the SOS dataset such that the amount of examples per class per epoch is always equal (hopefully reducing problems encountered with class imbalance, see [section X.X](#imbalance)). Another reason for constructing a hybrid dataset was the fact that the generation process of synthetic images was noted to produce 1. fairly unrealistic looking examples and 2. considerably less suitable than natural data for supporting subitizing performance [@zhang2016salient]. A further intuition behind this dataset is thus that the representation of the VAE must always be at least a little supportive of natural images, instead of settling on some optimum for synthetic images. A final reason for including natural images is that any tested growth in dataset size during pre-training resulted into lower losses. The ratio of natural to synthethic images is increased over time, defined by a bezier curve with parameters $u_0=0, u_1=-0.01, u_2=0.02, u_3=1.0$ shown in figure X.X. We grow the original data size by roughly 8 times, pre-training with a total of 80000 hybrid samples per epoch. Testing many different parameters for the hybrid dataset was not given much priority as the total loss seemed to shrink with dataset expansion and training and testing a full model was time expensive.

<!-- Make this a pdf when exporting, maybe in the generate figures.fish thing -->
![bezier_curve](https://github.com/rien333/numbersense-vae/blob/master/thesis/bezier.png "Bezier curve defining the ratio of natural images over syntethic images at time _t_")

Synthetic images are generated by pasting cutout objects from THUS10000 [@cheng2015global] onto the SUN background dataset [@xiao2010sun]. The subitizing label is aquired by pasting an object $N$ times, with $N \in \lbrack0, 4\rbrack$. For each paste, the object is transformed in equivalent manner to @zhang2016salient. However, subitizing is noted be more difficult when objects are superimposed, forcing recource to external processes as counting by object enumeration [@dehaene2011number, p. 57.], implying that significant paste object overlap should be avoided. @zhang2016salient avoid object overlap by defining a threshold $t in \lbrack0,1\rbrack$ whereby an object's visible pixels $P_{visible}$ and total pixel amount $P_{total}$ should satisfy $P_{visible} > t * P_{total$}. For reasons given above, we define the object overlap threshold as $t = \sum_{n=0}^{N} 0.5 + n_i * 0.1$ with $N \in \lbrack0,4\rbrack$ compared to @zhang2016salient's static $t=0.5$, as VAEs are espcially prone produce blurry reconstructions [@hou2017deep; @larsen2015autoencoding], which requires extra care with overlapping objects as to not distort class labels.

# Experiments #

## Hidden Representation Classifier ## {#readout}

### Classifier architecture ### {#classifierarch}

<!-- Maybe or maybe not mention dropout probabilities (should be final tho) -->
<!-- TODO
    - [ ] Mention that the experiments used a slightly different architecture, with less conv. 
      filters and fully conntected layers appended before and after the latent represntation
          - so this is theoritical performance (not time avialable for extra test)
    - [ ] Mention  per class performance being 10-20% higher depending how we skew the imbalance
      ratio's and techniques
-->

To asses whether the learned latent space of the VAE showcases the emergent ability to perform in subitizing, a two layer fully-connected net is fed with latent activation vectors $z_{X_i}$ created by the encoder module of the VAE from an image $X_i$, and a corresponding subitizing class label $Y_i$, where $X_i$ and $Y_i$ are respectively an image and class label from the SOS training set. Both fully-connected layers contain 160 neurons. Each of the linear layers is followed by a batch normalization layer [@ioffe2015batch], a ReLU activation function and a dropout layer [@srivastava2014dropout] with dropout probability $x$ and $y$ respectively. A fully-connected net was chosen because using another connectionist module for read-outs of the hidden representation heightens the biological plausibility of the final approach [@zorzi2013modeling]. Additionaly, @zorzi2013modeling note that the appended connectionist classifier module be concieved of as a cognitive response module supporting a particular behavioral task, although the main reason behind training this classifier is to asses it's performance against other algorithmic data. 

### Class imbalance ### {#imbalance}

Class imbalance is a phenomenon encountered in datasets whereby the number of instances belonging to on or more classes is significantly higher than the amount of instances belonging to any of the other classes. Although there is no consensus on an exact definition of what constitutes a dataset with class imbalance, we follow @fernandez2013 in that given over-represented class $c_m$ the number of instances $N_{c_i}$ of one the classes $c_i$ should satisfy $N_{c_i} < 0.4 * N_{c_m}$ for a dataset to be considered imbalanced. For the SOS dataset, $N_{c_0} = 2596$, $N_{c_1} = 4854$,  $N_{c_2} = 1604$ $N_{c_3} = 1058$ and $N_{c_4} = 853$, which implies that $c_0$ and $c_1$ are _majority classes_, while the others should be considered _minority classes_. Most literature makes a distinction between three general algorithm-agnostic approaches that tackle class imbalance [for a discussion, see @fernandez2013]. The first two rebalance the class distribution by altering the amount of examples per class. However, class imbalance can not only be conceived of in terms of quantitative difference, but also as qualitative difference, whereby the relative importance of some class is weighted higher than others (e.g. in classification relating to malignant tumors, misclassifying malignant examples as nonmalignant could be weighted stonger  than other misclassifications) Qualitive difference might be relevant to the SOS dataset, because examples with overlapping (i.e. multiple) objects make subitizing inherently more difficult (see section X.X (#hybrid)), and previous results on subitizing show that some classes are more difficult to classify than others [@zhang2016salient]. 

1. *Oversampling techniques* are a particularly well performing set of solutions to class imbalance. Oversampling alters the class distribution by producing more examples of the minority class, for example generating synthetic data that resembles minority examples [e.g. @he2008adasyn; @chawla2002smote], resulting in a more balanced class distribution. 
2. *Undersampling techniques*. Undersampling balances the class distribution by discarding examples from the majority class. Elimination of majority class instances can for example ensue by removing those instances that are highly similar [e.g. @tomek1976two]
3. *Cost sensitive techniques.*  Cost sensitive learning does not alter the distribution of class instances, but penalizes misclassification of certain classes. Cost sensitive techqniques are especially useful for dealing with minority classes that are inherently more difficult (or "costly") to correctly classify, as optimisation towards easier classes could minimize cost even in quantitatively balanced datasets if the easier classes for example require lesser representational resources of the learning model.

An ensemble of techniques was used to tackle the class imbalance in the SOS dataset. First, slight random under-sampling with replacement of the two majority classes ($c_0$ and $c_1$) is performed [see @JMLR:v18:16-365], reducing their size by ~10%. Furthermore, as in practice many common sophisticated under- and oversampling techniques (e.g. data augmentation or outlier removal, for an overview see @fernandez2013) proved largely non-effective, a cost-sensitive class weighting was applied. The uneffectiveness of quantive sampling techniques is likely to be caused by that in addition to the quantitative difference in class examples, there is also a slight difficulty factor whereby assesing the class of latent vector $z$ is significantly if belongs to $c_2$ or $c_3$ versus any other class, for these two classes require rather precise contours to discern invidual objects, even more so with overlapping objects, while precise contours remain hard for VAEs given their tendency to produce blurred reconstructions [@larsen2015autoencoding]. The classifier network therefore seems inclined to put all of its representational power towards the easier classes, as this will result in a lower total cost, whereby this inclination will become even stronger as the quantitative class imbalance grows. The class weights for cost sensitive learning are set according to the quantitative class imbalance ratio [equivalent to section 3.2 in @fernandez2013], but better accuracy was obtained by slightly altering the relative difference between the weights by raising them to some power $n$. In our experiments, $n=3$ resulted in a balance between high per class accuray scores and aforementioned scores roughly following the same shape as in other algorithms, which hopefully implies that the classifier is able to generalize in a manner comparable to previous approaches. For the SOS dataset with random majority class undersampling, if $n \gg 3$ the classifier accuracy for the majority classes shrinks towards chance, and, interestingly, accuracy for the minority classes becomes comparable to the state of the art machine learning techqniques.


# Results & Discussion #


## Variational Autoencoder Performance ##

<!-- TODO
    - [x] graphs of loss (eh niey zo boeied)
    - [x] Mus example with all the dimensions
    - [x] Discuss and show quality of representation (visual problem/dfc versus plain)
-->


After pre-training for 102 epochs, and fine-tuning on the SOS dataset for 39 epochs, we found that the loss of our VAE did not shrink anymore. For the specific purpose of subitizing, we can see that using FPL loss is beneficial (indeed, that is what we found when comparing the two models in the classification task described in [section X.X](#readout)) The reconstructions of a plain VAE and a VAE that uses FPL as it's reconstruction optimization objective are show in figure X.X. To get an idea of what sort of properties the latent space of the VAE encodes, refer to figure Z.z. 

![fpl_vs_bce](https://github.com/rien333/numbersense-vae/blob/master/thesis/dfc_bce.png "Comparison of BCE and FPL reconstruction loss measures. The top row are the original images from the SOS dataset, and the other two images are reconstructions made by using FPL and BCE loss, respectively. (the VAE using BCE was pre-trained for 40 epochs, and fine-tuned for 35 epochs)")

Although the reconstructions show increased quality of contour reconstruction, there are a few reoccurring visual disparities between original and reconstruction. First of, novel patterns often emerge in the reconstructions, possibly caused by a implementational glitch, or a considerable difference in tested datasets (FPL is frequently paired with the CelebA [@liu2015deep] dataset). Datasets other than the SOS dataset showed slightly better performance, indicating that the SOS dataset is either to small, too varied or requires non standard tweaking for FPL to work in it's current form. Most of the improvement in more uniform datasets came from the fact that the VAE learned to create more local patterns to give the appearance of uniformly colored regions, but upon closer inspection placed pixels of colors in a grid such that they gave the appearance of just one color, similar to how for example LED screens function. Another reconstructional problem is that small regions (maybe provide an example) such as details are sometimes left out, which could possibly distort class labels (object might start to resemble each other less if they lose detail).

![latent_representation](https://github.com/rien333/numbersense-vae/blob/master/thesis/bird_variedCP.png) "Reconstructions of the image in the top-left made by slightly increasing the response value of the VAE's latent representation $z$, at different individual dimensions $z_i$ Some of these dimensions give you a slight idea of what they encode (e.g. a light source at a location)."


## Subitizing Read-Out ##

<!-- TODO
    - [ ] Mention something about what other alogithms do -->

Accuracy of the `zclassifier` (i.e. the classifier as described in [**section x.x**](#classifierarch) concerned with classification of latent activation patterns to subitizing labels) is reported over the withheld SOS test set. We report best performances using a slightly different  VAE architecture than the one described in [section X.X](#vae) (scoring a mean accuracy of 40.4). The main difference between the VAE used in this experiment and the one that is used throughout the rest of this research is it places intermediate fully connected layers (with size 3096) between the latent representation and the convolutional stacks. Accuracy scores of other algorithms were copied over from @zhang2016salient. For their implementation, refer to @zhang2016salient.

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

The subitizing performance of the VAE is comparable to highest scoring non-machine learning algorithm, and performs worse overall than the CNNs trained by @zhang2016salient. This can be explained by a number of factors. First of all, the `CNN_ft` algorithm used by @zhang2016salient has been pretrained on a large, well tested and more varied dataset, namely ImageNet [@russakovsky2015imagenet], which contains $\approx 1300x$  more images. Additionally, their model is capable of more complex representations due its depth and the amount of modules it contains [the applied model from @szegedy2015going uses 22, compared to the 12 in our approach]. Moreover, all their algorithms are trained in a supervised manner, providing optimization algorithms such as stochastic gradient descent with a more directly guided optimization objective, an advantage over present research's unsupervised training setup. 

## Qualitative Analysis ##

<!-- TODO
    - [ ] Largely just write what you wrote to Tom + info on obj/background selection
        and then apparent inhibitory role of some neurons
-->

Artificial and biological neural populations concerned with visual numerosity support quantitative judgements invariant to object size and, conversely, some populations detect object size without responding to quantity, indicating a separate encoding scheme for both properties [@stoianov2012; @harvey2013topographic]. Analogously, we tested whether our VAE's latent representation contained dimensions $z_i$ encoding either one of these properties. To test this, we first created a dataset with synthetic examples containing $N$ objects ($N \in \lbrack 0, 4\rbrack$, with N uniformly distributed over the dataset) and corresponding cumulative area values $A$ that those N objects occupied (measured in pixels, with $A$ normally distributed over the dataset[^🍃]). The object overlap threshold was set to 1 for each example, to reduce noise induced by possible weak encoding capabilities and reasons outlined in [section X.X](#hybrid). As visualisations showed that each dimension $z_i$ encodes more than one type of visual feature (see figure Y.Y) special care was undertaken to reduce $z_i$'s response variance by only generating data with 15 randomly sampled objects from the object cut-out set, and one random background class from the background dataset (performance is reported on the "sand deserts" class, which contains particularly visually uniform examples). A dimension $z_i$ is said to be able to perform as either a numerical or area detector when regressing it's response over novel synthetic dataset ($n=33280$) supports the following relationship between normalized variables $A$ and $N$ [@stoianov2012]:

<!-- If you want you could make this tag automatic with "\begin{align}" or something -->
$$z_i = \beta_1 \log(N) + \beta_2\log(A) + \textepsilon \, (\textrm{with }\, N \in [0, 4]) \tag{1} $$
 
The regression was accomplished with linear regression algorithms taken from @newville2016lmfit (Levenberg–Marquardt proved best). The criteria set by @stoianov2012 for being a good fit of $\tag{1}$ are **(1)** the regression explaining at least 10% of the variance ($R^2 \geq 0.1$) **(2)** and a "ideal" detector of some property should have a low ($\mid\beta_i\mid < 0.1$) regression coefficient for the complementary property. We slightly altered criteria **(1)** to fit our training setup. The complexity of the SOS dataset in comparison to the binary images used by @stoianov2012 requires our model to encode a higher variety of information, meaning that any fit is going to have more noise as no dimension $z_i$ has one role (see figure Y.Y for an overview). Moreover, the synthetic data we use for the regression includes more complex information than the dataset used by @stoianov2012. Nevertheless, we still found a small number of reoccurring detectors of $A$ and $N$, with $R > 0.065 \pm 0.020$  (all $z_i$ with $R < 0.033$ resulted in an average  $R= 0.06 \pm 0.01$). Due to randomisation in the fitting process (synthetic examples are randomly generated at each run) the role distribution varied slightly with each properties being encoded by about 1-2 dimensions, out of the total of 182 (anymore would indicate an unlikely redundancy, given that the small latent space should provide an efficient encoding scheme). Some latent dimensions that provide a better fit of $\tag{1}$ exist, but don't satisfy criteria **(2)**. An interesting note is that whenever the regression showed multiple dimensions encoding area, they either exhibited positive or negative responses (i.e. positive or negative regression coefficients) to area increase, in accordance with how visual numerosity might rely on a size normalisation signal, according to some theories on the neurocomputational basis for numerosity [see @stoianov2012 for a discussion]. A large negative response (in contrast to a positive) to cumulative area might for example be combined with other response in the VAE's decoder network as an indicatory or inhibitory signal that the area density does not come from just one object, but from multiple.

Figure X.x (a) provide characteristic response profiles for dimensions encoding either cumulative area or a subitizing count. For the area dimensions, extreme cumulative area samples bend the mean distribution either upwards or downwards, while the response distribution to cumulative area for numerosity encoding dimensions stays relatively centered. For the numerosity dimension, (Figure X.x (b)) both the total response of and the center of the response distribution increased with numerosity. In contrast, the dimension that was sensitive shows a fairly static response to changes in subitizing count. With some extra time, the visual clarity and overall performance of this qualitative analysis could probably be greatly improved, given that only a short focus on reducing response variance increased $R$ by almost a factor of 10 in some cases.

<!-- Include more/better figures -->
![num_dim](https://github.com/rien333/numbersense-vae/blob/master/thesis/Nn-z77.png "")
![area_dim](https://github.com/rien333/numbersense-vae/blob/master/thesis/Na-z88.png "(a) shows a typical response profile for a numerosity detector ($R=0.055$).  Subitizing label $N$ was normalized. (b) shows a typical response profile of dimension that encodes cumulative area while being invariance to numerosity information ($R=0.056$). Cumulative area ($A$) was normalized and is displayed across a logarithmic scale. For visual convenience, examples with $A=0$ were shifted next to lowest value of A in the dataset." )

# Conclusion #

We described a setup for training a VAE on a subitizing task, while satisfying some important biological constraints. A possible consequence thereof is that our final model showcases properties also found in biological neural networks (as well as other artificial algorithms). Firstly, an ability to subitize emerged as an implicitly learned skill. Second of all, the learned encoding scheme indicates support for encoding numerosities without resorting to counting schemes relying to cumulative (objective) area, and conversely encodes cumulative area without using numerosity information, in accordance with previous (other?) comparable artificial models [@stoianov2012]. However, more research is needed to asses , as more properties of input need to be varied (such as visual variation and the distribution of variables ), there is room for improvement in the VAE's reconstructional abilities, i.e. efficiency of coding scheme. These two problems also indicate room for improvement in the subitizing classification task, which has the additional improvement of solving the class imbalance problem. Nevertheless, visual numerosity-like skills have emerged during the training of the VAE, showing the overall ability to perceive numerosity within the subitizing range without using information provided by visual features other than quantity. We can thus speak of a fairly abstract sense of number, as the qualitative analysis of the encoding yielded promising results over a large variation of images, whereby especially abstraction in regard to scale has been demonstrated.

# References #


[^🍃]: A uniform distribution of cumulative area might have worked better, but required algorithmic changes to the synthetic data generation process that were inhibited by the amount of time still available.
