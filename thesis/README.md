Introduction
============

(**Note:** the quality and the content of the text is not yet reflective
of whatever I have in mind for my thesis)

Although various machine learning approaches dealing with the numerical
determination of the amount of objects in images already exist, some
research can be impartial to broader cognitive debate, resulting in
models that are somewhat unhelpful for progressing the understanding of
numerical cognition. Any approach to computational modeling of numerical
cognition that aims to maintain biological plausibility should adhere to
neurological findings about the general characteristics of cognitive
processes related to numerical tasks. Essential to understanding the
cognitive processes behind number sense is their perceptual origin, for
so called *visual numerosity* has been posed as the fundamental basis
for developmentally later kinds of number sense, such as that required
for arithmetical thinking and other more rigorous concepts of number
found in mathematics (Lakoff and Núñez 2000, chap. 2; Piazza and Izard
2009). Visual numerosity is the perceptual capability of many organisms
to perceive a group of items as having either a distinct or approximate
cardinality. Some specific characteristics of visual numerosity can be
derived from it's neural basis. Nieder (2016) and Harvey et al. (2013)
present research where topologically organized neural populations
exhibiting response profiles that remained largely invariant to all
sensory modalities except quantity were discovered. Specifically, the
found topological ordering was such that aside from populations
responding to their own preferred numerosity, they also showed
progressively diminishing activation to preferred numerosities of
adjacent populations, in a somewhat bell-shaped fashion (Nieder 2016).
This correspondence between visual quantity and topological network
structure leads them to conclude that neural populations can directly
(i.e. without interposition of higher cognitive processes) encode
specific visual numerosities. This coding property of neurons
participating in visual numoristy was found in humans and other animals
alike (see Nieder 2016; Harvey et al. 2013).

Notwithstanding the success of previous biologically informed approaches
to modeling numerical cognition with artificial neural networks
(Stoianov and Zorzi 2012), more work is to be done in applying such
models to natural images. The main reason behind pursuing natural images
is improving the biological plausibility over previous approaches
relying on binary images containing only simple geometric shapes (for
examples, see Stoianov and Zorzi 2012; Wu, Zhang, and Du 2018), given
that natural images are closer to everyday sensations than binary
images. Furthermore, any dataset with uniform object categories does not
capture how visual number sense in animals is abstract in regard to the
perceived objects (Nieder 2016), implying that a model should be able
show that it performs equally well between objects of different visual
complexities. Another way in which biological plausibility will be
improved is by guiding some algorithmic decisions on the previously
described discovery of the direct involvement of certain neural
populations with numerosity perception. Moreover, the properties of
these neurons' encoding scheme provide interesting evaluation data to
the visual numerosity encoding scheme used by our final model. Some
important and closely related characteristics of visual numerosity that
constrain our approach, but hopefully improve the biological
plausibility of the final model are:

1.  Visual number sense is a purely *automatic* appreciation of the
    sensory world. It can be characterized as "sudden", or as visible at
    a glance (Dehaene 2011, p57; J. Zhang, Ma, et al. 2016).
    *Convolutional neural networks* (CNNs) not only showcase excellent
    performance in extracting visual features from complicated natural
    images (Mnih et al. 2015; Krizhevsky, Sutskever, and Hinton 2012;
    for visual number sense and CNNs see J. Zhang, Ma, et al. 2016), but
    are furthermore functionally inspired by the visual cortex of
    animals (specifically cats, see LeCun and Bengio 1995). CNNs
    mimicking aspects of the animal visual cortex thus make them an
    excellent candidate for modeling automatic neural coding by means of
    numerosity percepts.

2.  The directness of visual number entails that no interposition of
    external processes is required for numerosity perception, at least
    no other than lower-level sensory neural processes. More
    appropriately, the sudden character of visual number sense could be
    explained by it omitting higher cognitive processes, such as
    conscious representations (Dehaene 2011 indeed points to types of
    visual number sense being preattentive) or symbolic processing
    (visual numerosity is completely non-verbal, Nieder 2016).
    Furthermore, the existence of visual sense of number in human
    newborns (Lakoff and Núñez 2000), animals (Davis and Pérusse 1988)
    and cultures without exact counting systems (Dehaene 2011, p261;
    Franka et al. 2008) further strengthens the idea that specific kinds
    of sense of number do not require much mediation and can purely
    function as an interplay between perceptual capabilities and neural
    encoding schemes, given the aforementioned groups lack of facilities
    for abstract reasoning about number (see Everett 2005, 626; Lakoff
    and Núñez 2000, chap. 3, for a discussion on how cultural
    facilitaties such as fixed symbols and linguistic practices can
    facilitate the existence of discrete number in humans). Indeed,
    Harvey et al. (2013) show that earlier mentioned neural populations
    did not display their characteristic response profile when
    confronted with Arabic numerals, that is, symbolic representations
    of number. These populations thus show the ability to function
    seperately from higer-order representational facilities. Visual
    number sense being a immediate and purely perceptual process implies
    that our model should not apply external computational techniques
    often used in computer vision research on numerical determination
    task such as counting-by-detection (requiring both arithmetic and
    iterative attention to all group members, see J. Zhang, Ma, et al.
    2016, and @zhang2016unconstrained) or segmenting techniques (e.g.
    Chattopadhyay et al. 2016). Instead, we want to our model to operate
    in an autonomous and purely sensory fashion.

3.  Relatedly, visual sense of number is an emergent property of neurons
    embedded in generative hierarchical learning models, either
    artificial or biological (Stoianov and Zorzi 2012). The fact that
    visual number sense exist in animals and human newborns suggests
    that it is an implicitly learned skill learned at the neural level,
    for animals do not exhibit a lot of vertical learning, let alone
    human newborns having received much numerical training. Deemed as a
    generally unrealistic trope of artificial learning by AI critics
    (Dreyfus 2007) and research into the human learning process (Zorzi,
    Testolin, and Stoianov 2013a), modeling visual number necessitates
    non-researcher depended features. This will restrict the choice of
    algorithm to so called *unsupervised* learning algorithms, as such
    an algorithm will learn its own particular representation of the
    data distribution. Given their ability to infer the underlying
    stochastic representation of the data, thus performing autonomous
    feature determination, *Varitional Autoencoders* (VAEs) seem fit to
    tackle this problem (see [**section x.x**](#vae) for more detail).
    Moreover, VAEs are trained in an unsupervised manner, similar to how
    learning visual number sense, given appropriate circumstances, does
    not require labeled data due it being emergent. Another interesting
    aspect of VAEs is their relatively interpretable and overseeable
    learned feature space, which might tell us something about how it
    deals with visual numerosity, and thus allow us to evaluate the
    properties of the VAE's encoding against biological data.

Unfortunately, no dataset fit for visual numerosity estimation satisfied
above requirements (sizable collections of natural image with large and
varied, precisely labeled objects groups are hard to construct), forcing
present research towards *subitizing*, a type of visual number sense
which had a catered dataset readily available. Subitizing is the ability
of many animals to immediately perceive the number of items in a group
without resorting to counting or enumeration, given that the number of
items falls within the subitizing range of 1-4 (Kaufman et al. 1949;
Davis and Pérusse 1988) Most of the research above was conducted on
approximate numerical cognition, but the aforementioned characterisics
of visual sense of number hold equally well for a more distinct sense of
number such as subitizing. Similarly, subitizing is suggested to be a
parallel preattentive process in the visual system (Dehaene 2011, p57),
the visual system likely relying on it's ability to recognize holistic
patterns for a final subitizing count (Jansen et al. 2014; Dehaene 2011,
p57; Piazza et al. 2002). This means that the "sudden" character of
subitizing is caused by the visual system's ability to process simple
geometric configurations of objects in parallel, whereby increasing the
size of a group behind the subitizing range deprives perceiving this
group of it's sudden and distinct numerical perceptual character for
this would strain our parallelization capabilities too much. The
difference in perceptual character is due to a recourse to enumeration
techniques (and possibly others) whenever the subizting parallelization
threshold is exceeded, which differ from suddenness in being a
consciously guided (i.e. attentive) patterned type of activity.

Present research therefore asks: how can artificial neural networks be
applied to learning the emergent neural skill of subitizing in a manner
comparable to their biological equivalents? To answer this, we will
first highlight the details of our training procedure by describing a
dataset constructed for modeling subitizing and how we implementented
our VAE algorithm to learn a representation of this dataset. Next, as
the subitizing task is essentialy an image classification task, a
methodology for evaluating the unsupervised VAE model on subitizing
classification is described. We demonstrate that the performance of our
unsupervised approach is comparable with supervised approaches using
handcrafted features, although performance is still behind state of the
art supervised machine learning due to problems inherent to the
particulair VAE implementation. Finally, testing the final models
robustness to changes in visual features shows the emergence of a
property similar to biological neuron, that is to say, the encoding
scheme specifically supports particulair numerosity percepts invariant
to visual features other than quantity.

Related Work
============

Visual Number Sense
-------------------

<!-- Probably skip, but the arxiv paper and stoianov2012 can be reharsed -->
<!-- Investigate the goodfellow2016deep reference as to why it is somewhat computationally expensive -->
As previously described Stoianov and Zorzi (2012) applied artificial
neural netwoks to visual numerosity estimation, although without using
natural images. They discoverd that some resultant neural populations
concerned with numerosity estimation shared multiple properties with
biological populations participating in similar tasks, most prominently
an encoding scheme that was invariant to the cumaltative surface area of
the objects present in the provided images. Present research hopes to
discover a similar kind of invarience to surface area. Likewise, we will
employ the same scale invarience test, although a succesfull application
to natural images already shows a fairly abstract representation of
number, as the objects therein already contain varied visual features.

Some simplicity of the dataset used by Stoianov and Zorzi (2012) is due
their use of the relatively computationally expensive restricted
boltzmann machine (Goodfellow et al. 2016, vol. 1, chap. 20). Given
developments in generative algorithms and the availablity of more
computational power, we will therefore opt for a different algorithmic
approach (as discussed in the introduction) that will hopefully scale
better to natural images.

Salient Object Subitizing Dataset
---------------------------------

<!-- Also mention something about how this dataset is concustructed -->
<!-- Do you also explain synthetic data here? (yes seems alright) -->
<!-- Mention the results of their approach in some section -->
As seen in the [figure](#sub) below, the goal of the *Salient Object
Subitizing* (SOS) dataset as defined by J. Zhang, Ma, et al. (2016) is
to clearly show a number of salient objects that lies within the
subitizing range. As other approaches often perform poor on images with
complex backgrounds or with a large number of objects, J. Zhang, Ma, et
al. (2016) also introduce images with no salient objects, as well as
images where the number of salient objects lies outside of the
subitizing range (labeled as "4+"). The dataset was constructed from an
ensemble of other datasets to avoid potential dataset bias, and contains
approximately 14K natural images (J. Zhang, Ma, et al. 2016).

![sos\_example](https://github.com/rien333/numbersense-vae/blob/master/thesis/subitizing.png "Example images from the SOS dataset")

Methods
=======

Variational Autoencoder
-----------------------

                                              ┌───┐
                                              │ z⁰│
                                              │ . │
               ┌─────────────────┐            │ . │            ┌─────────────────┐
               │                 │            │ . │            │                 │
               │                 │            │ . │            │                 │

┏━━━━━━━━┓ │ │ │ . │ │ │ ┏━━━━━━━━┓ ┃●●●●●●●●┃ │ │ │ . │ │ │ ┃■■■■■■■■┃
┃●●●●●●●●┃────▶│ Encoder Network │───────────▶│ . │───────────▶│ Decoder
Network │────▶┃■■■■■■■■┃ ┃●●●●●●●●┃ │ │ │ . │ │ │ ┃■■■■■■■■┃ ┗━━━━━━━━┛
│ │ │ . │ │ │ ┗━━━━━━━━┛ X │ │ │ . │ │ │ X' │ │ │ . │ │ │
└─────────────────┘ │ . │ └─────────────────┘ │ zⁿ│ └───┘

                                  Latent vector z, with zⁱ ∈ R

VAEs are part of the family of autoencoder algorithms, owing this title
to the majority of their structure consisting of an encoder and a
decoder module (Doersch 2016) (see figure X for the schematics of an
autoencoder). In an regular autoencoder, the encoder module learns to
map features from data samples <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/27ac585e125a63ba0fb7a06816735b9e.svg?invert_in_darkmode" align=middle width=54.998006249999996pt height=22.648391699999998pt/> into latent
variables <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/b1b69f720063db1ed0b492cea440d19d.svg?invert_in_darkmode" align=middle width=51.99578669999999pt height=22.648391699999998pt/> often so that <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/7137b9b8600538350885d74d5ac21074.svg?invert_in_darkmode" align=middle width=49.87057679999999pt height=17.723762100000005pt/> and thus
performs in dimenstionality reduction, while the decoder function learns
to reconstruct latent variables <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.367621899999993pt height=14.15524440000002pt/> into <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/aaee0864ff46a34cc8c6dd45b2c64665.svg?invert_in_darkmode" align=middle width=54.99798809999999pt height=31.141535699999984pt/>
such that <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/a8696e2983fb53ac47995ac9596e2b5e.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=31.141535699999984pt/> matches <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=22.465723500000017pt/> according to some predefined similarity
measure (Liou et al. 2014). Reducing the input to be of much lower
dimenstionality forces the autoencoder to learn only the most emblematic
regularities of the data, as these will minimize the reconstrution
error. The latent space can thus be seen as an inferred hidden feature
representation of the data.

Where VAEs primarily differ from regular autoencoders is that rather
than directly coding data samples into some feature space, they learn
the parameters of a distribution that represents that feature space.
Therefore, VAEs perform stochastic inference of the underlying
distribution of the input data, instead of only creating some efficient
mappping to a lower dimenstionality that simultaneously facilitates
accurate reconstruction. Now provided with statistical knowledge of the
characteristics of the input, VAEs can not only perform reconstruction,
but also generate novel examples that are similar to the input data
based on the inferred statistics. The ability to generate novel examples
makes VAEs a *generative* algorithm.

The task of the encoder network in a VAE is to infer the mean and
variance parameters of a probability distribution of the latent space
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/594f2089f42b33713578c10ddb0f8d36.svg?invert_in_darkmode" align=middle width=133.40607719999997pt height=24.65753399999998pt/> such that samples <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.367621899999993pt height=14.15524440000002pt/>
drawn from this distribution facilitate reconstruction of <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=22.465723500000017pt/> (Doersch
2016). Novel sampled <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.367621899999993pt height=14.15524440000002pt/> vectors can then be fed into the decoder
network as usual. <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/c5c66b75f27ea3a5e8a20dfc22ab0394.svg?invert_in_darkmode" align=middle width=37.599025199999986pt height=24.65753399999998pt/> and <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/588e8c2c9441c1142fd70e76d1d5b7f1.svg?invert_in_darkmode" align=middle width=39.56627894999999pt height=24.65753399999998pt/> are constrained to roughly
follow a unit gaussian by minizing the Kullback-Leibler divergence
(denoted as <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/92a3cfa10cc205f69999b396a8a3be8c.svg?invert_in_darkmode" align=middle width=24.100592999999986pt height=22.465723500000017pt/>) between <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/6ba9ca2293d01d450671128ce5062746.svg?invert_in_darkmode" align=middle width=52.73634794999999pt height=24.65753399999998pt/> and
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/594f2089f42b33713578c10ddb0f8d36.svg?invert_in_darkmode" align=middle width=133.40607719999997pt height=24.65753399999998pt/>, where <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/92a3cfa10cc205f69999b396a8a3be8c.svg?invert_in_darkmode" align=middle width=24.100592999999986pt height=22.465723500000017pt/>
measures the distance between probability distribitions. Normally
distributed latent variables capture the intuition behind generative
algorithms that they should support sampling latent variables that
produce reconstructions that are merely *similair* to the input, and not
necessarily accurate copies. Furthermore, optimizing an abritary
distribution would be intractable, thus we need to rely on the fact that
given a set of normally distributed variables
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/74eb1332c3719400e7517a687c0a6cd8.svg?invert_in_darkmode" align=middle width=84.99981764999998pt height=22.465723500000017pt/> with <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/bb80d7867cd70ef2ba19bcdc87929a24.svg?invert_in_darkmode" align=middle width=51.25553069999999pt height=22.648391699999998pt/> and
any sufficiently complicated function <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/68ea884fbd70c05c5339170fabc350c8.svg?invert_in_darkmode" align=middle width=30.308325299999993pt height=24.65753399999998pt/> (such as a neural network),
there exists a mapping <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/7a1825b059f9c0700cce0667107cc8ff.svg?invert_in_darkmode" align=middle width=71.14121849999998pt height=31.141535699999984pt/> from which we can generate
any abritary distribution <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/b35d6f59a5e2434e1ab2e6def0bbb465.svg?invert_in_darkmode" align=middle width=80.75901899999998pt height=24.65753399999998pt/> with
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/0203b3b718dba73491715eeed753d9b0.svg?invert_in_darkmode" align=middle width=73.47586619999998pt height=31.141535699999984pt/> (Doersch 2016).

Therefore, the optimization objectives of a VAE become (see figure 4 of
Doersch 2016):

1.  <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/4a5d495a4a9a1140380700f4affe6e59.svg?invert_in_darkmode" align=middle width=208.26828659999998pt height=24.65753399999998pt/>
2.  Some reconstruction loss. Within visual problems, plain VAEs can for
    example minimize the binary cross entropy between $X$ and $\hat{X}$.

The first objective of generating an appropate distribution (research
into optimizing distribitions differently exists) function of the latent
spance makes VAEs generative, partly satisfying the third constraint
outlined in the introduction. To fully satisfy this constraint, the
final architecture uses deep neural networks for both the encoder and
decoder module (see Experiments VAE architecture), making the
implementation an hierarchical model. As an VAEs latent space encodes
the most important features of the data, it is hoped the samples drawn
from the encoder provide information regarding it's subitizing
performance (see [**section x.x**
implementing a VAE, refer to Kingma and Welling (2013) and Doersch
(2016).

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
been trained on ImageNet). (One notable downside of this approach is
that although layers from the VGG-19 represent important visual
information , it is well known fact that the first few layers (which
seem to work best in our own test, as well as in their original
application ) only encode simple features such as edges and lines (i.e
countours) which are combined into more complex features in deeper
layers. This means that the optimization is somewhat unambitious, in
that it will never try to learn any other visual features aside from
what the set of predfined layers <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/bb29cf3d0decad4c2df62b08fbcb2d23.svg?invert_in_darkmode" align=middle width=9.55577369999999pt height=22.831056599999986pt/> encode, such as detailed object
textures. Indeed, although countour reconstruction has greatly improved
with this loss function, reconstructed detail such as facial features
show less improvement.

Extending The Dataset With Syntethic Images
-------------------------------------------

Experiments
===========

Hidden Representation Classifier
--------------------------------

### Classifier architecture

To asses whether the learned latent space of the VAE network supports
subitizing/(the VAE) showcases the (emergent) ability to perform in
subitizing, a two layer fully-connected net is fed with latent
activation vectors <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/8ec2ef7749d65d10a01bcd8e313835d4.svg?invert_in_darkmode" align=middle width=22.81049759999999pt height=14.15524440000002pt/> created by the encoder module of the VAE
from an image <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/1338d1e5163ba5bc872f1411dd30b36a.svg?invert_in_darkmode" align=middle width=18.269651399999987pt height=22.465723500000017pt/>, and a corresponding subitizing class label <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/c503cd3cc90b9dc8646daa73c42365ae.svg?invert_in_darkmode" align=middle width=14.19429989999999pt height=22.465723500000017pt/>,
where <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/1338d1e5163ba5bc872f1411dd30b36a.svg?invert_in_darkmode" align=middle width=18.269651399999987pt height=22.465723500000017pt/> and <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/c503cd3cc90b9dc8646daa73c42365ae.svg?invert_in_darkmode" align=middle width=14.19429989999999pt height=22.465723500000017pt/> are respectively an image and class label from the
SOS training set. Both fully-connected layers contain 160 neurons. Each
of the linear layers is followed by a batch normalization layer (Ioffe
and Szegedy 2015), a ReLU activation function and a dropout layer,
respectively. A fully-connected net was choosen because using another
connectionist module for read-outs of the hidden representation
heightens the biological plausibility of the final approach (Zorzi,
Testolin, and Stoianov 2013b). Zorzi, Testolin, and Stoianov (2013b)
note that the appended connectionist classifier module can for example
be concieved of as cognitive response module (?), although the main
reason behind training this classifier is to asses it's performance
against other algorithmic and human data.

### Class imbalance

Class imbalance is a phenomenon encountered in datasets whereby the
number of instances belonging to at least one of the class is
significantly higher than the amount of instances belonging to any of
the other classes. Although there is no consensus on an exact definition
of what constitutes a dataset with class imbalance, we follow Fernández
et al. (2013) in that given over-represented class <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/e710976777475b48fa8cd4acca538748.svg?invert_in_darkmode" align=middle width=18.778654949999993pt height=14.15524440000002pt/> the number of
instances <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/a20a9547a59ab0b2b1981cdbae565bd3.svg?invert_in_darkmode" align=middle width=23.46794504999999pt height=22.465723500000017pt/> of one the classes <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/3bc6fc8b86b6c61889f4e572c7546b8e.svg?invert_in_darkmode" align=middle width=11.76470294999999pt height=14.15524440000002pt/> should satisfy
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/4bc2a2d82a13d9f7cd3aceb96521c510.svg?invert_in_darkmode" align=middle width=112.73481449999998pt height=22.465723500000017pt/> for a dataset to be considered imbalanced. For
the SOS dataset, <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/5e7857afed85618077f516aef1f83f5d.svg?invert_in_darkmode" align=middle width=81.11441909999999pt height=22.465723500000017pt/>, <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/ccf450d9f9b37675e9737077577750e4.svg?invert_in_darkmode" align=middle width=81.11441909999999pt height=22.465723500000017pt/>, <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/a5df93fc55d423f6f51cfbb403f3d6a7.svg?invert_in_darkmode" align=middle width=81.11441909999999pt height=22.465723500000017pt/>
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f39adc9940c0f9d49d251fce913d1a49.svg?invert_in_darkmode" align=middle width=81.11441909999999pt height=22.465723500000017pt/> and <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/c815afb96a24431ab4cdb4f3d4e2dd6e.svg?invert_in_darkmode" align=middle width=72.89520974999999pt height=22.465723500000017pt/>, which implies that <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/09d819a43c6e2990856e40dbda09f893.svg?invert_in_darkmode" align=middle width=13.666351049999989pt height=14.15524440000002pt/> and <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/988584bba6844388f07ea45b7132f61c.svg?invert_in_darkmode" align=middle width=13.666351049999989pt height=14.15524440000002pt/>
are *majority classes*, while the others should be considered *minority
classes*. Most literature makes a distinction between three general
algorithm-agnostic approaches that tackle class imbalance (for a
discussion, see Fernández et al. 2013). The first two rebalance the
class distribution by altering the amount of examples per class.
However, class imbalance can not only be conceived of in terms of
quantitative difference, but also as qualitative difference, whereby the
relative importance of some class is (weighted?) higher than others
(e.g. in classification relating to malignant tumors, misclassifying
malignant examples as nonmalignant could be weighted more strongly, see
([**???**]{.citeproc-not-found data-reference-id="ref"})).
<!-- mention the relevance of qualitative difference to SOS? -->

1.  *Oversampling techniques* are a particularly well performing set of
    solutions to class imbalance are . Oversampling alters the class
    distribution by producing more examples of the minority class, for
    example generating synthetic data that resembles minority examples
    (e.g. He et al. 2008; Chawla et al. 2002), resulting in a more
    balanced class distribution.
2.  *Undersampling techniques*. Undersampling balances the class
    distribution by discarding examples from the majority class.
    Elimination of majority class instances can for example ensue by
    removing those instances that are highly similar (e.g. Tomek 1976)
3.  *Cost sensitive techniques.*

An ensemble of techniques was used to tackle the class imbalance in the
SOS dataset. First, slight random under-sampling with replacement of the
two majority classes (<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/09d819a43c6e2990856e40dbda09f893.svg?invert_in_darkmode" align=middle width=13.666351049999989pt height=14.15524440000002pt/> and <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/988584bba6844388f07ea45b7132f61c.svg?invert_in_darkmode" align=middle width=13.666351049999989pt height=14.15524440000002pt/>) is performed (see Lemaître,
Nogueira, and Aridas 2017), reducing their size by \~10%. Furthermore,
as in practice many common sophisticated under- and oversampling
techniques (e.g. data augmentation or outlier removal, for an overview
see Fernández et al. (2013)) proved largely non-effective, a
cost-sensitive class weighting was applied. Cost-senstive ... consists
of .... The uneffectiveness of quantive sampling techniques is likely to
be caused by that in addition to the quantitative difference in class
examples, there is also a slight difficulty factor whereby assesing the
class of latent vector <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.367621899999993pt height=14.15524440000002pt/> is significantly if belongs to <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/e355414b8774603011922d600510b1df.svg?invert_in_darkmode" align=middle width=13.666351049999989pt height=14.15524440000002pt/> or <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/c2411b9678590eb9d75c90bfebb7c5c1.svg?invert_in_darkmode" align=middle width=13.666351049999989pt height=14.15524440000002pt/>
versus any other class, for these two classes require rather precise
contours to discern the invidual objects, in case they for example
overlapping, which remains hard for VAEs given their tendency to produce
blurred reconstructions. The classifier network therefore seems inclined
to put all of its representational power towards the easier classes, as
this will result in a lower total cost, whereby this inclination will
become even stronger as the quantitative class imbalance grows. The
class weights for cost sensitive learning are set according to the
quantitative class imbalance ratio ([**???**]{.citeproc-not-found
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

Hybrid Dataset
--------------

<!-- TODO
    - [ ] 
-->
Subitizing has been noted to become harder when objects are
superimposed, forcing recource to external processes as counting by
object enumeration (Dehaene 2011, p57.). Therefore, the object overlap
threshold is increased by <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/22f2e6fc19e491418d1ec4ee1ef94335.svg?invert_in_darkmode" align=middle width=21.00464354999999pt height=21.18721440000001pt/> starting from <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/cde2d598001a947a6afd044a43d15629.svg?invert_in_darkmode" align=middle width=21.00464354999999pt height=21.18721440000001pt/> for every object
added to an image, compared to J. Zhang, Ma, et al. (2016)'s the static
value of <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/cde2d598001a947a6afd044a43d15629.svg?invert_in_darkmode" align=middle width=21.00464354999999pt height=21.18721440000001pt/>, as VAEs have been noted to produce blurry
reconstructions, indicating a poor ability to code object edges, so
distorting class labels.

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
J. Zhang, Ma, et al. (2016).

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
trained by J. Zhang, Ma, et al. (2016). This can be explained by a
number of factors. First of all, the `CNN_ft` algorithm used by J.
Zhang, Ma, et al. (2016) has been pretrained on the large, well tested
databese of images it is trained on (i.e. ImageNet, which contains N
images, while our procedure uses M syntethic and N2 natural images).
Additionaly, their model is capable of more complex representations due
its depth and the amount of modules it contains (). Moreover, all their
alogirhtms are trained in a supervised manner, which can sometimes be a
lot easier than unsupervised training ([**???**]{.citeproc-not-found
data-reference-id="ref"})

Qualitive Analysis
------------------

Conclusion
==========

References
==========

<!--  LocalWords:  numbersections
 -->
Chattopadhyay, Prithvijit, Ramakrishna Vedantam, Ramprasaath R
Selvaraju, Dhruv Batra, and Devi Parikh. 2016. "Counting Everyday
Objects in Everyday Scenes." *arXiv Preprint arXiv:1604.03505*.

Chawla, Nitesh V, Kevin W Bowyer, Lawrence O Hall, and W Philip
Kegelmeyer. 2002. "SMOTE: Synthetic Minority over-Sampling Technique."
*Journal of Artificial Intelligence Research* 16: 321--57.

Davis, Hank, and Rachelle Pérusse. 1988. "Numerical Competence in
Animals: Definitional Issues, Current Evidence, and a New Research
Agenda." *Behavioral and Brain Sciences* 11 (4). Cambridge University
Press: 561--79.

Dehaene, Stanislas. 2011. *The Number Sense: How the Mind Creates
Mathematics*. OUP USA.

Doersch, Carl. 2016. "Tutorial on Variational Autoencoders." *arXiv
Preprint arXiv:1606.05908*.

Dreyfus, Hubert L. 2007. "Why Heideggerian Ai Failed and How Fixing It
Would Require Making It More Heideggerian." *Philosophical Psychology*
20 (2). Taylor & Francis: 247--68.

Everett, Daniel L. 2005. "Cultural Constraints on Grammar and Cognition
in Pirahã." *Current Anthropology* 46 (4). University of Chicago Press:
621--46. <https://doi.org/10.1086/431525>.

Fernández, Alberto, Victoria López, Mikel Galar, María José del Jesus,
and Francisco Herrera. 2013. "Analysing the Classification of Imbalanced
Data-Sets with Multiple Classes: Binarization Techniques and Ad-Hoc
Approaches." *Knowledge-Based Systems* 42 (April). Elsevier BV: 97--110.
<https://doi.org/10.1016/j.knosys.2013.01.018>.

Franka, Michael C, Daniel L Everettb, Evelina Fedorenkoa, and Edward
Gibsona. 2008. "Number as a Cognitive Technology: Evidence from Pirahã
Language and Cognitionq." *Cognition* 108: 819--24.

Goodfellow, Ian, Yoshua Bengio, Aaron Courville, and Yoshua Bengio.
2016. *Deep Learning*. Vol. 1. MIT press Cambridge.

Harvey, Ben M, Barrie P Klein, Natalia Petridou, and Serge O Dumoulin.
2013. "Topographic Representation of Numerosity in the Human Parietal
Cortex." *Science* 341 (6150). American Association for the Advancement
of Science: 1123--6.

He, Haibo, Yang Bai, Edwardo A Garcia, and Shutao Li. 2008. "ADASYN:
Adaptive Synthetic Sampling Approach for Imbalanced Learning." In
*Neural Networks, 2008. IJCNN 2008.(IEEE World Congress on Computational
Intelligence). IEEE International Joint Conference on*, 1322--8. IEEE.

Hou, Xianxu, Linlin Shen, Ke Sun, and Guoping Qiu. 2017. "Deep Feature
Consistent Variational Autoencoder." In *Applications of Computer Vision
(Wacv), 2017 Ieee Winter Conference on*, 1133--41. IEEE.

Ioffe, Sergey, and Christian Szegedy. 2015. "Batch Normalization:
Accelerating Deep Network Training by Reducing Internal Covariate
Shift." *arXiv Preprint arXiv:1502.03167*.

Jansen, Brenda RJ, Abe D Hofman, Marthe Straatemeier, Bianca MCW Bers,
Maartje EJ Raijmakers, and Han LJ Maas. 2014. "The Role of Pattern
Recognition in Children's Exact Enumeration of Small Numbers." *British
Journal of Developmental Psychology* 32 (2). Wiley Online Library:
178--94.

Kaufman, E. L., M. W. Lord, T. W. Reese, and J. Volkmann. 1949. "The
Discrimination of Visual Number." *The American Journal of Psychology*
62 (4). JSTOR: 498. <https://doi.org/10.2307/1418556>.

Kingma, Diederik P, and Max Welling. 2013. "Auto-Encoding Variational
Bayes." *arXiv Preprint arXiv:1312.6114*.

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E Hinton. 2012. "Imagenet
Classification with Deep Convolutional Neural Networks." In *Advances in
Neural Information Processing Systems*, 1097--1105.

Lakoff, George, and Rafael E Núñez. 2000. "Where Mathematics Comes from:
How the Embodied Mind Brings Mathematics into Being." *AMC* 10: 12.

LeCun, Yann, and et al. Bengio Yoshua. 1995. "Convolutional Networks for
Images, Speech, and Time Series." *The Handbook of Brain Theory and
Neural Networks* 3361 (10): 1995.

Lemaître, Guillaume, Fernando Nogueira, and Christos K. Aridas. 2017.
"Imbalanced-Learn: A Python Toolbox to Tackle the Curse of Imbalanced
Datasets in Machine Learning." *Journal of Machine Learning Research* 18
(17): 1--5. <http://jmlr.org/papers/v18/16-365>.

Liou, Cheng-Yuan, Wei-Chen Cheng, Jiun-Wei Liou, and Daw-Ran Liou. 2014.
"Autoencoder for Words." *Neurocomputing* 139. Elsevier: 84--96.

Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel
Veness, Marc G Bellemare, Alex Graves, et al. 2015. "Human-Level Control
Through Deep Reinforcement Learning." *Nature* 518 (7540). Nature
Publishing Group: 529.

Nieder, Andreas. 2016. "The Neuronal Code for Number." *Nature Reviews
Neuroscience* 17 (6). Springer Nature: 366--82.
<https://doi.org/10.1038/nrn.2016.40>.

Piazza, Manuela, and Véronique Izard. 2009. "How Humans Count:
Numerosity and the Parietal Cortex." *The Neuroscientist* 15 (3). Sage
Publications Sage CA: Los Angeles, CA: 261--73.

Piazza, Manuela, Andrea Mechelli, Brian Butterworth, and Cathy J Price.
2002. "Are Subitizing and Counting Implemented as Separate or
Functionally Overlapping Processes?" *Neuroimage* 15 (2). Elsevier:
435--46.

Stoianov, Ivilin, and Marco Zorzi. 2012. "Emergence of a'visual Number
Sense'in Hierarchical Generative Models." *Nature Neuroscience* 15 (2).
Nature Publishing Group: 194.

Tomek, Ivan. 1976. "Two Modifications of Cnn." *IEEE Trans. Systems, Man
and Cybernetics* 6: 769--72.

Wu, Xiaolin, Xi Zhang, and Jun Du. 2018. "Two Is Harder to Recognize
Than Tom: The Challenge of Visual Numerosity for Deep Learning." *arXiv
Preprint arXiv:1802.05160*.

Zhang, Jianming, Shuga Ma, Mehrnoosh Sameki, Stan Sclaroff, Margrit
Betke, Zhe Lin, Xiaohui Shen, Brian Price, and Radomír Měch. 2016.
"Salient Object Subitizing." *arXiv Preprint arXiv:1607.07525*.

Zhang, Jianming, Stan Sclaroff, Zhe Lin, Xiaohui Shen, Brian Price, and
Radomir Mech. 2016. "Unconstrained Salient Object Detection via Proposal
Subset Optimization." In *Proceedings of the Ieee Conference on Computer
Vision and Pattern Recognition*, 5733--42.

Zorzi, Marco, Alberto Testolin, and Ivilin P. Stoianov. 2013a. "Modeling
Language and Cognition with Deep Unsupervised Learning: A Tutorial
Overview." *Frontiers in Psychology* 4. Frontiers Media SA.
<https://doi.org/10.3389/fpsyg.2013.00515>.

Zorzi, Marco, Alberto Testolin, and Ivilin Peev Stoianov. 2013b.
"Modeling Language and Cognition with Deep Unsupervised Learning: A
Tutorial Overview." *Frontiers in Psychology* 4. Frontiers: 515.
