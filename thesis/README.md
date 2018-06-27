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
    a glance (Dehaene 2011, 57; J. Zhang, Ma, et al. 2016).
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
    conscious representations (Dehaene 2011, 58 indeed points to types
    of visual number sense being preattentive) or symbolic processing
    (visual numerosity percepts are understood non-verbally,
    Nieder 2016). Furthermore, the existence of visual sense of number
    in human newborns (Lakoff and Núñez 2000, chap. 1), animals (Davis
    and Pérusse 1988) and cultures without exact counting systems
    (Dehaene 2011, 261; Franka et al. 2008) further strengthens the idea
    that specific kinds of sense of number do not require much mediation
    and can purely function as an interplay between perceptual
    capabilities and neural encoding schemes, given the aforementioned
    groups lack of facilities for abstract reasoning about number (see
    Everett 2005, 626; Lakoff and Núñez 2000, chap. 3, for a discussion
    on how cultural facilitaties such as fixed symbols and linguistic
    practices can facilitate the existence of discrete number in
    humans). Indeed, Harvey et al. (2013) show that earlier mentioned
    neural populations did not display their characteristic response
    profile when confronted with Arabic numerals, that is, symbolic
    representations of number. These populations thus show the ability
    to function seperately from higer-order representational facilities.
    Visual number sense being a immediate and purely perceptual process
    implies that our model should not apply external computational
    techniques often used in computer vision research on numerical
    determination task such as counting-by-detection (which requires
    both arithmetic and iterative attention to all group members, see J.
    Zhang, Ma, et al. 2016; J. Zhang, Sclaroff, et al. 2016) or
    segmenting techniques (e.g. Chattopadhyay et al. 2016). Instead, we
    want to our model to operate in an autonomous and purely sensory
    fashion.

<!-- I'm not sure if this holds in the biological case, but you could argue that "Comparable to how some research describes the brain as Bayesian machine [for a discussion see @bayesianbrain; @bayes2], VAEs learn a stochastic model of sensory input data by optimizing the parameters of a probability distribution such that the probability distribution maximizes the likelihood of the input (or "training") data." 🌸 Maybe after the blosssom: although not specifically true for numerosity learning, the brain is suggestively organized in a similair fashion as some research suggests ... 

Another solution would be ommiting or replacing the two constraints, by for example stating about possibility for complex organisation, and in the case of artificial neurons those embedded in hierarchically structured, generative learning models

"The generative basis of natural number concepts" vaguely supports the biological case for generative models and number sense (generation being some kind of innate skill or something)

(after biological:) under the probable assumption that the brain is also a preditive modeller/stochastic inferational device
-->
3.  Relatedly, visual sense of number is an emergent property of
    hierarchically organized neurons embedded in generative learning
    models, either artificial or biological (Stoianov and Zorzi 2012;
    the brain can be characterized as building a predictive modeler, or
    a "bayesian machine", Knill and Pouget 2004; Pezzulo and
    Cisek 2016). The fact that visual number sense exist in animals and
    human newborns suggests that it is an implicitly learned skill
    learned at the neural level, for animals do not exhibit a lot of
    vertical learning, let alone human newborns having received much
    numerical training. Deemed as a generally unrealistic trope of
    artificial learning by AI critics (Dreyfus 2007) and research into
    the human learning process (Zorzi, Testolin, and Stoianov 2013a),
    modeling visual number necessitates non-researcher depended
    features. This will restrict the choice of algorithm to so called
    *unsupervised* learning algorithms, as such an algorithm will learn
    its own particular representation of the data distribution. Given
    their ability to infer the underlying stochastic representation of
    the data, i.e. perform in autonomous feature determination,
    *Varitional Autoencoders* (VAEs) seem fit to tackle this problem
    ([**section x.x**](#vae) details their precise working). Moreover,
    VAEs are trained in an unsupervised manner similar to how, given
    appropriate circumstances, visual numoristy abilities are implictly
    learned skills that emerge without "labeled data". Another
    interesting aspect of VAEs is their relatively interpretable and
    overseeable learned feature space, which might tell something about
    how it deals with visual numerosity, and thus allows us to evaluate
    the properties of the VAE's encoding against biological data.

Unfortunately, no dataset fit for visual numerosity estimation task
similair to Stoianov and Zorzi (2012) satisfied above requirements
(sizable collections of natural image with large and varied, precisely
labeled objects groups are hard to construct), forcing present research
towards *subitizing*, a type of visual number sense which had a catered
dataset readily available. Subitizing is the ability of many animals to
immediately perceive the number of items in a group without resorting to
counting or enumeration, given that the number of items falls within the
subitizing range of 1-4 (Kaufman et al. 1949; Davis and Pérusse 1988)
Most of the research above was conducted on approximate numerical
cognition, but the aforementioned characterisics of visual sense of
number hold equally well for a more distinct sense of number such as
subitizing. Similarly, subitizing is suggested to be a parallel
preattentive process in the visual system (Dehaene 2011, 57), the visual
system likely relying on it's ability to recognize holistic patterns for
a final subitizing count (Jansen et al. 2014; Dehaene 2011, 57; Piazza
et al. 2002). This means that the "sudden" character of subitizing is
caused by the visual system's ability to process simple geometric
configurations of objects in parallel, whereby increasing the size of a
group behind the subitizing range deprives perceiving this group of it's
sudden and distinct numerical perceptual character for this would strain
our parallelization capabilities too much. The difference in perceptual
character is due to a recourse to enumeration techniques (and possibly
others) whenever the subizting parallelization threshold is exceeded,
which differ from suddenness in being a consciously guided
(i.e. attentive), patterned type of activity.

Present research therefore asks: how can artificial neural networks be
applied to learning the emergent neural skill of subitizing in a manner
comparable to their biological equivalents? To answer this, we will
first highlight the details of our training procedure by describing a
dataset constructed for modeling subitizing and how we implementented
our VAE algorithm to learn a representation of this dataset. Next, as
the subitizing task is essentialy an image classification task, a
methodology for evaluating the unsupervised VAE model's performance on
the subitizing classification task is described. We demonstrate that the
performance of our unsupervised approach is comparable with supervised
approaches using handcrafted features, although performance is still
behind state of the art supervised machine learning approaches due to
problems inherent to the particulair VAE implementation. Finally,
measuring the final models robustness to changes in visual features
shows the emergence of a property similar to biological neurons, that is
to say, the VAE's encoding scheme specifically supports particulair
numerosity percepts invariant to visual features other than quantity.

Related Work
============

<!-- TODO
    - [ ] Related work is often structured with line subsection, do that as it also saves space
-->
Visual Number Sense
-------------------

<!-- Probably skip, but the arxiv paper and stoianov2012 can be reharsed -->
<!-- TODO
    - [x] Investigate the goodfellow2016deep reference as to why it is somewhat computationally expensive
-->
As previously described, Stoianov and Zorzi (2012) applied artificial
neural netwoks to visual numerosity estimation, although without using
natural images. They discoverd neural populations concerned with
numerosity estimation that shared multiple properties with biological
populations participating in similar tasks, most prominently an encoding
scheme that was invariant to the cumaltative surface area of the objects
present in the provided images. Present research hopes to discover a
similar kind of invarience to surface area. Likewise, we will employ the
same scale invarience test, although a succesfull application to natural
images already shows a fairly abstract representation of number, as the
objects therein already contain varied visual features.

Some simplicity of the dataset used by Stoianov and Zorzi (2012) is due
their use of the relatively computationally expensive Restricted
Boltzmann Machine (RBM) (with the exception of exploiting prior
knowledge of regularities in the probability distribution over the
observered data, equation <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/40a3674c9358d0becab2b3c56aaeab67.svg?invert_in_darkmode" align=middle width=8.21920935pt height=14.15524440000002pt/> from Goodfellow et al. 2016 shows
that computational cost grows as a multiple of the size of hidden and
observered units in the RBM). Given developments in generative
algorithms and the availablity of more computational power, we will
therefore opt for a different algorithmic approach (see [section
X.X](vae)) that will hopefully scale better to natural images.

Salient Object Subitizing Dataset
---------------------------------

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

VAEs are part of the family of autoencoder algorithms, owing this title
to the majority of their structure consisting of an encoder and a
decoder module (Doersch 2016) (see figure X for the schematics of an
autoencoder). In an regular autoencoder, the encoder module learns to
map features from data samples <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/27ac585e125a63ba0fb7a06816735b9e.svg?invert_in_darkmode" align=middle width=54.998006249999996pt height=22.648391699999998pt/> into latent
variables <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/b1b69f720063db1ed0b492cea440d19d.svg?invert_in_darkmode" align=middle width=51.99578669999999pt height=22.648391699999998pt/> often so that <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/7137b9b8600538350885d74d5ac21074.svg?invert_in_darkmode" align=middle width=49.87057679999999pt height=17.723762100000005pt/> and thus
performs in dimenstionality reduction, while the decoder function learns
to reconstruct latent variables <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.367621899999993pt height=14.15524440000002pt/> into <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/0662449c28bd0449f7fc10e8c706e5d7.svg?invert_in_darkmode" align=middle width=59.609881649999984pt height=24.7161288pt/> such
that <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/deaa67ea7054ce0a18480a658535949d.svg?invert_in_darkmode" align=middle width=18.69862829999999pt height=24.7161288pt/> matches <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=22.465723500000017pt/> according to some predefined similarity measure
(Liou et al. 2014). Reducing the input to be of much lower
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
distribution would be intractable, thus VAEs need to rely on the fact
that given a set of normally distributed variables
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/6b04d182a4cc153c99e63cb11624e429.svg?invert_in_darkmode" align=middle width=102.26015249999999pt height=24.65753399999998pt/> with <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/bb80d7867cd70ef2ba19bcdc87929a24.svg?invert_in_darkmode" align=middle width=51.25553069999999pt height=22.648391699999998pt/> and
any sufficiently complicated function <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/fb5a9656c65930c54541391b5aa251d4.svg?invert_in_darkmode" align=middle width=35.78112284999999pt height=24.65753399999998pt/> (such as a neural
network), there exists a mapping <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/2b54f183e96b653f73efc4f4031291a2.svg?invert_in_darkmode" align=middle width=74.93116124999999pt height=24.7161288pt/> from which we can
generate any abritary distribution <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/b35d6f59a5e2434e1ab2e6def0bbb465.svg?invert_in_darkmode" align=middle width=80.75901899999998pt height=24.65753399999998pt/>
with <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/7b98e2bfe759ee7b24a3cf145ba3056f.svg?invert_in_darkmode" align=middle width=78.08775974999999pt height=24.7161288pt/> (Doersch 2016).

Therefore, the optimization objectives of a VAE become (also see figure
4 of Doersch 2016):

1.  <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/4a5d495a4a9a1140380700f4affe6e59.svg?invert_in_darkmode" align=middle width=208.26828659999998pt height=24.65753399999998pt/>
2.  Some reconstruction loss. Within visual problems, plain VAEs can for
    example minimize the binary cross entropy (BCE) between $X$ and
    $X'$.

Objective **(1)** grants VAEs the ability to generate new samples from
the learned distribution, partly satisfying the constraint outlined in
the introduction whereby visual numerisoty skills emerge in generative
learning models. To fully satisfy this constraint, the final
architecture uses deep neural networks for both the encoder and decoder
module (see figure X for the VAE architecture), making the
implementation an hierarchical model as well. As an VAEs latent space
encodes the most important features of the data, it is hoped the samples
drawn from the encoder provide information regarding it's subitizing
performance (see [**section x.x**](#readout)). For a complete overview
of implementing a VAE, refer to Kingma and Welling (2013) and Doersch
(2016).

Deep Feature Consistent Perceptual Loss
---------------------------------------

<!-- TODO
    - [ ] Comparison on the sos dataset (see compare_rnd.py)
    - [ ] Note something about alpha and beta? Our dataset worked better with alpha ~= 0.001 and beta, based on reported final reconstrution error, although more work in optimizing this ratio is to be done.
    - [x] Find a reference for the properties of the first few layers of a deep CNN
-->
<!-- This reason is a little strange. I think you need to focus more on how CNNs or say something about CNNs capturing more visual characterisics/human (visual) similarity judgemets in general -->
Because the frequently used pixel-by-pixel reconstrution loss measures
in VAEs do not necessarily comply with human perceptual similarity
judgements, Hou et al. (2017) propose optimizing the reconstructions
with help of the hidden layers of a pretrained deep CNN network, because
these models are particularly better at capturing spatial correlation
compared to pixel-by-pixel measurements (Hou et al. 2017). The ability
of the proposed *Feature Perceptual Loss* (FPL) to retain spatial
correlation should reduce the noted blurriness (Larsen et al. 2015) of
the VAE's reconstructions, which is especially problematic in subitizing
tasks since blurring merges objects which in turn distorts subitizing
labels. Hou et al. (2017) and present research employ VGG-19 (Simonyan
and Zisserman 2014) as the pretrained network <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/5e16cba094787c1a10e568c61c63a5fe.svg?invert_in_darkmode" align=middle width=11.87217899999999pt height=22.465723500000017pt/>, trained on the
large and varied ImageNet (Russakovsky et al. 2015) dataset. FPL
requires predefining a set of layers <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/63edf80c57e86e2888288fe255830df2.svg?invert_in_darkmode" align=middle width=41.656051799999986pt height=22.831056599999986pt/> from pretrained network
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/5e16cba094787c1a10e568c61c63a5fe.svg?invert_in_darkmode" align=middle width=11.87217899999999pt height=22.465723500000017pt/>, and works by minimizing the mean squared error (MSE) between the
hidden representations of input <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> and VAE reconstruction <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/33717a96ef162d4ca3780ca7d161f7ad.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=18.666631500000015pt/> at
every layer <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/bb29cf3d0decad4c2df62b08fbcb2d23.svg?invert_in_darkmode" align=middle width=9.55577369999999pt height=22.831056599999986pt/>. Aside from the <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/92a3cfa10cc205f69999b396a8a3be8c.svg?invert_in_darkmode" align=middle width=24.100592999999986pt height=22.465723500000017pt/>-divergence, the VAE's
second optimization objective is now as follows:

<!-- 
This but as a sum.
You could also just note two objectives, and so also introduce alpha and beta 
-->
<p align="center"><img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/1ce8859789816bd831df6a21198a44a3.svg?invert_in_darkmode" align=middle width=179.3380512pt height=37.90293045pt/></p>

The intuition behind FPL is that whatever some hidden layer <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/bb29cf3d0decad4c2df62b08fbcb2d23.svg?invert_in_darkmode" align=middle width=9.55577369999999pt height=22.831056599999986pt/> of the
VGG-19 network encodes should be retained in the reconstruction
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/33717a96ef162d4ca3780ca7d161f7ad.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=18.666631500000015pt/>, as the VGG-19 has proven to model important visual
characteristics of a large variety of image types. In Hou et al.
(2017)'s and our experiments
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/502cc1e9fcfec07b3e1241ce483069b0.svg?invert_in_darkmode" align=middle width=258.04564005pt height=24.65753399999998pt/> resulted in the best
reconstructions. One notable shortcoming of FPL is that although the
layers from VGG-19 represent important visual information, it is a known
fact that the first few layers of deep CNNs only encode simple features
such as edges and lines (i.e they support countours), which are only
combined into more complex features deeper into the network (Liu et al.
2017; FPL's authors Hou et al. 2017 note something similair). This means
that the optimization objective is somewhat unambitious, in that it will
never try to learn any other visual features (for examples, refer to Liu
et al. 2017, Fig. 6.) aside from what the set of predefined layers <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode" align=middle width=11.18724254999999pt height=22.465723500000017pt/>
represents. Indeed, although countour reconstruction has greatly
improved with FPL, the reconstruction of detail such as facial features
shows less improvement. Although Hou et al. (2017) show a successful
application of FPL, they might have been unaware of this shortcoming due
to using a more uniform dataset consisting only of centered faces. For a
comparison between FPL loss and BCE reconstruction loss see the figure
below.

Hybrid Dataset
--------------

<!-- TODO
    - [x] Hybrid dataset (intuition, settings graph etc, pretrain,)
    - [x] Synthetic image settings
    - [ ] Generate bezier graph
-->
We follow J. Zhang, Ma, et al. (2016) in pretraining our model with
synthetic images and later fine-tuning on the SOS dataset. However, some
small chances to their synthetic image pretraining setup are proposed.
First, the synthetic dataset is extened with natural images from the SOS
dataset such that the amount of examples per class per epoch is always
equal (hopefully reducing problems encountered with class imbalance, see
[section X.X](#imbalance)). Another reason for constructing a hybrid
dataset was the fact that the generation process of synthetic images was
noted to produes 1. fairly unrealistic looking examples and 2.
considerably less suitable than natural data for supporting subitizing
performance (J. Zhang, Ma, et al. 2016). A further intuition behind this
dataset is thus that the representation of the VAE must always be at
least a little supportive of natural images, instead of settling on some
optimum for synthetic images. A final reason for including natural
images is that any tested growth in dataset size during pretraining
resulted into lower losses. The ratio of natural to synthethic images is
increased over time, defined by a bezier curve with parameters
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f2c5b82fa8ce51e3945414af55db9555.svg?invert_in_darkmode" align=middle width=277.18409235pt height=21.18721440000001pt/> shown in figure X.X. We grow the
original data size by roughly 8 times, pretraining with a total of 80000
hybrid samples per epoch. Testing many different parameters for the
hybrid dataset was not given much priority as the total loss seemed to
shrink with dataset expansion and training and testing a full model was
time expensive.

<!-- Make this a pdf when exporting, maybe in the generate figures.fish thing -->
![bezier\_curve](https://github.com/rien333/numbersense-vae/blob/master/thesis/bezier.png "Bezier curve defining the ratio of natural images over syntethic images at time _t_")

Synthetic images are generated by pasting cutout objects from THUS10000
(Cheng et al. 2015) onto the SUN background dataset (Xiao et al. 2010).
The subitizing label is aquired by pasting an object <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/> times, with
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/7c16ba2fd920e93050add3e2649148bd.svg?invert_in_darkmode" align=middle width=67.967856pt height=24.65753399999998pt/>. For each paste, the object is transformed in
equivalent manner to J. Zhang, Ma, et al. (2016). However, subitizing is
noted be more difficult when objects are superimposed, forcing recource
to external processes as counting by object enumeration (Dehaene 2011,
57.), implying that significant paste object overlap should be avoided.
J. Zhang, Ma, et al. (2016) avoid object overlap by defining a threshold
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/d41f04792af6771ea47e0d7ce294d917.svg?invert_in_darkmode" align=middle width=54.34294964999999pt height=24.65753399999998pt/> whereby an object's visible pixels
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/4c4fc8a101c20a451986cbb2b60d50b8.svg?invert_in_darkmode" align=middle width=49.28961839999998pt height=22.465723500000017pt/> and total pixel amount <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/ebf0273cdafcc16c896a6ae57cb8c360.svg?invert_in_darkmode" align=middle width=38.328005099999984pt height=22.465723500000017pt/> should satisfy
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/3d58d741d8c4ef6d86a04c2bd5140d5b.svg?invert_in_darkmode" align=middle width=131.81812709999997pt height=22.465723500000017pt/>}. For reasons given above, we define the
object overlap threshold as <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/58ada7345cb70b6b52e95713ca680e10.svg?invert_in_darkmode" align=middle width=166.50200489999997pt height=32.256008400000006pt/> with
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/2b4091190736e955f50962736c8a4891.svg?invert_in_darkmode" align=middle width=67.967856pt height=24.65753399999998pt/> compared to J. Zhang, Ma, et al. (2016)'s
static <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f720bb0bc5c021946e95974e4ae6a230.svg?invert_in_darkmode" align=middle width=48.858371099999985pt height=21.18721440000001pt/>, as VAEs are espcially prone produce blurry
reconstructions (Hou et al. 2017; Larsen et al. 2015), which requires
extra care with overlapping objects as to not distort class labels.

Experiments
===========

Hidden Representation Classifier
--------------------------------

### Classifier architecture

<!-- Maybe or maybe not mention dropout probabilities (should be final tho) -->
<!-- TODO
    - [ ] Mention that the experiments used a slightly different architecture, with less conv. 
      filters and fully conntected layers appended before and after the latent represntation
          - so this is theoritical performance (not time avialable for extra test)
    - [ ] Mention  per class performance being 10-20% higher depending how we skew the imbalance
      ratio's and techniques
-->
To asses whether the learned latent space of the VAE showcases the
emergent ability to perform in subitizing, a two layer fully-connected
net is fed with latent activation vectors <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/8ec2ef7749d65d10a01bcd8e313835d4.svg?invert_in_darkmode" align=middle width=22.81049759999999pt height=14.15524440000002pt/> created by the
encoder module of the VAE from an image <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/1338d1e5163ba5bc872f1411dd30b36a.svg?invert_in_darkmode" align=middle width=18.269651399999987pt height=22.465723500000017pt/>, and a corresponding
subitizing class label <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/c503cd3cc90b9dc8646daa73c42365ae.svg?invert_in_darkmode" align=middle width=14.19429989999999pt height=22.465723500000017pt/>, where <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/1338d1e5163ba5bc872f1411dd30b36a.svg?invert_in_darkmode" align=middle width=18.269651399999987pt height=22.465723500000017pt/> and <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/c503cd3cc90b9dc8646daa73c42365ae.svg?invert_in_darkmode" align=middle width=14.19429989999999pt height=22.465723500000017pt/> are respectively an
image and class label from the SOS training set. Both fully-connected
layers contain 160 neurons. Each of the linear layers is followed by a
batch normalization layer (Ioffe and Szegedy 2015), a ReLU activation
function and a dropout layer (Srivastava et al. 2014) with dropout
probability <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> and <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.649225749999989pt height=14.15524440000002pt/> respectively. A fully-connected net was chosen
because using another connectionist module for read-outs of the hidden
representation heightens the biological plausibility of the final
approach (Zorzi, Testolin, and Stoianov 2013b). Additionaly, Zorzi,
Testolin, and Stoianov (2013b) note that the appended connectionist
classifier module be concieved of as a cognitive response module
supporting a particular behavioral task, although the main reason behind
training this classifier is to asses it's performance against other
algorithmic data.

### Class imbalance

Class imbalance is a phenomenon encountered in datasets whereby the
number of instances belonging to on or more classes is significantly
higher than the amount of instances belonging to any of the other
classes. Although there is no consensus on an exact definition of what
constitutes a dataset with class imbalance, we follow Fernández et al.
(2013) in that given over-represented class <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/e710976777475b48fa8cd4acca538748.svg?invert_in_darkmode" align=middle width=18.778654949999993pt height=14.15524440000002pt/> the number of
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
relative importance of some class is weighted higher than others
(e.g. in classification relating to malignant tumors, misclassifying
malignant examples as nonmalignant could be weighted stonger than other
misclassifications) Qualitive difference might be relevant to the SOS
dataset, because examples with overlapping (i.e. multiple) objects make
subitizing inherently more difficult (see section X.X (\#hybrid)), and
previous results on subitizing show that some classes are more difficult
to classify than others (J. Zhang, Ma, et al. 2016).

1.  *Oversampling techniques* are a particularly well performing set of
    solutions to class imbalance. Oversampling alters the class
    distribution by producing more examples of the minority class, for
    example generating synthetic data that resembles minority examples
    (e.g. He et al. 2008; Chawla et al. 2002), resulting in a more
    balanced class distribution.
2.  *Undersampling techniques*. Undersampling balances the class
    distribution by discarding examples from the majority class.
    Elimination of majority class instances can for example ensue by
    removing those instances that are highly similar (e.g. Tomek 1976)
3.  *Cost sensitive techniques.* Cost sensitive learning does not alter
    the distribution of class instances, but penalizes misclassification
    of certain classes. Cost sensitive techqniques are especially useful
    for dealing with minority classes that are inherently more difficult
    (or "costly") to correctly classify, as optimisation towards easier
    classes could minimize cost even in quantitatively balanced datasets
    if the easier classes for example require lesser representational
    resources of the learning model.

An ensemble of techniques was used to tackle the class imbalance in the
SOS dataset. First, slight random under-sampling with replacement of the
two majority classes (<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/09d819a43c6e2990856e40dbda09f893.svg?invert_in_darkmode" align=middle width=13.666351049999989pt height=14.15524440000002pt/> and <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/988584bba6844388f07ea45b7132f61c.svg?invert_in_darkmode" align=middle width=13.666351049999989pt height=14.15524440000002pt/>) is performed (see Lemaître,
Nogueira, and Aridas 2017), reducing their size by \~10%. Furthermore,
as in practice many common sophisticated under- and oversampling
techniques (e.g. data augmentation or outlier removal, for an overview
see Fernández et al. (2013)) proved largely non-effective, a
cost-sensitive class weighting was applied. The uneffectiveness of
quantive sampling techniques is likely to be caused by that in addition
to the quantitative difference in class examples, there is also a slight
difficulty factor whereby assesing the class of latent vector <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.367621899999993pt height=14.15524440000002pt/> is
significantly if belongs to <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/e355414b8774603011922d600510b1df.svg?invert_in_darkmode" align=middle width=13.666351049999989pt height=14.15524440000002pt/> or <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/c2411b9678590eb9d75c90bfebb7c5c1.svg?invert_in_darkmode" align=middle width=13.666351049999989pt height=14.15524440000002pt/> versus any other class, for
these two classes require rather precise contours to discern invidual
objects, even more so with overlapping objects, while precise contours
remain hard for VAEs given their tendency to produce blurred
reconstructions (Larsen et al. 2015). The classifier network therefore
seems inclined to put all of its representational power towards the
easier classes, as this will result in a lower total cost, whereby this
inclination will become even stronger as the quantitative class
imbalance grows. The class weights for cost sensitive learning are set
according to the quantitative class imbalance ratio (equivalent to
section 3.2 in Fernández et al. 2013), but better accuracy was obtained
by slightly altering the relative difference between the weights by
raising them to some power <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/>. In our experiments, <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/aa6905d780872f0007f642420d7a2d9c.svg?invert_in_darkmode" align=middle width=40.00371704999999pt height=21.18721440000001pt/> resulted in a
balance between high per class accuray scores and aforementioned scores
roughly following the same shape as in other algorithms, which hopefully
implies that the classifier is able to generalize in a manner comparable
to previous approaches. For the SOS dataset with random majority class
undersampling, if <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/4df892aafd70851e6daaeca903b9acf1.svg?invert_in_darkmode" align=middle width=43.65668669999999pt height=21.18721440000001pt/> the classifier accuracy for the majority
classes shrinks towards chance, and, interestingly, accuracy for the
minority classes becomes comparable to the state of the art machine
learning techqniques.

Results & Discussion
====================

Variational Autoencoder performance
-----------------------------------

<!-- TODO
    - [x] graphs of loss (eh niey zo boeied)
    - [ ] Mus example with all the dimensions
    - [ ] Discuss and show quality of representation (visual problem/dfc versus plain)
-->
After pretraining for X epochs, and fine-tuning on the SOS dataset for Y
epochs, we found that the loss of our VAE did not shrink anymore. For
the specific purpose of subitizing, we can see that using FPL loss is
benificial (indeed, that is what we found when comparing the two models
in the classification task described in [section X.X](#readout)) The
reconstructions of a plain VAE and a VAE that uses FPL as it's
reconstruction optimization objective are show in figure X.X. To get an
idea of what sort of properties the latent space of the VAE encodes,
refer to figure Z.z.

Although the reconstructions show increased quaility of contour
reconstrion, there are a few reoccuring visual disperaties between
original and reconstruction. First of, novel patterns often emerge in
the reconstructions, possibly caused by a implementational glitch, or a
considerable difference in tested datasets (FPL is frequently paired
with the celebdataset (Liu et al. 2015)). Datasets other than the SOS
dataset showed slightly better performance, indicating that the SOS
dataset is either to small, too varied or requires non standard tweaking
for FPL to work in it's current form. Most of the improvent in more
uniform datasets came from the fact that the VAE learned to create more
local patterns to give the appereance of uniformly colored regions, but
upon closer inspection placed pixels of colors in a grid such that they
gave the appereance of just one color, similair to how for example LED
screens function. Another reconstrional problem is that small regions
(maybe provide an example) such as details are sometimes left out, which
could possibly distort class labels (object might start to resemble each
other less if they lose detail).

![latent\_representation](https://github.com/rien333/numbersense-vae/blob/master/thesis/bird_varied.png)
"Reconstructions of the image in the top-left made by slightly
increasing the reponse value of the VAE's latent representation <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode" align=middle width=8.367621899999993pt height=14.15524440000002pt/>, at
different individual dimenions <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.29555249999999pt height=14.15524440000002pt/> Some of these dimensions give you a
slight idea of what they encode (e.g. a light source at a location)."

Subitizing Read-Out
-------------------

<!-- TODO
    - [ ] Mention something about what other alogithms do -->
Accuray of the `zclassifier` (i.e. the classifier as described in
[**section x.x**](#classifierarch) concerned with classifcation of
latent activation patterns to subitizing labels) is reported over the
witheld SOS test set. We report best performances using a slightly
different VAE architecture than the one described in [section X.X](#vae)
(scoring a mean accuray of 40.4). The main difference between the VAE
used in this experiment and the one that is used throughout the rest of
this research is it places intermedtiate fully connected layers (with
size 3096) between the latent representation and the convolutional
stacks. Accuracy scores of other algorithms were copied over from J.
Zhang, Ma, et al. (2016). For their implementation, refer to J. Zhang,
Ma, et al. (2016).

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
Zhang, Ma, et al. (2016) has been pretrained on a large, well tested and
more varied dataset, namely ImageNet (Russakovsky et al. 2015), which
contains <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/66f1f450f0c7a9025ad5fa60853b6051.svg?invert_in_darkmode" align=middle width=59.62335719999999pt height=21.18721440000001pt/> more images. Additionaly, their model is
capable of more complex representations due its depth and the amount of
modules it contains (the applied model from Szegedy et al. 2015 uses 22,
compared to the 12 in our approach). Moreover, all their alogirhtms are
trained in a supervised manner, providing optimization algorithms such
as stochastic gradient descent with a more directly guided optimization
objective, an advantage over present research's unsupervised training
setup.

Qualitive Analysis
------------------

<!-- TODO
    - [ ] Largely just write what you wrote to Tom + info on obj/background selection
        and then apparent inhibitory role of some neurons
-->
Artificial and biological neural populations concerned with visual
numerosity support quantitative judgements invariant to object size and,
conversely, some populations detect object size without responding to
quantity, indicating a seperate encoding scheme for both properties
(Stoianov and Zorzi 2012; Harvey et al. 2013). Analogously, we tested
wether our VAE's latent representation contained dimensions <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.29555249999999pt height=14.15524440000002pt/>
encoding either one of these properties. To test this, we first created
a dataset with synthetic examples containing <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/> objects
(<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f079a79d2483ab629ab646984b6a1a00.svg?invert_in_darkmode" align=middle width=67.967856pt height=24.65753399999998pt/>, with N uniformly distributed over the
dataset) and corresponding cumalative area values <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> that those N
objects occupied (measured in pixels, with <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> normally distributed over
the dataset[^1]). The object overlap threshold was set to 1 for each
example, to reduce noise induced by possible weak encoding capabilities
and reasons outlined in [section X.X](#hybrid). As visualisations showed
that each dimension <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.29555249999999pt height=14.15524440000002pt/> encodes more than one type of visual feature
(see figure Y.Y) special care was undertaken to reduce <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.29555249999999pt height=14.15524440000002pt/>'s response
varience by only generating data with 15 randomly sampled objects from
the object cut-out set, and one random background class from the
background dataset (performance is reported on the "sand deserts" class,
which contains particularly visually uniform examples). A dimension
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.29555249999999pt height=14.15524440000002pt/> is said to be able to perform as either a numerical or area
detector when regressing it's reponse over novel synthetic dataset
(<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/e3b0e5bb817a6d275abc4a9720726da9.svg?invert_in_darkmode" align=middle width=72.88055444999999pt height=21.18721440000001pt/>) supports the following relationship between normalized
variables <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> and <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/> (Stoianov and Zorzi 2012):

<!-- If you want you could make this tag automatic with "\begin{align}" or something -->
<p align="center"><img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/25927ca8c051e1831fdf4138c0bba2c8.svg?invert_in_darkmode" align=middle width=333.0826785pt height=16.438356pt/></p>

The regression was accomplised with linear regression algorithms taken
from (Newville et al. 2016) (Levenberg--Marquardt proved best). The
criteria set by Stoianov and Zorzi (2012) for being a good fit are
**(1)** the regression explaining at least 10% of the varience
(<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/689c953c229748bd489397076f23b867.svg?invert_in_darkmode" align=middle width=62.90520389999999pt height=26.76175259999998pt/>) **(2)** and a "ideal" detector of some property should
have a low (<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/ed447abaf96248c199e24b5fc45f761a.svg?invert_in_darkmode" align=middle width=71.39158124999999pt height=24.65753399999998pt/>) regression coefficient for the
complementary property. We slightly altered criteria **(1)** to fit our
training setup. The complexity of the SOS dataset in comparison to the
binary images used by Stoianov and Zorzi (2012) requires our model to
encode a higher variety of information, meaning that any fit is going to
have more noise as no dimension <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.29555249999999pt height=14.15524440000002pt/> has one role (see figure Y.Y for
an overview). Moreover, the syntethic data we use for the regression
includes more complex information than the dataset used by Stoianov and
Zorzi (2012). Nevertheless, we still found a small number of reoccuring
detectors of <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> and <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/>, with <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/ff1d203db819ac4e917233cceea0ee81.svg?invert_in_darkmode" align=middle width=129.50341469999998pt height=22.465723500000017pt/> (all <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/6af8e9329c416994c3690752bde99a7d.svg?invert_in_darkmode" align=middle width=12.29555249999999pt height=14.15524440000002pt/> with
<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/0b47087c766b52e6dc2e3d0c75fe067d.svg?invert_in_darkmode" align=middle width=71.96916209999999pt height=22.465723500000017pt/> resulted in an average <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/ef6fae64c77bebb97f843149d5586a18.svg?invert_in_darkmode" align=middle width=113.06499599999998pt height=22.465723500000017pt/>). Due to
randomisation in the fitting process (synthetic examples are randomly
generated at each run) the role distribution varied slighty with each
properties being encoded by about 1-2 dimensions, out of the total of
182 (anymore would indicate an unlikely reduncancy, given that the small
latent space should provide an efficient encoding scheme). Latent
dimensions that provide a better fit of <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/876af6dcdadde76f9725f4e40ba6d535.svg?invert_in_darkmode" align=middle width=8.21920935pt height=14.15524440000002pt/> exist, but don't
satisfy the low coefficient criteria. An interesting note is that
whenever the regression showed multiple dimensions encoding area, they
either exhibited positive or negative responses (i.e. positive or
negative regression coefficients) to area increase, in accordance with
how visual numerisoty migt rely on a size normalisation signal,
according to some theories on the neurocomputational basis for
numerisoty (see Stoianov and Zorzi 2012 for a discussion). A large
negative reponse (in constrast to a positive) to cumalative area might
for example be combined with other respones in the VAE's decoder network
as an indicatatory or inhibitory signal that the area density does not
come from just one object, but from multiple.

Figure X.x (a) provide characteristic reponse profiles for dimensions
encoding either cumalative area or a subitizing count. For the area
dimensions, extreme cumaltative area samples bend the mean distribition
either upwards or downwards, while the response distribition to
cumaltative area for numerosity encoding dimensions stays relatively
centered. (neuron 35 is not a numerisoty neuron btw) (neuron 88
transparent and neuron 77 transparent) For the numerosity dimension,
(Figure X.x (b)) both the total reponse of and the center of the
response distribution increased with numerisoty. In contrast, the
dimension that was sensitive shows a fairly static response to changes
in subizing count. With some extra time, the visual clarity and overall
performance of this qualitative analysis could probably be greatly
improved, given that only a short focus on reducing response varience
increased <img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode" align=middle width=12.60847334999999pt height=22.465723500000017pt/> by almost a factor of 10 in some cases.

<!-- Include more/better figures -->
<!-- (figure caption) -->
Figure X.X (a) shows a typical response profile for a numerosity
detector (<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/55adb01cbe8bf05b3f3be8b6be725c42.svg?invert_in_darkmode" align=middle width=71.96916209999999pt height=22.465723500000017pt/>). Subitizing label N was normalized. Figure X.x (b)
shows a typical response profile of dimension that encodes cumaltive
area while being invarience to numeriosity information (<img src="https://rawgit.com/rien333/numbersense-vae/master/svgs/d205b9bea34bcad42c2ea22970868dca.svg?invert_in_darkmode" align=middle width=71.96916209999999pt height=22.465723500000017pt/>).
Cumalative area (A) was normalized and is displayed across a logirithmic
scale. For visual convenience, examples with A=0 were shifted next to
lowest value of A in the dataset.

Conclusion
==========

We described a setup for training a VAE on a subitizing task, while
satisfying some important biological constraints. A possible consequence
thereof is that our final model showcases properties also found in
biological neural networks (as well as other artificial algorithms).
Firstly, an ability to subitize emerged as an implicitly learned skill.
Second of all, the learned encoding scheme indicates support for
encoding numerosities without resorting to counting schemes relying to
cumaltative (objective) area, and conversly encodes cumaltative area
without using numerisity information, in accordance with previous
(other?) comparables artificial models (Stoianov and Zorzi 2012).
However, more research is needed to asses , as more proteries of input
need to be varied (such as visual variation and the distribution of
variables ), there is room for improvement in the VAE's reconstrional
abilities, i.e. efficiency of coding scheme. These two problems also
indicate room for improvent in the subitizing classification task, which
has the additional improvement of solving the class imbalance problem.
Nevertheless, visual numerisoity-like skills have emerged during the
training of the VAE, showning the overall ability to percieve numerisoty
within the subitizing range without using information provided by visual
features other than quantitiy. We can thus speak of a fairly abstract
sense of number, as the qualitive analysis of the encoding yielded
promosing results over a large variation of images, whereby espcially
abstracttion in regard to scale has been demonstrated.

References
==========

Chattopadhyay, Prithvijit, Ramakrishna Vedantam, Ramprasaath R
Selvaraju, Dhruv Batra, and Devi Parikh. 2016. "Counting Everyday
Objects in Everyday Scenes." *arXiv Preprint arXiv:1604.03505*.

Chawla, Nitesh V, Kevin W Bowyer, Lawrence O Hall, and W Philip
Kegelmeyer. 2002. "SMOTE: Synthetic Minority over-Sampling Technique."
*Journal of Artificial Intelligence Research* 16: 321--57.

Cheng, Ming-Ming, Niloy J Mitra, Xiaolei Huang, Philip HS Torr, and
Shi-Min Hu. 2015. "Global Contrast Based Salient Region Detection."
*IEEE Transactions on Pattern Analysis and Machine Intelligence* 37 (3).
IEEE: 569--82.

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

Knill, David C, and Alexandre Pouget. 2004. "The Bayesian Brain: The
Role of Uncertainty in Neural Coding and Computation." *TRENDS in
Neurosciences* 27 (12). Elsevier: 712--19.

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E Hinton. 2012. "Imagenet
Classification with Deep Convolutional Neural Networks." In *Advances in
Neural Information Processing Systems*, 1097--1105.

Lakoff, George, and Rafael E Núñez. 2000. "Where Mathematics Comes from:
How the Embodied Mind Brings Mathematics into Being." *AMC* 10: 12.

Larsen, Anders Boesen Lindbo, Søren Kaae Sønderby, Hugo Larochelle, and
Ole Winther. 2015. "Autoencoding Beyond Pixels Using a Learned
Similarity Metric." *arXiv Preprint arXiv:1512.09300*.

LeCun, Yann, and et al. Bengio Yoshua. 1995. "Convolutional Networks for
Images, Speech, and Time Series." *The Handbook of Brain Theory and
Neural Networks* 3361 (10): 1995.

Lemaître, Guillaume, Fernando Nogueira, and Christos K. Aridas. 2017.
"Imbalanced-Learn: A Python Toolbox to Tackle the Curse of Imbalanced
Datasets in Machine Learning." *Journal of Machine Learning Research* 18
(17): 1--5. <http://jmlr.org/papers/v18/16-365>.

Liou, Cheng-Yuan, Wei-Chen Cheng, Jiun-Wei Liou, and Daw-Ran Liou. 2014.
"Autoencoder for Words." *Neurocomputing* 139. Elsevier: 84--96.

Liu, Mengchen, Jiaxin Shi, Zhen Li, Chongxuan Li, Jun Zhu, and Shixia
Liu. 2017. "Towards Better Analysis of Deep Convolutional Neural
Networks." *IEEE Transactions on Visualization and Computer Graphics* 23
(1). IEEE: 91--100.

Liu, Ziwei, Ping Luo, Xiaogang Wang, and Xiaoou Tang. 2015. "Deep
Learning Face Attributes in the Wild." In *Proceedings of the Ieee
International Conference on Computer Vision*, 3730--8.

Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel
Veness, Marc G Bellemare, Alex Graves, et al. 2015. "Human-Level Control
Through Deep Reinforcement Learning." *Nature* 518 (7540). Nature
Publishing Group: 529.

Newville, Matthew, Till Stensitzki, Daniel B Allen, Michal Rawlik,
Antonino Ingargiola, and Andrew Nelson. 2016. "LMFIT: Non-Linear
Least-Square Minimization and Curve-Fitting for Python." *Astrophysics
Source Code Library*.

Nieder, Andreas. 2016. "The Neuronal Code for Number." *Nature Reviews
Neuroscience* 17 (6). Springer Nature: 366--82.
<https://doi.org/10.1038/nrn.2016.40>.

Pezzulo, Giovanni, and Paul Cisek. 2016. "Navigating the Affordance
Landscape: Feedback Control as a Process Model of Behavior and
Cognition." *Trends in Cognitive Sciences* 20 (6). Elsevier: 414--24.

Piazza, Manuela, and Véronique Izard. 2009. "How Humans Count:
Numerosity and the Parietal Cortex." *The Neuroscientist* 15 (3). Sage
Publications Sage CA: Los Angeles, CA: 261--73.

Piazza, Manuela, Andrea Mechelli, Brian Butterworth, and Cathy J Price.
2002. "Are Subitizing and Counting Implemented as Separate or
Functionally Overlapping Processes?" *Neuroimage* 15 (2). Elsevier:
435--46.

Russakovsky, Olga, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh,
Sean Ma, Zhiheng Huang, et al. 2015. "Imagenet Large Scale Visual
Recognition Challenge." *International Journal of Computer Vision* 115
(3). Springer: 211--52.

Simonyan, Karen, and Andrew Zisserman. 2014. "Very Deep Convolutional
Networks for Large-Scale Image Recognition." *arXiv Preprint
arXiv:1409.1556*.

Srivastava, Nitish, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever,
and Ruslan Salakhutdinov. 2014. "Dropout: A Simple Way to Prevent Neural
Networks from Overfitting." *The Journal of Machine Learning Research*
15 (1). JMLR. org: 1929--58.

Stoianov, Ivilin, and Marco Zorzi. 2012. "Emergence of a'visual Number
Sense'in Hierarchical Generative Models." *Nature Neuroscience* 15 (2).
Nature Publishing Group: 194.

Szegedy, Christian, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich,
and others. 2015. "Going Deeper with Convolutions." In. Cvpr.

Tomek, Ivan. 1976. "Two Modifications of Cnn." *IEEE Trans. Systems, Man
and Cybernetics* 6: 769--72.

Wu, Xiaolin, Xi Zhang, and Jun Du. 2018. "Two Is Harder to Recognize
Than Tom: The Challenge of Visual Numerosity for Deep Learning." *arXiv
Preprint arXiv:1802.05160*.

Xiao, Jianxiong, James Hays, Krista A Ehinger, Aude Oliva, and Antonio
Torralba. 2010. "Sun Database: Large-Scale Scene Recognition from Abbey
to Zoo." In *Computer Vision and Pattern Recognition (Cvpr), 2010 Ieee
Conference on*, 3485--92. IEEE.

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

[^1]: A uniform distribution of cumalative area might have worked
    better, but required algorithmic changes to the synthethic data
    generation process that were inhibited by the amount of time still
    avialable.
