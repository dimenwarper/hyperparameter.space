---
title: "A survey of tokenization in different data domains"
date: 2023-09-14T12:48:45-07:00
draft: false 
---

***Featurization by any other name would self attend as sweet***


Even before taking the world by storm through LLMs, transformers have for some time now been explored in various different domains, albeit with specific architectural adaptations. Eventually – and even more so with the advent of LLMs – it started to become commonplace to fix much of the transformer architecture and put the onus of domain application almost entirely into data featurization itself, in a way reminiscent of how tabularization, xgboost, and friends took over much of the structured regression setups in any domain. This particular form of featurization is commonly known as tokenization: ways to split the data into sequences of interdependent chunks that resemble a “language” and that are readily self-attended by a transformer. Tokenization is attractive for many reasons, but perhaps the most salient ones I can think of is that (1) it can be seen both the distillation of the atomic building blocks of a domain of data as well as a unit of compute useful for downstream optimization, (2) tokenizing data to be consumed by an autoregressor piggybacks on the already massive investment in transformer-land, and (3) it allows the integration of multiple data domains without introducing too much architectural changes as the adapter is more in the data itself than in the model. 

So how are things all over data land getting tokenized? Let’s find out

## Images

NLP and CV have enjoyed a symbiotic relationship in their deep learning era and at one point NLP was trying to take cues from CV by borrowing different flavors of CNNs for language tasks. But in a reversal, after the success of transformers in language, images were one of the first domains to adopt the tokenization mantra. In [ViT](https://arxiv.org/pdf/2010.11929.pdf), images are split into patches which are embedded and then fed as a sequence with added positional embeddings to encode the relationship between patches. Interestingly, vanilla positional embeddings are generally sufficient even though this is technically 2D data. More complex 2D embeddings don’t seem to help with performance that much. 

There are a few parameters to play with when tokenizing images, with the most important one being patch size, which decides the “resolution” of how the model looks at an image: the lesser the patch size the finer grain the model can take a look at the image paying a quartical price in complexity (increasing both the number of tokens and the dimension of the attention matrix). Some clever tricks can be found around this but perhaps the best one given how low effort it is to implement is [Flexivit](https://arxiv.org/pdf/2212.08013.pdf), which feeds different patch resolutions to the same model, using bilinear interpolation to adapt embedding weights from different resolutions but ultimately sharing the model parameters for all patch sizes.


![](/img/tokenization/flexivit.png#center)
*FlexiViT strategy for combining different image patch sizes* 

## Audio

Relative to images and text, audio is a bit lagging behind in terms of generative fidelity in generative models. This is in great part because of how information dense the data is and in smaller part because of the relatively smaller sizes of datasets out there. Tokenization is not as straightforward either because of the many layers of complexity of the data: in a single stream of audio you have many layers of abstractions with absurdly long dependency contexts (when compared to images or text at least) and various features like modulation that encode pitch, amplitude, intonation, and semantic meaning are all mixed together. Because of this, you may have multiple models dedicated for multiple aspects of the sound wave. For example, you might use one model to get the overall acoustics and another to get more specialized meaning like speech content and speaker qualities. [AudioLM](https://arxiv.org/pdf/2108.06209.pdf) does exactly this, using [w2v-BERT](https://arxiv.org/abs/2108.06209) to compute tokens related to speech semantics and [SoundStream](https://arxiv.org/abs/2107.03312) to compute tokens describing the general acoustics. [MusicLM](https://arxiv.org/abs/2301.11325) goes even further than this by adding “conceptual” tokens from a MuLan model that is trained on music videos with text descriptions. Continual embeddings from all of these models are tokenized by using either residual vector quantization for acoustic encodings (SoundStream and [MuLan](https://arxiv.org/abs/2208.12415_) or k-means for semantic ones (w2v-BERT).


![](/img/tokenization/musiclm.png#center)
*MusicLM's tokenization strategy*

## Video

Effective video representations depend on encoding both the spatial (in-frame) and temporal (inter-frame) aspects of the video stream. Thus, if the spatial aspects are modeled correctly into tokens, a transformer can take care of the temporal aspects through its attention mechanism. Focusing on the spatial aspects to tokenize is what the popular [TokenLearner](https://openreview.net/pdf?id=z-l1kpDXs88) paper does, decomposing each frame into learnable spatial maps that weigh regions of each frame. A spatial pooling operation then combines such spatial maps into tokens, one for each map that can be fed into a [ViViT](https://arxiv.org/abs/2103.15691) video transformer. The number of maps used dictate the amount of information that is carried over for each frame, a bit like an “attentive resolution”. Interestingly, in the TokenLearner experiments, they find that not many tokens are required per frame, on the order of 8 to 16 seem to be enough for their datasets which are comprised of 224-256 square frames.
As with audio, video has much more information content that can be further exploited, and [co-tokenization strategies](https://arxiv.org/pdf/2208.00934.pdf) leverage TokenLearner to tokenize the same video a multiple space/time scales (by varying the spatial and temporal resolutions) and even adds some language tokens to increase the information content passed through the transformer. One thing I noted in these models is that sometimes the tokenization strategy is actually applied to every layer of the transformer, as opposed to tokenizing only at the beginning or at some intermediate layers, going out of their way to “fusing” the tokens and re-tokenizing them, presumably to capture more and more spatio-temporal patterns hierarchically.


![](/img/tokenization/tokenlearner.png#center)
*Tokenlearner focuses on spatial, intra-frame features*


## Point clouds

Encoding point clouds is all about capturing local geometries. PointBERT goes this route by forming “patches” of points and then discretizing them. To achieve this, it samples random points in the cloud using farthest point sampling to mostly cover it and then grabbing the k nearest neighbors for each. These small sub point clouds are projected into an embedding using a small [PointNet](http://stanford.edu/~rqi/pointnet/) and then these embeddings are concatenated and thrown into an discrete variational autoencoder or dVAE (they build their own custom one, using a point cloud autoencoder + the Gumbel-softmax trick to make it discrete) to get the tokens. What I like about this approach is how generic it is: for any structured object, you can divide it into chunks, project it into a small embedding, and discretize it via a dVAE. It is a very generalizable tokenization strategy! One curious glitch that sometimes surfaces using this recipe is that dissimilar tokens can encode similar patches and vice versa, making token space a distorted view of actual point space. To correct this, [McP-BERT](https://arxiv.org/abs/2207.13226) uses multi-choice tokens, which grabs the continuous, pre-Gumbel-softmax dVAE representations and calculates a general distribution (via a separate softmax) for each token. This probability distribution is then reweighed by outcome probabilities of a separate PointBERT instance, effectively using the relationships learned by the transformer to disentangle the token space. The result is a tokenization system that resembles point cloud space with higher fidelity.


![](/img/tokenization/pointbert.png#center)
*PointBERT has a very generalizable tokenization strategy* 

## Graphs

Most tokenization strategies I’ve seen for graphs are node-centric, and are generally of the flavor of encoding the node’s neighborhood at different degrees of separation. [NAGFormer’s](https://arxiv.org/pdf/2206.04910.pdf) Hop2Token for example takes a node feature and propagates it K times using the adjacency matrix of the graph, each time going to an extra layer of neighbors of the node. The features of these hops are then concatenated, projected and fed to a transformer. 


![](/img/tokenization/hop2token.png#center)


Directed graphs like knowledge graphs are similarly handled. In [Nodepiece](https://arxiv.org/pdf/2106.12144.pdf), a predefined set of nodes are used as anchors. These anchors are used as a compass of sorts for each node: the outgoing edge data and node data of the K nearest anchors are hashed and passed through an encoder.

![](/img/tokenization/nodepiece.png#center)
*Nodepiece uses predefined anchor nodes as waypoints to tokenize nodes*

Maybe it’s because I’ve only skimmed the literature here but it surprises me a bit that prior art in vectorizing networks is not widely used in these tokenization schemes. For example, I can think of vectorizing a node using node2vec or similar and then applying a dVAE to get some tokens, similar to how PointBERT does it with point clouds. Additionally, I haven't seen small graph tokenization where a whole graph is tokenized, though this use case is mostly for molecular ML which already has its own tokenization universe; and speaking of which...

## Molecules

Molecules of all types have been tokenized in all sorts of ways for a long time even before transformers were discovered. In addition to the standard residue representations for proteins and nucleic acids, small molecule structure can be represented using the widely used SMILES, SELFIES, SMARTS, and derivatives. More complex molecular forms like metalo-organic frameworks simply extend SMILES with additional inter molecular structure languages like in the case of [MOFid](https://pubs.acs.org/doi/full/10.1021/acs.cgd.9b01050) or give simple textual descriptions like in [CatBERTa](https://arxiv.org/pdf/2309.00563.pdf) for catalyst screening. Because textual representations of molecules were already so pervasive, molecules were one of the first applications of language models, ChemBERTA is already 3 years old! Beyond small molecules, there is still ongoing work in tokenizing molecular views such as protein structures. [Foldseek](https://www.nature.com/articles/s41587-023-01773-0) for example introduced the 3Di language for protein 3D structure clustering coupled segments of itneracting backbone configurations into an “alphabet” which can be use to roughly describe the topology of a protein fold. While Foldseek uses this to enable ultra fast protein structure searches, it has also been leveraged by [ProsT5](https://www.biorxiv.org/content/10.1101/2023.07.23.550085v1) to model joint sequence and 3D structure of a large number of proteins. It remains to be seen if a more complex tokenization scheme that captures protein folds in higher resolution (e.g. using an approach similar to the point cloud strategies above) would yield better results. Arguably, however, molecular sequences already encode a non-trivial amount of structure, as ESMfold has shown.


![](/img/tokenization/3di.png#center)
*The 3Di alphabet for protein structure. Each "letter" is a structural state two fragments made up of three aminoacids, learned from the protein structure universe*

## Time series and tabular data

Time series and tabular data always seem to me, paradoxically, like a very niche deep learning application. Probably because there are already simple methods that model these data pretty well and for which deep learning doesn’t bring as much to the table except maybe scalability. Because of this, there isn’t much work on exploring a “language model for time series” since you could just use a transformer as is and feed it the series without needing to tokenize it (though using a big foundation model to model all the time series universe does sound like an interesting proposition). Nevertheless, there is one old paper I could find on exploring “tokenization” of time series. The method, [SAX](https://www.cs.ucr.edu/~eamonn/SAX.pdf), basically decomposes the series into step changes which are expressed as a combination of discrete steps at binned intervals. It seems that in its day it was used to enable clustering and other algorithms at a large scale in streaming data, though those applications are probably already enabled as-is by simply having more compute than 20 years ago.


![](/img/tokenization/sax.png#center)
*SAX combines step functions to discretize a timeseries*


For tabular data, the story is similar. [TAPAS](https://aclanthology.org/2020.acl-main.398.pdf) for example uses BERT directly to do Q&A on tables simply by flattening the table. There is however, an interesting special case of tabular data: gene expression matrices, which are *samples* X *genes* matrices with an intensity value in each cell. Gene expression distributions in these matrices can be rewritten as ranked lists of genes, where the gene with the highest expression comes first, then the next, and so forth. Each row in the matrix is then converted to rank ordered sequences of gene tokens and a language model can be used to discern patterns of expression. This is the basis of the recent wave of gene expression foundation models like [Geneformer](https://www.nature.com/articles/s41586-023-06139-9) that can further be enriched with more data beyond the values of the gene expression matrix (like other assay data or gene-specific information). The flexibility of adding more data into the mix, as well as mixing and matching modalities, might be an argument to take tabular data as a language modeling task seriously.

