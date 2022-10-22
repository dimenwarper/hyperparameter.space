---
title: What if we just learn a language model for all of life?
date: 2022-10-22T10:24:09-07:00
draft: false
---

***What's the language of life?***

There’s something about current large language models that feels like a big paradigm shift not only in NLP but in many other domains. Generally,advances in modeling sequences and language permeate many different domains and get applied to all sorts of tasks. Back in their day hidden Markov models, stochastic context-free grammars, and conditional random fields trickled from NLP to a variety of applications with generally good success rates. But this time it feels very different. It’s not just the model or training technique that has made an impact, i.e. large transformers and the like, but the whole training framework of pre-training using self-supervision: train a big model to predict masked tokens then fine-tune, or, zero-shot your way into new solutions with simple prompting. The whole paradigm encourages domino effects, where many tasks downstream are solved with little extra effort as long as the pretrained model leverages massive data that covers many corner cases. The advantage is clear: the many tasks that you would want to answer with language (like question-and-answering) are in a way instances of masked-token prediction, so why not just solve that? The array of impressive feats current large language models can achieve in the natural language domai have taken us by surprise and make us wonder if the same magic bullet could be possible in other domains – like biomedicine

The applications of large language models in the biomedical domain have followed similar paths to previous NLP-driven revolutions: the techniques are mostly applied to protein/DNA/RNA sequences where data is vast and the models can be applied with no major modification to the underlying techniques. Before them, hidden markov models gave us protein families and conditional random fields gave us solutions to nucleic acid secondary structure among many other things. And sure enough, this time around we’ve gotten large language models for macromolecule sequences, the most popular being protein sequence models that already exhibit an impressive capacity for tackling downstream tasks like structure prediction and variant effect quantification despite being trained in a full self-supervised manner.

To be sure protein sequences, encoding billions of years of evolution, are formidable to model. But life is bigger than proteins. And DNA/RNA sequences. It is made of phenotypes as well. Of the on/off patterns that genes follow to perform their duties. Of interactions and feedback loops. Of sequences folding into structures and structures folding other structures. Life is a lot more than just our sequencing databases. 

And natural language is the same way! Natural language has poetry and blog posts and tables, and data. And somehow large language models can model the bulk of it. So why stop at sequences when applying these models to biology? Why not model *the whole thing*? All of life? Or at least every aspect that we know of, that we can measure. One might protest that there’s too much disparate data: matrices of gene expressions, graphs of interacting molecules, sequences, structures. And one would be right. Maybe what’s needed is a unifying language, one that encompasses everything, bottom’s up.

## What is the basis task in biology?

Masked pre-training is the basis task of language models in NLP, almost any other task can be reduced to masked-token prediction. So what’s the analogous “basis task(s)” for life sciences? As mentioned above, masked-token modeling of biological sequences is not enough as it fails to capture phenotypes and other downstreams as well the relationships between them, which arguably is the more important part. The same can be said with language models applied to small molecules and structures, they don’t capture higher order relationships.

Maybe to be more specific, we need to lay out the tasks that we want to encompass. Here’s one possible list, divided in discovery and design classes:

**Discovery**
* Predict phenotypes and behaviors (structure, gene expression, traits, disease) from genotype or lower order phenotypes (be it variants, nucleic acids, proteins)
* Establish causal links (drug -> disease in drug development, environmental effects -> disease in public health policy, genetic variant -> disease in diagnostics)
* Establish associations, e.g. protein-protein interactions

**Design**
* Design chemical matter targeting a specific protein/RNA/DNA/etc
* Design chemical matter with other specific properties (glowing/metabolism)
* Design pathways for chemical matter synthesis
* Design assays for measuring chemical matter activity

Note that none of these tasks can be done via sequence language modeling alone, though sequence models could be/is a very useful ingredient in their solution. For the sake of enumeration completeness, let’s brainstorm about what what data sources we have to solve these tasks:

* Sequence databases
* Structure databases (static structures and MD simulations)
* Small molecule databases
* Interaction and pathways graphs
* Gene expression databases
* Gene-phenotype databases
* Phylogenetic databases
* Biological theories, which are the glue or generators of the above data

Three main data types are represented in these sources: graphs, collections of 3D points, and sequences. Graphs connect sequences, 3D points, and possibly other graphs (e.g. viewing an interaction database through phylogenetic relationships), and of course the edges themselves need to be labeled with a pretty expansive vocabulary, possibly with additional quantitative traits (“is expressed in”, “inhibits”, “binds to”, etc, etc). Graphs are the central object, so our “basis task” of all biology should probably be something akin to graph prediction and generation: you start with a sequence or structure, or a matrix of gene expression and you aim to predict what relationships will this have with either existing data or possibly new data. The prompt here becomes a subgraph of your knowledge graph, where you think knowledge can be expanded. 

![](/img/llm_for_all_life/network_prediction.png#center)
*The basis task of all biology could be knowledge graph generation*

It is sort of a hypothesis generating machine, which can then be refined by active learning in a similar way that language models can be [“steered”](https://openai.com/blog/instruction-following/) with humans in the loop.Such automated hypothesis generation and testing loop has been attempted before in biomedicine, but generally to find relationships in limited domains and it’s mostly for association discovery (e.g. generating edges, not nodes).


## How to unify all biomedical primitives into one big language?

One decision we have to take to make this work is how to incorporate the different data primitives mentioned above. There are two choices we can make: 

### Multi-modality

We can take a page from image generation from language and grab language models for each data type: one for protein sequences, DNA, small molecules, etc. and put them in a context of a graph learner that connects them together. The graph learner can, for example, take learned representations of each data model and use it to generate the next possible knowledge subgraph graph, with other representations contained within them. The representations can then be decoded to finish generating the objects needed.

### Language unification

The alternative path is probably too crazy: describe a universal language of biomedicine, with all primitives. Train one big language model on all known corpora. This essentially has to be a “full knowledge graph” generating language, where the model has a knowledge graph context and then predicts the entities and relationships in the vicinity of the predicted entities. It might be too much for a single model to handle (just imagine if you somehow trained large language models in NLP to also predict sequences of bits in images…I mean, it’s still language no?). The only way this might be feasible is if there’s some smart way to unify the types within the nodes: maybe all molecules, large, small, and phenotypes can be thought of as 3D point clouds of some form. 

## How would we go about this?

In terms of assembling the data needed, it shouldn’t prove *that* difficult given that there are already highly curated datasets for protein-protein interactions, drug-molecule interactions, genetic association databases, etc. Certainly onerous and time-consuming, but doable. In terms of methods used we can take one of the two common themes for graph prediction and compression:
* Random walks: Train a transformer or similar architecture on random walks of the knowledge graph, masking nodes and edges along the way.
* Masked subgraph: Train a GNN to predict masked subgraphs from the graph. This can also be thought of a generalization of the above point.
* De novo generation: Use some generative model to generate subgraphs from scratch. For example a diffusion model whose diffusion process creates and deletes nodes/edges with some conditional probability.


![](/img/llm_for_all_life/network_generation_methods.png#center)
*Three possible routes for knowledge graph generation: (A) masked graph prediction through GNN, (B) masked node/edge prediction on random walks through a graph, and (C) de novo geneartion through e.g. diffusion processes like in this molecular graph example*

## Is there enough data?

My guess is that yes. Consider that all the data that is used to train protein language models would be a fraction of this dataset. If anything, I think the challenge would be the quadratic explosion of the edges between the node entities. For example, if we want to consider homology relationships as edges, we would end up having to connect millions of proteins, resulting in billions of edges, just for homology. And because proteins tend to cluster into families, many would connect to higher order phenotypes, exploding the number of edges further, so much of the work would be on how to actually represent such a massive graph.


## Would this be useful?

The main argument against generative models in biomedicine is: how do you know if they are correct, don’t you have to test the resulting hypothesis anyway? And experimental testing is the real bottleneck, not analysis. And this is generally correct if you are trying to nail some biological mechanism or push a drug all the way to the clinic. But there are genuine use cases that benefit from a little, low effort exploration. Sometimes you want to see if a molecule you generated/synthetized has a bad toxicity profile, what proteins does it interact with, etc etc. One typically achieves this using glue scripts that fetch data/models from servers, other databases, and trains a small model here or there. It’s actually a significant time sink of effort to do this, having a centralized smart language model where you can just say “this is my new molecule, this is my disease context, tell me what you think” would actually fit this bill pretty well. And if we can make it steerable all the better.

