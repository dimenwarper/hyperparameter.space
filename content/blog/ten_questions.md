+++
date = "2020-07-24T14:32:16-08:00"
draft = false
title = "How do we test our molecular understanding something in biology? From ten commandments to ten questions"

+++

I recently read an interesting entry in the nnil? blog trying to frame our understanding of biology by asking several key questions. The questions were derived from Tinberg's four questions, which mostly focus on animal behavior. While I've dabbled in biology for some time, I've mostly been focused studying molecular mechanisms. My world is that of genes, enzymes, nucleic acids, and maybe cells. What are the questions we ask then when we want to test our understanding of molecular machines and how they relate to higher order phenotypes? When searching the literature for hints of these questions, I was reminded of a set of ten precepts that the legendary Arthur Kornberg published in his article "Ten Commandments: Lessons from the Enzymology of DNA Replication". Kornberg's ten commandments are a mix of recommendations of empirical frameworks that have been successful time and time again as well as warnings of the limits of intuition and rationalization when dealing with entities (like enzymes) that we can only interrogate and probe under limited contexts. Crucially, they are the distilled thoughts of the origin of modern biochemistry and perfectly capture the zeitgeist of the molecular thinking that is still prevalent today. Because of this, Kornberg's commandments could be cast as questions, which could help probe the limits of our molecular understanding of phenomena. Here's my take on such task.

### What is the minimal system that maintains function?
*Commandment: RELY ON ENZYMOLOGY TO CLARIFY BIOLOGIC QUESTIONS*

In the first commandment, Kornberg says that the first goal of studying an any enzyme is to extract it from the cell and probe it preferably on a cell-free system, so it may be tested under many controlled conditions. In other words, to reconstitute is to understand. Be it in vitro, in vivo, or in silico, the extent in which we can isolate and probe a biological system demarcates very well how we understand it-- if the function of interest cannot be reconstituted, then the system is missing a piece, or the function that we think the system performs is a subset of a much bigger or multitude of roles.

### How many ways can we explain the data?
*Commandment: DO NOT BELIEVE SOMETHING BECAUSE YOU CAN EXPLAIN IT*

Perhaps the most hazardous mental trap when studying biological systems is that it is many times very easy to come up with explanations of why something works. Indeed, we can and do create lengthy stories that _make complete sense_ to explain the data at hand. Kornberg recalls how he was prey to this great garden of forking explanations when he missed the discovery of RNA polymerase when he and his collaborators explained the activity they saw on an ATP/ADP incorporation assay to be the result of a phosphorylase. The number of models that make biological sense and explain the data as well as the number of those models that can be further falsified provides a good measure of how well we understand the system.

### How reliable are the measurements from which we infer mechanism?
*Commandment: DO NOT WASTE CLEAN THINKING ON DIRTY ENZYMES*

One of the most crucial questions in biochemistry when interpreting results is "how pure is it?". Whether it be whole cell extracts to multiple fractionations of one specific molecule, the quality of the conclusions and even the insights that can be hoped to obtain will depend on the answer. Similarly in other settings, different qualities of meausurements will provide different resolutions of insights, be it analyzing bulk measurements compared to single cell or extrapolating conclusions from a generic cell line vs a specific one with an engineered genetic background.

### How reliable are the measurements which we use to test predictions?
*Commandment: DO NOT WASTE CLEAN ENZYMES ON DIRTY SUBSTRATES*

Related to the commandment and question above, after we are set with a mechanism and want to test it, it is sensible to ask if we can rely on the measurements to test our predictions. While many times we could use the same experimental framework where we inferred the mechanism to make our predictions, as we test a model in increasing generality we will switch experimental approaches as well (use different measurement types [proteomics vs transcriptomics], use an in vivo model, etc.). Each of these approaches will have limitations that will narrow the conclusions that we can make when testing predictions.

### What other minimal systems in nature replicate the function?
*Commandment: DEPEND ON VIRUSES TO OPEN WINDOWS*

Phages were an important tool in Kornberg's work to understand how DNA and RNA were synthesized, primed, and replicated. After all, viruses are minimal replication machines: a small strand of nucleic acids surrounded by the minimal proteins to protect it and bootstrap their replication. If a function is biologically important, it will likely be extracted, repurposed, or duplicated elsewhere. These naturally occurring minimal systems provide smoking gun evidence for suspected mechanisms.

### Under what biological conditions has a minimal system been tested?
*Commandment: CORRECT FOR EXTRACT DILUTION WITH MOLECULAR CROWDING*

One of the banes of (cell free) in vitro work in biochemistry is (the lack of) crowding. Enzymes function under a very crowded environmnet where a myriad of other reactions are happening and local concentrations of substrates can be sky high. Kornberg recalls how he was stumped for 10 years attempting to replicate an intact chromosome...until he added PEG to the mixture, making it more 'cell-like'. Similarly, the conclusions extracted from probing function in any minimal system will depend on how close the conditions are to the actual biological context. How similar is that cell line to the actual tumor? What naturally occurring molecular partners are missing in that single-molecule experiment?

### Does our understanding of the mechanism include its dynamics?
*Commandment: RESPECT THE PERSONALITY OF DNA*

If there is one pattern of understanding that repeats itself over and over in molecular biology, it's that everything is far more dynamical than we realized. As Kornberg puts it "DNA was regarded as a rigid rod devoid of personality and plasticity...Then we came to realize that the shape of DNA is dynamic in ways essential for its multiple functions...Especially noteworthy is breathing, the transient thermodynamic-driven opening (melting) of the duplex that facilitates the binding of specific proteins such as the helicase responsible for priming and the onset of replication.". In the same way, single-cell measurements have revealed the twisted transcriptional paths that cells undergo from one cell state to the next, revealing a continuum rather than discrete cell states. In the most desired end, full understanding of a biological function will be fully distilled in the differential equations of its dynamics.

### How is our mechanism altered or ablated genetically? 
*Commandment: USE REVERSE GENETICS AND GENOMICS*

The ability to go from enzyme function to protein sequence to gene and then probing the enzymes function by changing the underlying gene was something magical in Kornberg's eyes. Indeed, being able to predict how genetic alterations in the system's components affects function is an ultimate test of understanding it. Molecular tools have advanced considerably in the past decade and we now have CRISPR and friends to edit genes as we please to understand downstream function. Even in more challenging contexts, large-scale efforts to collate genetic and phenotypic data, ranging from thousands of genomic and metagenomic sequencing projects (metagenomes, microbiomes, you name it!), and clinical biobanks provide a solid foundation to test hypothesis on almost any organism that we know. 

### What new questions does our mechanism allow to us to answer?
*Commandment: EMPLOY ENZYMES AS UNIQUE REAGENTS*

In the last commandment, Kornberg lays out why modern biochemistry became so powerful: with each new enzyme discovered, new functions become possible, and avenues for discovering new enzymes open up. In a virtuous feedback loop, enzymes aid discovering other enzymes that yield new reactions to be repurposed for engineering. Understanding our mechanism in isolation is only part of the journey. Rather, grasping how it fits in the grand scheme of things, what other questions it allows us to answer, and better yet, how can it be repurposed, engineered, and tweaked is the true mark of the maturity of understanding. 



If evolution provides the main pillars for understanding in biology, enzymology stands as the arcs that hold the roof. As Kornberg puts it "time and again, spontaneous reactions, such as the melting of DNA and the folding of proteins, are found to be driven and directed by enzymes; in the case of DNA, its melting in a cell is catalyzed by several different helicases". 
