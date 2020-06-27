+++
date = "2019-07-05T14:32:16-08:00"
draft = false
title = "Sense and sensitivity (and specificty and utility)"
+++

<!-- Fonts to support Material Design -->
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap" />
<!-- Icons to support Material Design -->
<link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons" />

A recent tweet from Ash Jogalekar got me thinking.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">List of compounds medicinal chemists wouldn&#39;t have bothered to pursue because they didn&#39;t fit &quot;intuition&quot; about &quot;druglike&quot; rules<br><br>Aspirin<br>Metformin ($400M revenue)<br>Cyclosporin (&gt;$1B)<br>Dimethyl fumarate (&gt;$4B)<br><br>In drug discovery, there will always be enough exceptions to the rules</p>&mdash; Ash Jogalekar (@curiouswavefn) <a href="https://twitter.com/curiouswavefn/status/1143267155967766528?ref_src=twsrc%5Etfw">June 24, 2019</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

Translating it to more 'machine-learning-ish' language this means that the problem of predicting ultimately successful drug candidates is the pathological case where the cost of your predictions is really high and the rewards are higher but concentrated in a space defined by unmeasured covariates (a molecule might be a terrible drug for one indication but a really good drug for another and the space of possible/probable indications is vast and for the most part unknown). To make
matters worse, it is also many times an out-of-distribution problem. Take for example natural products -- which can be broadly defined as any chemical compound found in nature. They comprise more than half of the FDA-approved drugs and yet there are many examples where evolution has made them so chemically weird that there's no way of telling if they'll make a good drug. Keith Robison, in response to Ash's tweet, illustrates this case. 

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">3/4 natural products. And how many NP would a chemist say “that looks good!”?</p>&mdash; Keith Robison (@OmicsOmicsBlog) <a href="https://twitter.com/OmicsOmicsBlog/status/1143341958439604224?ref_src=twsrc%5Etfw">June 25, 2019</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

So yeah, that's why drug discovery is hard in a nutshell. While there are many ways of defining what is or is not druglike and there has been many methods from heuristics, to logic-based reasoning, to machine learning, to detect drug-like compounds, it ultimately boils down to making a decision under the knowledge that the cost/reward structure is highly skewed in ways we can't always predict. How can we go about and reason in this context?  

## A nibble of decision theory

Whenever you look for ways to frame an informed decision given data and a cost/reward structure, you inevitably run into normative decision theories -- frameworks that deal with the most optimal decisions an agent can take to maximize its outcomes. There are two key abstract ingredients in these decision theories: the preferences that the agent has and the prospects of getting them. In many such theories, these preferences typically translate to some measure of *utility* and the
prospects turn into *probabilities*. The most natural and intuitive way of combining utilities and probabilities is through the expected utility function, which is a sum over events of the utility of an each event weighed by its probability. While explored and informally defined by many, including Bernoulli, since at least the 1700's it was really until 1947 that Von Neumann and Morgenstern formalized this notion and discovered how tight the an agent's preferences were entwined with the probability of the
outcomes.

## Objective probability

In the Von Neumann and Morgenstern world, the agent has to pick between different worlds, called lotteries, where each event is assigned a probability of occurring. Further, an agent's preferences obeys four main rules: 

1. You can either prefer one lottery over another, vice-versa, or be indifferent; that is, the relationship is complete.
2. Your preferences between lottery are transitive
3. Given three lotteries with a set preference order, $L \leq M \leq N$, you can always combine $L$ and $N$ via weighted averages of their respective probabilities in such a way that they are at least equivalent or greater than $M$. That is, that there is no lottery that is so bad or so good that it always is better or worse than combinations of other lotteries.
4. Given two lotteries where you prefer one over the other, this preference is maintained regardless of adding the same perturbation in probabilities to both.

Turns out that in this scenario, given an order of preference of the lotteries, we can find a *utility function* that maps events to real utility values that when applied to each of the lotteries in expectation (weighed by their probabilities) such expected utilities follow the same order. This is also true in reverse, a utility function then defines a preferential order.

## Subjective probability

In the Von Neumann-Morgensetern world, the risk of the lotteries encoded by the probabilities of the events, is absolute. The state of the world according to the agent and its beliefs is never modeled and ironically the agent really loses its agency since the lotteries encode its fate. A more agent-centric view of the world was put forth by Savage in 1954, in which the agent can act on the world via some state-action function according to its beliefs of what will happen next, and in which its actions will be tied with some utility function; a 
framework that is very similar to Markov decision processes and other reinforcement learning beasts. Interestingly, the same kind of link between expected utility and an agent's beliefs hold here as well. That is, given a utility function, we can find a probability distribution that encodes the agent's beliefs such that its actions maximize the utility function's expected utility -- and vice versa.  This analogous conclusion requires reasoning and comparing preferences between the state-action functions as well, and a mirror of Von Neumann-Morgenstern's axioms on lotteries (that perferences over state-action functions are complete, transitive, are that they can be perturbed both in ways their preference ordering changes or is maintained) is put in place to fulfill this.

## Sensitivity/specificity/prevalence trade-off decisions

Let's return to our drug discovery case. Here, you have a heuristic/intuition/experimental assay/computational method that yields a go/no-go decision on whether to pursue the drug further. It all really boils down to three main factors: how good our decision making instrument is, how tractable is the real problem, and what is our utility function with regards to possible outcomes. Such a scenario is general and comes up a lot in other areas such as diagnostics. Following decision theory, we will want to combine all three into an expected utility we can examine.
Let's attach some quantities to each of these three factors. To quantify how good our decision instrument is (e.g. the state of our test), we can use quantities such as sensitivity and specificity which could be quantified retroactively via the rate of true positives found by the test versus all true positives and the rate of true negatives found by the test versus all true negatives, respectively. To quantify how hard the problem is (e.g. the state of the world) we can use the prevalence of the true positives, that
is, on average how many actual good drugs we can expect there to be in the general universe of molecules. Finally, the utility function will ultimately depend on how much the decision will cost (pursuing the drug further) and how much we can expect the benefits of finding a good drug to be. We can also add a third term, a penalization of sorts that happens when we chase a false positive, which we will call the follow-up cost, and which can be a real burden not only in drug discovery but in
diagnostics and health policy as well (what if the detected cancer wasn't really there and we follow up with more invasive procedures?).These costs and rewards will ultimately be shaped by some functional form into a final utility function. This functional form can be as straightforward as linear but also as drastic as a mirrored double exponential modeling a high-risk/high-reward scenario, or dampened by a logarithm to signify diminishing returns, or tempered with an isoelastic function to model
balanced risk aversion.

## A simple model

Putting everything we enumerated above together, we can build a simple model of the decision making process. Let $se$ be the sensitivity of the test, $sp$ the specificity, $p$ the prevalence of the true positives in the general population, $c$ the cost of a go decision, $r$ the reward, $fup$ the follow-up cost as defined above, and $f$ be our utility functional form (e.g. linear, logarithmic, etc.). In all of this we will assume that the cost of the test itself is constant and
therefore not included in the final utility values. We will also assume that the probability of finding a true positive is exactly the sensitivity $se$ and the probability of finding a true negative is exactly the specificity $sp$. In the general case, these probabilities correlate with the sensitivity and specificity but are not expected to be the same thing (especially in borderline out-of-distribution scenarios). There are four possible scenarios when performing a test:

* With probability $p$ we get an actual true positive. We then perform the test and with probability $se$ we find the true positive. Our reward is $f(r - c)$ which is weighed by the probability of this scenario $p \times se$
* With probability $p$ we get an actual true positive. We then perform the test and with probability $1 - se$ we miss the true positive. Our utilty is $f( c )$ which is weighed by the probability of this scenario $p \times (1- se)$
* With probability $1 - p$ we get an actual true negative. We then perform the test and with probability $sp$ we correctly decide not to pursue. Our utilty is $f(0)$ in this case since we do not take further action. This is weighed by the probability of this scenario $(1 - p) \times sp$
* With probability $1 - p$ we get an actual true negative. We then perform the test and with probability $1 - sp$ we flag this, falsely, as a positive. We pay the follow-up cost and our utilty is $f(-c-fup)$ which is weighed by the probability of this scenario $(1 - p) \times (1 - sp)$

Summing all these quantities, weighed by the probabilities of each scenario playing out, will give us the expected utility.

## Visualization

We could write out the expected utility and ponder on boundary cases, singularities, etc. But it's much more fun if we can visualize it interactively. I've written a small visualization tool for this simple model, which you can use to explore how utility changes with each choice in prevalence, cost, reward, utility functional form, etc. Here, I'm plotting utilty as a heatmap over sensitivity and specificity, so you can see how e.g. importance of sensitivity increases when prevalence decreases and specificity increases when prevalence increases. I clipped the min/max colormap values of the heatmap to always be on the [-1,1] range so you can see the changes more easily.

<div id="root"></div>
<script src="/apps/sense_and_sensitivity.js"></script>

## Addendum: Causal Decision Theory

One could go further than reasoning over an agent's beliefs and instead reason over the causal models the agent has of the world, a causal decision theory. Here, the literature gets murkier as there are many ways to go about and insert causal models into the framework -- treating the agent's acts as a do-operator or treating the agent's model of the world as a causal model to name a couple. In any case, there seems to be no firm answer of what produces the best results and there
are unfortunately too few tests of these theories, at least to my knowledge.
