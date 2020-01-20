+++
date = "2020-01-20T21:17:31-07:00"
draft = true
title = "What is your loss function actually measuring?"

+++

A question that from time to time pops in my head when reading about/dealing with models is "what is a loss function, actually?". While in some cases it's clear cut, many times I feel that it is an overloaded concept -- it can mean many things and in the end only makes sense in the context of a task at hand. A loss function can be measuring risk, likelihood of an event, reification of our beliefs, or a proxy for what we think our system should be doing. This is not necessarily a bad thing, as it's many times useful to analyze a loss function with many lenses (it's always fun to tease out the "Bayesian interpretation" of some loss function to make it more concise what our beliefs are...for some definition of fun anyway). But failing to consider all the implications of choosing a loss function can also lead to misconceptions on how the model is going to perform or what we actually want out of it. Many times we focus on distributional aspects of a model: e.g. how accurate is it compared to others in the same task? how well does it generalize? is its sensitivity and specificity acceptable? did my MCMC converge on a sensible space? Answering these questions are necessary, but many times the most crucial questions come at the end: is using this model worth it? and if so, by how much? The utility function derived from applying the model is in the end what's going to dictate its impact in the world. You can't assess the risk of using a model or the 
