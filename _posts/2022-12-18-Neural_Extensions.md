---
layout: post
categories: posts
title:  Learning with Neural Extensions
subtitle: 'Part 1: Scalar set function extensions'
tags:
  - Learning with discrete functions
  - Neural Networks
  - Combinatorial optimization
date-string: 'December 18, 2022'
featured-image: /images/extensions/lovasz.jpg
preview-image: /images/extensions/geometry2.jpg
published: false
---

## Motivation
The goal of this post is to serve as an intuitive and practical explainer of our recent work on [neural set function extensions][1], recently presented at NeurIPS 2022. As in previous posts, I will be skipping potentially important details in the interest of brevity and understandability. For the full details I recommend reading the paper or even just emailing me. Now, let's get started.

Suppose that you have a neural network which produces a score vector $$ \mathbf{x} \in [0,1]^{n}$$ given some input problem instance (a graph, an image, etc.), which is then used to solve some downstream task. That vector could be marginal probabilities of graph nodes for some kind of node selection task (e.g., probabilities that nodes are solving a combinatorial problem), or even the class probabilities for a classification task. For the purpose of this example, let's say $$x$$ are marginal probabilities for node selection, i.e., $$x_i$$ is the probability that node $$i$$ belongs to a set of nodes $$S$$ that is meant to solve some combinatorial problem.

Now assume that you have a function $$f: 2^n \rightarrow \mathbb{R} $$ whose input is any subset of $$ n $$ items. We can view the domain of $$f$$ as the space of n-dimensional binary vectors, i.e., $$ f: \{0,1\}^n \rightarrow \mathbb{R}$$. Those are all the possible indicator vectors for all possible subsets we could pick.   Again, for the sake of simplicity, let's assume that $$f$$ is the cardinality function, i.e., the function that counts how many items the provided set has. For example, for $$n=3$$ a set $$S$$ could be represented by a 3-dimensional binary vector, e.g., $$\mathbf{1}_S= \begin{bmatrix}1 & 1 & 0 \end{bmatrix}$$. Then, clearly $$ f(S) = f(\mathbf{1}_S) = 2 $$.

Naturally, that kind of function is not compatible with  $$ \mathbf{x}$$ due to the continuity of $$\mathbf{x}$$. It doesn't make much sense to ask what is the cardinality of $$  \begin{bmatrix} 0.3 & 0.5 & 0 \end{bmatrix}$$ to begin with. If we want to find scores that minimize (or maximize) the function in a differentiable neural network pipeline, we're in trouble.
Here, there are certain options one would consider:

- Discretize the continuous output. We could sample with the Gumbel trick or even just threshold at 0.5 the values of $$\mathbf{x}$$ to obtain a binary vector. The we could use a [straight-through estimator][2] in the backward pass to go through the discretization procedure. While this provides a discrete vector that $$ f $$ is compatible with, it may only work if the function $$ f $$ itself *is differentiable*. 

- Assume the function $$ f $$ is a black box, so we have no guarantees that we can differentiate through it. In that case, some kind of [stochastic gradient estimation][3] might be our next option. For example we could use REINFORCE, i.e., sample sets $$ S $$ from $$ \mathbf{x} $$ then use the log-derivative trick to backpropagate through $$\mathbf{x}$$ while treating $$f$$ as the reward function.

- Use a known continuous relaxation of the function. Again that assumes that we have access to the function and a bespoke relaxation exists. That does not 
require any discretization. When available, it's a pretty good option.

- #### Neural Extension: Define a new function $$ \mathfrak{F}: [0,1]^n \rightarrow \mathbb{R} $$  which *extends* the original in a differentiable way.

The main objective of the paper is to provide a foundation for that last option. It is a general framework for constructing continuous extensions of discrete functions defined on sets, that can be *deterministically* computed, even when we're only given black-box access to them. The paper presents multiple extensions unified under the same foundation, some already known like the Lovasz extension and new ones we propose like the bounded-cardinality Lovasz extension. Furthermore, we provide a way to not only extend to continuous domains, but also onto *high-dimensional* ones. More on that on part 2.
<hr>

## Scalar Extensions
We call  $$ \mathfrak{F} $$ an extension of a function $$ f $$, if  $$ \mathfrak{F}(\mathbf{1}_S) = f(S) $$ for all $$ S $$ in the domain of $$f$$. For example,  $$ \mathfrak{F}(\mathbf{0}) = f(\emptyset)$$, i.e., the extension evaluated at the origin (all-zeros vector) should correspond to $$ f $$ being evaluated at the empty set. 


### How can you compute  $$ \mathfrak{F} $$?
There is a simple trick to computing $$ \mathfrak{F}$$. We can express any continuous point in the n-hypercube $$ \mathbf{x} \in [0,1]^n $$ in the following way:
#### $$ \displaystyle  \mathbf{x} = \sum_{i} a_i \mathbf{1}_{S_i}, \quad \displaystyle \sum_i a_i = 1. $$
In other words, for every continuous point in the hypercube, we can find certain corners of the hypercube and express said point as a convex combination of those corners. This defines a distribution $$ \mathbf{x} =  \mathbb{E}_{S \sim \mathbf{p_x}}[S] $$.
Once we have $$ S_i $$ and $$a_i$$ defining $$\mathfrak{F}$$  is simple:
#### $$ \mathfrak{F}(\mathbf{x})  \triangleq \sum_{i} a_i f({S_i}) = \mathbb{E}_{S \sim \mathbf{p_x}}[f(S)]$$.
Here $$\mathbf{p_x} $$ is the probability distribution induced by $$ \mathbf{x} $$. This distribution is fully described by the coefficients $$a_i$$ in the definition. The distribution is supported only on sets $$ S_i $$ and each set has a corresponding probability of $$ a_i $$.
Intuitively, the value of the extension at the continuous point is just a weighted combination of the values of the original function $$f$$ at the corners that form the convex combination of $$\mathbf{x}$$. Furthermore, the weights of the weighted combination are precisely the same weights used to express $$\mathbf{x}$$ as a convex combination of hypercube corners.

A large chunk of the paper is dedicated to properly formalizing this trick and establishing why it makes sense. Obviously, I will keep things simple here so I won't get into all that. I encourage the curious reader to check out the paper.
Now, there are a couple of questions that naturally have to be addressed. One has to do with how we find those sets $$S_i$$ and their coefficients $$a_i$$. Is it computationally tractable? Can we do it fast? 
The other has to do with whether this thing is differentiable at all and how? 
Tha answer to all those questions is, modulo certain disclaimers, yes.

### How do we find sets $$S$$ and how many do we need?
As you may have observed, I did not provide the set of indices for the sum in the definition of the extensions.






## To be continued: Neural Extensions
I intentionally avoided discussing the *scalar* in the subtitle of the post. This has to do with the domain of the function. If $$ \mathbf{x} \in [0,1]^n $$, we are assigning a single scalar score to each item in space of $$n $$ items. In the sequel, we will see how we can stretch the definition of extensions to allow for $$\mathbf{X} \in [0,1]^{n \times d}$$, i.e., $$d$$-dimensional embeddings for each item, which is usually the kind of representation that the layers of a neural network operate with ($$ d $$ would just be the width of a NN). 


[1]: https://arxiv.org/abs/2208.04055
[2]: https://www.hassanaskary.com/python/pytorch/deep%20learning/2020/09/19/intuitive-explanation-of-straight-through-estimators.html#what-is-a-straight-through-estimator
[3]: https://arxiv.org/abs/1711.00123