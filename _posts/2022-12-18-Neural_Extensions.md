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
This post is meant to serve as an intuitive and practical explainer of our recent work on [neural set function extensions][1], recently presented at NeurIPS 2022. As in previous posts, I will be skipping potentially important details in the interest of brevity and understandability. For the full details, I recommend reading the paper or even just emailing me. Now, let's get started.

## Motivation
Suppose that you have a neural network which produces a score vector $$ \mathbf{x} \in [0,1]^{n}$$ given some input problem instance (a graph, an image, etc.), which is then used to solve some downstream task. That vector could be marginal probabilities of graph nodes for some kind of node selection task (e.g., probabilities that nodes are solving a combinatorial problem), or even the class probabilities for a classification task. From now on, let's say $$\mathbf{x}$$ are marginal probabilities for node selection, i.e., $$x_i$$ is the probability that node $$i$$ belongs to a set of nodes $$S$$ that is meant to solve some graph problem.

Now assume that you have a function $$f: 2^n \rightarrow \mathbb{R} $$ whose input is any subset of $$ n $$ items. We can view the domain of $$f$$ as the space of n-dimensional binary vectors, i.e., $$ f: \{0,1\}^n \rightarrow \mathbb{R}$$. I'm obviously abusing notation here to emphasize that the *inputs* of this function are *discrete*, however the *outputs* are allowed to be *continuous*. 
The input domain consists of all the possible indicator vectors for all the subsets of $$n $$ items we could pick. For example, let's assume $$f$$ is the cardinality function, i.e., the function that counts how many items the provided set has. Then for $$n=3$$, a set $$S$$ could be represented by a 3-dimensional binary vector, e.g., $$\mathbf{1}_S= \begin{bmatrix}1 & 1 & 0 \end{bmatrix}$$. In the case of that example vector, clearly the cardinality is $$ f(S) = f(\mathbf{1}_S) = 2 $$.

Naturally, that kind of function is not compatible with arbitrary $$ \mathbf{x}$$ due to the continuity of $$\mathbf{x}$$. It doesn't make much sense to ask what is the cardinality of (for example) $$  \begin{bmatrix} 0.3 & 0.5 & 0.2 \end{bmatrix}$$ to begin with. 
 If we want to find a set that minimizes (or maximizes) the function by improving the scores in $$\mathbf{x}$$ in a differentiable neural network pipeline, we have to square away this incompatibility.
Here, there are certain options one would consider:

- Discretize the continuous output. We could sample with the Gumbel trick or even just threshold at 0.5 the values of $$\mathbf{x}$$ to obtain a binary vector. The we could use a [straight-through estimator][2] in the backward pass to go through the discretization procedure. While this provides a discrete vector that $$ f $$ is compatible with, it may only work if the function $$ f $$ itself *is differentiable*. 

- Assume the function $$ f $$ is a black box, so we have no guarantees that we can differentiate through it. In that case, some kind of [stochastic gradient estimation][3] might be our next option. For example we could use REINFORCE, i.e., sample sets $$ S $$ from $$ \mathbf{x} $$ then use the log-derivative trick to backpropagate through $$\mathbf{x}$$ while treating $$f$$ as the reward function.

- Use a known continuous relaxation of the function. Again that assumes that we have access to the function and a bespoke relaxation exists. That does not 
require any discretization. When available, it's a pretty good option.


These are all valid approaches that could make sense in certain scenarios. Here, we will propose a different strategy that aims to mitigate the drawbacks of some of the listed approaches. First, we can go back to the main obstacle we started from. What does it mean to ask about the cardinality of the input (value of the discrete function) when the input is not a set (binary vector) but a vector of continuous values?  There is a way to make this question well-posed. We can treat the continuous values as parameters of a probability distribution over sets. Then we can ask what is the *expected value of the function* over the distribution of sets encoded by $$ \mathbf{x}$$. 
### So what is a scalar set function extension? It is simply a new function $$ \mathfrak{F}: [0,1]^n \rightarrow \mathbb{R} $$  which *extends* the original in a differentiable way. We will define it as the expected value of the function $$f$$ over a distribution of sets encoded by $$\mathbf{x}$$.


The main objective of the paper is to provide a foundation for this approach. It is a general framework for constructing continuous extensions of discrete functions defined on sets, that can be *deterministically* computed, even when we're only given black-box access to $$f$$. The paper presents multiple extensions that originate from a common mathematical formulation. Some are already known like the Lovasz extension, and others are new, like the bounded cardinality Lovasz extension. Furthermore, we provide a way to not only extend to continuous domains, but also onto *high-dimensional* ones. We will explore this in an upcoming post (part 2 of the series).
<hr>

## Scalar Extensions
We call  $$ \mathfrak{F} $$ a scalar extension of a function $$ f $$, if  $$ \mathfrak{F}(\mathbf{1}_S) = f(S) $$ for all $$ S $$ in the domain of $$f$$. For example,  $$ \mathfrak{F}(\mathbf{0}) = f(\emptyset)$$, i.e., the extension evaluated at the origin (all-zeros vector) should correspond to $$ f $$ being evaluated at the empty set. 


### How can you compute  $$ \mathfrak{F} $$?
There is a simple trick to defining $$ \mathfrak{F}$$. As explained earlier, we can view $$ \mathbf{x}$$ as encoding a distribution over sets.
 To do this, we can express any continuous point in the n-hypercube $$ \mathbf{x} \in [0,1]^n $$ in the following way:
#### $$ \displaystyle  \mathbf{x} = \sum_{i} a_i \mathbf{1}_{S_i}, \quad \displaystyle \sum_i a_i = 1. $$
In other words, for every continuous point in the hypercube, we can find certain corners of the hypercube and express said point as a convex combination of those corners. This defines a distribution:
####  $$ \mathbf{x} =  \displaystyle\mathbb{E}_{S \sim \mathbf{p_x}}[S] $$.
Here $$\mathbf{p_x} $$ is the probability distribution induced by $$ \mathbf{x} $$. This distribution is fully described by the coefficients $$a_i$$ in the convex combination.
Once we have $$ S_i $$ and $$a_i$$ defining $$\mathfrak{F}$$  is simple:
#### $$ \mathfrak{F}(\mathbf{x})  \triangleq \displaystyle\sum_{i} a_i f({S_i}) = \mathbb{E}_{S \sim \mathbf{p_x}}[f(S)]$$.
The distribution is supported only on sets $$ S_i $$ and each set has a corresponding probability of $$ a_i $$.
Intuitively, the value of the extension at the continuous point is just a weighted combination of the values of the original function $$f$$ at the corners that form the convex combination of $$\mathbf{x}$$. Furthermore, the weights of the weighted combination are precisely the same weights used to express $$\mathbf{x}$$ as a convex combination of hypercube corners.

A large chunk of the paper is dedicated to properly formalizing this trick. Obviously, I will keep things simple here so I won't get into all that. I encourage the curious reader to check out the paper.
Now, there are a couple of questions that naturally have to be addressed. One has to do with how we find those sets $$S_i$$ and their coefficients $$a_i$$. Is it computationally tractable? Can we do it fast? 
The other has to do with whether this thing is differentiable at all? 
Tha answer to all those questions is, modulo certain disclaimers, yes.

### How can we backpropagate through $$ \mathfrak{F}(\mathbf{x}) $$? 
The trick for this is simple. As long as $$ a_i = g(\mathbf{x}) $$, and $$g$$ is some continuous function of $$ \mathbf{x}$$, then gradients can just go through
the coefficients $$a_i$$ in the sum that defines $$ \mathfrak{F}(\mathbf{x}) $$.


### How do we find sets $$S_i$$ and their coefficients $$a_i$$? 
In the paper we provide multiple examples of scalar extensions. Each one comes with its own way of computing sets $$S_i$$ and their probabilities $$a_i$$. Crucially, the coefficients $$a_i$$ depend continuously on $$\mathbf{x}$$ which allows us to do backpropagation.  The 'cheapest' extensions that require only black-box access to the function $$ f $$ are the Lovasz extension, the bounded-cardinality Lovasz extension, the singleton, and the permutation/involutory extensions.
These all require $$n+1$$ sets (including the empty set) and coefficients.  Before I go into specific extensions and how to compute them, I want to emphasize that the key point to remember is that you could find your own extensions by figuring out ways to express continuous points as convex combinations of discrete points.
<hr>

## In practice: The Lovasz extension
The Lovasz extension is well known in the fields of discrete analysis/optimization and submodularity. It has various particularly nice properties but perhaps the most important one is that iff $$f$$ is a [submodular function][4], then the Lovasz extension of $$f$$ is convex. Examples of submodular functions include graph cuts, coverage functions, rank functions, and so on. 
Computing the Lovasz extension is straightforward.
First, we index the entries of $$\mathbf{x}$$ in sorted, decreasing order: $$ x_i \geq x_{i+1} $$ for $$i=1,2,\dots , n-1$$. That means that $$ x_1 $$ corresponds to the largest entry in $$\mathbf{x}$$, $$x_2$$ to the second largest, and so on.
 The coefficients and the sets of the Lovasz extension are then defined as follows:
### $$a_i = x_i - x_{i+1}$$
### $$S_i = \{1:i \} $$.
Here, I'm using a bit of coding notation with $$1:i$$ to indicate "all elements up to $$i$$".
Given the sets and their coefficients, the Lovasz extension is computed by
### $$ \mathfrak{F}(\mathbf{x}) = \displaystyle \sum_{i=1}^{n}  (x_i - x_{i+1})f( \{1: i \} ) $$.
Clearly, $$ a_i $$ are differentiable with respect to $$\mathbf{x}$$ as they're just pairwise differences of the coordinates of $$ \mathbf{x}$$.
To make things concrete, let's do the calculation for our example vector from before: $$\mathbf{x} = \begin{bmatrix} 0.3 & 0.5 & 0.2\end{bmatrix}$$.
Based on the ranking of the elements, we will have the following sets
### $$ 1_{S_1} = \begin{bmatrix} 0 & 1 & 0 \end{bmatrix}, 1_{S_2} = \begin{bmatrix} 1 & 1 & 0 \end{bmatrix}, 1_{S_3} = \begin{bmatrix} 1 & 1 & 1 \end{bmatrix} $$
and the following coefficients
### $$ a_1 = 0.5-0.3 = 0.2, $$
### $$ a_2 = 0.3-0.2 = 0.1,  $$
### $$ a_3 = 0.2 $$ .
It is easy to verify that $$ \mathbf{x} = \sum_{i} a_i \mathbf{1}_{S_i} $$. One might notice that $$\sum_i a_i  <1$$ in this case, even though I initially said that we need a sum to one. Thankfully, that's not a problem because we can allocate the remaining probability mass to the empty set. By convention $$ f(\emptyset) = 0$$ so that term just cancels out. Therefore, we don't strictly need the sum to 1, $$\sum_i a_i \leq 1 $$ can also be fine.


## In practice: Extensions for combinatorial optimization



## To be continued: Neural Extensions
I intentionally avoided discussing the meaning of the word *scalar* in the subtitle of the post. This has to do with the domain of the function. If $$ \mathbf{x} \in [0,1]^n $$, we are assigning a single scalar score to each item among $$n $$ items. In the sequel, we will see how we can stretch the definition of extensions to allow for $$\mathbf{X} \in [0,1]^{n \times d}$$, i.e., $$d$$-dimensional embeddings for each item, which is usually the kind of representation that the layers of a neural network operate with ($$ d $$ would just be the width of a NN). But that's a story for another time. Until then, I hope this has been helpful.


[1]: https://arxiv.org/abs/2208.04055
[2]: https://www.hassanaskary.com/python/pytorch/deep%20learning/2020/09/19/intuitive-explanation-of-straight-through-estimators.html#what-is-a-straight-through-estimator
[3]: https://arxiv.org/abs/1711.00123
[4]: https://en.wikipedia.org/wiki/Submodular_set_function