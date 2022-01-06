---
layout: post
categories: posts
title:  Erdos goes neural
subtitle: 'The probabilistic method for neural combinatorial optimization'
tags:
  - 'Erdős goes neural, Combinatorial Optimization, Neural Networks'
date-string: 'January 5, 2022'
featured-image: /images/erdos_neural/erdos6.jpg
preview-image: images/erdos_neural/erdos5.jpg
published: true
---

### Disclaimer:
This is a brief tutorial on the probabilistic penalty method presented in our [paper][1]. I will be taking shortcuts in how I present it in order to make it straightforward and understandable. It does not cover the paper in its full generality. I will not explain the probabilistic method and I won't expand on the various important details that constitute the full paper.
Instead, I aim to provide an accessible primer into our work by minimizing the technical info and focusing on practicality/simplicity. I encourage you to read the paper and the supplementary material section for the complete treatment. Otherwise, if you're still unsure, just email me. Without further ado, let's get started.

## Erdos goes neural: a simplified tutorial

### Preliminaries:
Before we start, we should clear up what the goal of this work is. The main objective is to solve combinatorial problems with neural networks, 
**without supervision** (i.e., without access to labeled solutions). We focused on graph problems but the principles of our method can be applied to other settings. Furthermore, many (combinatorial) problems have an equivalent graph formulation, so you could reformulate your problem in a graph setting if you want to be as close to what we did as possible.  

OK, great. Now that this is out of the way, let's talk about how we are going to get things done.
First, let's be a bit more specific with the setup.

We want a set of objects $$S$$ that solves a given problem, e.g., finds the set of nodes that forms the largest possible clique on a graph (maximum clique problem). 
Since we focused on graph problems in the paper, we will work with sets of nodes on the input graph, but you can also do sets of edges/tuples/etc. It will depend on your particular problem and what is more convenient for you. 
**From now on I will be talking about sets of nodes. Feel free to substitute it with edges or whatever else you need to work with.**

Here is what we are going to need:
#### 1) A combinatorial problem with a nonnegative cost function and a set of constraints.
#### 2) A neural network that produces a probability for each node. This is the probability that the node belongs to the set $$S$$. 
#### 3) A differentiable loss function. This loss takes as input the probabilities that were produced by the network. The loss will be derived from your problem's objective (I will explain below).  



#### Technical Detail:
In the paper, we use Bernoulli variables over the nodes of the input graph. You could work with other distributions as well, but if you are unsure I recommend starting with Bernoulli  random variables over the entities you care about (over nodes/edges/tuples/etc.). We will need to derive an expectation of a function over those random variables and it happens that Bernoulli variables tend to lead to easier derivations.
For the rest of the post $$x_i$$ is a Bernoulli random variable placed on node $$i$$,

  $$x_i= \begin{cases} 1, \quad \text{with probability } p_i  \\
0, \quad \text{with probability } 1-p_i .
\end{cases} $$

## How to solve it

Now, suppose you have a combinatorial problem. You need to follow the steps below.
### Step 1:
Write down the cost function of your problem, for example in our paper we consider the maximum clique problem.
Here is the standard way of expressing it:
#### maximize weight($$S$$), subject to: $$S$$ is a clique.
For a simple undirected graph, weight(S) (henceforth $$w(S)$$) just counts the number of edges in the subgraph induced by S. 
To be in line with the paper, we need to switch to a  minimization problem, hence:
#### minimize $$\gamma-w(S)$$, subject to: $$S$$ is a clique.
$$\gamma$$ here is a sufficiently large constant;  Let $$\gamma \geq \text{max}_{S}(w(S))$$, so that the expression remains always nonnegative.
<hr>

### Step 2:
Set up a graph neural network (the choice of model depends on the task and engineering considerations) for your data. Using any of the mainstream
GNNs (GIN, GAT, etc.) should be fine to start with.
The GNN takes as input some node features. Its output is an N x 1 vector of probabilities, one for each node.
For the specifics on features, layers, normalizations, etc. you can just look at the code in the repo. Or you may improve upon the pipeline by working with your own features, layers, etc.
<hr>

### Step 3:
Derive a differentiable loss.
The differentiable loss function has to look like this:
#### Loss = Expected Cost + $$\beta$$ * P(S does not satisfy constraints).
$$\beta$$ here is a coefficient that controls the importance of the constraint in the loss. 
The expectation of the cost can be straightforward for many set functions. 
For the Prob(S does not satisfy constraints)  term (henceforth $$ P(S \notin \Omega) $$), we can use Markov's inequality to bound it. 

### Example: 
First, derive the expected cost. In our max-clique example, $$\text{cost} = \gamma-w(S)$$.
$$\gamma$$ is just a constant so $$E[cost] = \gamma - E[w(S)]$$.
We have $$w(S) = \sum_{(i,j) \in E}x_i x_j$$ where $$E$$ is the set of edges of the graph.


In other words, an edge $$(i,j)$$ is in  $$S$$ if both endpoints $$i$$ and $$j$$ are in  $$S$$.
We are using Bernoulli variables, therefore:
$$ E[w(S)] = \sum_{(i,j)\in E}p_i p_j. $$

OK, now onto the constraint part of the loss.
For the maximum clique problem, the constraint dictates that **the subgraph induced by S is a clique**, i.e, all pairs of nodes in S are connected by an edge.
An equivalent way to phrase this is that, **there are no edges in the complement of the subgraph induced by S**.
Let $$ \bar{S} $$ be the complement of S. Then,
#### $$ P(S \notin \Omega)  = P(w(\bar{S})>=1). $$
From Markov's inequality, this can be bounded as follows
#### $$  P(w(\bar{S})>=1) \leq E[w(\bar{S})]. $$
So our loss ends up being
#### $$ \text{Loss} = \gamma - E[w(S)] + \beta E[w(\bar{S})]. $$

<hr>

### Step 4:
Train the network using the derived loss. This is straightforward because you just have to plug in the probabilities in the expression and do backprop.
<hr>

### Step 5:
Retrieve the set $$S$$ from the probabilities of the network using the method of conditional expectation.
It works as follows.

Sort the nodes according to their probabilities.
Starting from the high probability nodes, for each node v_i do:


  1. Evaluate the loss for $$p_i=1$$ and for $$p_i=0$$.
  2. Set  $$p_i$$ to either 1 or 0 depending on what achieved the better loss.
  3. Move on to the next node and repeat from step 1.
  
When this is done, you should have a binary (indicator) vector that represents your set $$S$$, which is the solution to your problem.
Congrats! That's it, you're done.

[1]: https://arxiv.org/abs/2006.10643