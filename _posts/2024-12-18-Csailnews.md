---
layout: post
categories: posts
title: Are Neural Networks Optimal Approximation Algorithms?
#subtitle: Can GNNs learn optimal approximation algorithms?
tags:
  - Learning with approximation guarantees
  - Neural Networks
  - Combinatorial optimization
date-string: 'December 18, 2024'
#featured-image: /images/optapp/quarticspectrahedron.png
preview-image: /images/optapp/bwspectrahedra.jpg
published: true
---
This NeurIPS, we presented our work on the question of whether neural networks can be efficiently used to design optimal approximation algorithms. The answer is a conditional yes. Assuming the Unique Games Conjecture, there is a general
semidefinite program (SDP) that achieves optimal approximation guarantees. In our paper, we show that this SDP can be efficiently solved using a rather simple graph neural network architecture (GNN). More specifically, gradient descent on the quadratically penalized Lagrangian of the SDP leads to message passing iterations, which we execute through a neural network that we call OptGNN. 

This leads to an architecture that achieves strong empirical results across several NP-Hard combinatorial problems. The paper has also been featured on [MIT CSAIL news](https://www.csail.mit.edu/news/deep-learning-np-hard-problems) so feel free to check that out for some additional commentary.

