---
layout: page
title: "Machine Learning Glossary"
description: "Machine Learning Glossary / Cheat Sheet with a focus on intuition."
header-img: "img/machine-learning-glossary-bg.jpg"
order: 1
mathjax: true
comments:	true
---

Welcome,

This is my **first post ever** :bowtie:, I would love to get your [feedback](#disqus_thread){:.mdLink}.

<!-- 
What dropdown: start open
 -->
<p>
<details open>
  <summary>What</summary>
  
  <p>I will try to <b>summarise important terms and concepts of machine learning</b>. The <b>focus is on intuition</b> but there will also be practical and theoretical notes.  </p>
  
   <b>Target Audience, </b>this is for you if you:
  <ul style="display:inline-block;"> 
  	<li> Have a decent understanding of a concept but <b>want more intuition.</b></li>
    <li> Are <b>switching machine learning subdomains. </b> </li>
    <li> <b>Knew a term, but want to refresh</b> your knowledge as it's hard to remember everything (that's me :sweat_smile: ).</li>
    <li> Need to <b>learn the important concepts in an efficient way</b>. Students cramming for an exam: that's for you :four_leaf_clover: !</li>
    
 </ul> 
    
    
  
  
  
</details> 
</p>

* any text here
{:toc}

<!-- 
Why dropdown: start closed
 -->
<details>
  <summary>Why</summary>
  
  <p>Having a bad memory but beeing (at least considering myself to be :sweat_smile: ) a philomath who loves machine learning, I developped the habit of taking notes, then summarizing and finally making a cheat sheet for every new ML domain I encounter. There are multiple reasons I want to switch to a webpage: </p>
 
  <ul>
    <li><p>Having lots of paper is <b>not practical</b> and prone to loss.  </p></li>
    <li><p>The idea that someone I don't know (I'm talking about you :raising_hand: ) might read this post at some point <b>makes me want to write higher quality notes</b> .</p></li>
    <li><p>I have always been impressed by how much people are willing to spend time on forums and open source projects. I'm forever gratefull to them and <b>I now want to give back to the community</b> (The contribution isn't comparable, but I have to start somewhere :innocent: ).</p></li>
    <li><p>Taking notes on a computer is a necessary step for my migragtion from Biomedical Engineering to CS. I guess you could call that <b>peer pressure</b> :sweat_smile: .</p></li>
    <li><p> As wise man once said: <blockquote> You do not really understand something unless you can explain it to your grandmother. <cite> - Albert Einstein </cite>
    	   </blockquote> 
    	   My grandma's are awesome :heart: but not really into ML (yet). You have thus been designated "volunteer" to temporarily replace them.
    </p></li>
  </ul> 
</details>

<!-- 
How dropdown: start closed
 -->
<p>
<details>
  <summary>How</summary>
  
  <p>To make it easier to search the relevant information in the Glossary here is the color coding I will be using:  </p>
 
  <ul style="list-style: none;">
    <li class="col-xs-6"> :bulb: <span class="intuition"> Intuition </span> </li>
    <li class="col-xs-6"> :wrench: <span class="practice"> Practical </span> </li>
    <li class="col-xs-6"> :red_circle: <span class="disadvantage"> Disadvantage </span> </li>
    <li class="col-xs-6"> :white_check_mark: <span class="advantage"> Advantage </span> </li>
    <li class="col-xs-6"> :school_satchel: <span class="example"> Example </span> </li>
    <li class="col-xs-6"> :mag: <span class="note"> Side notes </span> </li>
    <li style="position:relative;left:15px;"> :information_source: <span class="ressource"> Ressources </span><br /> </li>
  </ul> 
</details>
</p>


**Disclaimer**: 
* This is my **first post ever** :bowtie:, I would love to get your [feedback](#disqus_thread){:.mdLink}.
* I'm bad at spelling. I **apologize in advance for any mistakes** (feel free to correct me).    
* **Check out the [ressources](/ressources/){:.mdLink}** from where I got most of this information. 
* ML subdomains overlap **A LOT**. I'll try not to make the separations to artificial. I separate domains both by *learning style* and by *algorithm similarity*. I find it more understandable that way, I hope that won't be an issue for clarity. Any suggestions would be appreciated :relaxed: .
* This is not meant to be a post read in order, but rather used as a "cheat-sheet". Simply use the [table of content](#markdown-toc){:.mdLink} or `Ctrl+f`.

Enough talking: let's get going :rocket: ! 


## General Machine Learning Terms
### Fundamental Concenpts
#### No Free Lunch theorem
#### Fundamental Tradeoff (bias-variance)
#### Parametric vs Non Parametric
#### Curse of Dimensionality
#### Frequentist vs Bayesian
#### Online Learning
#### Overfitting
#### Evalutaion / Model Selection


### Quick Definitions

**Kernels**

**Stochastic algorithms**

**Maximum Likelihood Estimation**

**MAP**

**Monte Carlo Estimation**

**KL divergence**

**Entropy**

**Inference**

**Surrogate Loss Function**

**Convex functions**

**Norms**

### Regularization
### Ensemble Learning
#### Boosting
#### Bootstrapped Aggregation (Bagging)
#### Stacked Generalization (blending)
#### Averaging Generalization 

## Supervised Learning

### Regression
#### Linear Models
##### Logistic Regression
##### Ordinary Least Squares
#### Decision Trees

### Classification
#### Discriinative Classifiers
##### Decision Trees
##### Logistic Regression (LR)
##### Softmax 
##### Support Verctor Machines (SVM)
##### Artificial Neural Networks
#### Generative Classifiers
##### Naive Bayes

## Unsupervised Learning

### Clustering

Nota Bene: Careful when ensemble learning to label switching
#### K-Neirest Neighbors
#### K-Means
#### Density Based Clustering
#### Hierarchical Clustering
#### Spectral Clustering

### Latent Factor Model
#### Dimensionality reduction
##### Principal Component Analysis (PCA)
##### Independant Component Analysis (ICA)
##### Sammon Mapping
##### Multidimensional Scaling (MDS)
##### Projection Pursuit
##### Linear Discriminant Analysis (LDA)
##### ISOMAP
##### T-SNE
##### Autoencoders

### Density Estimation
#### Collaborative Filtering

Recommender sytsems (also content based or hybrid)


### Outlier Detection

Nota Bene: distinguish global and local outliers
##### Model Based
##### Graphical Approaches
##### Cluster - Based
##### Distance - Based


### Association rules


## Reinforcement Learning

## Graphical Models
### Directed Graphical Models

### Undireted Graphical Models

## Partially supervised learning

non t called like that
### Active Learning
### Semi-supervised learning

---------

## Deep Learning
### How to train
### COnvolutional Neural Network
### Recurrent Neural Network
### Autoencoders

## Natural Language Processing

## Time Series

## Other
### Causal Learning
### State Space Models
## Computational neuroscience
### Spiking Neural Networks

## Optimisation
### Evolutionnary Methods

Unfortunately here ends today's journey together. But don't get too excited, I'm only getting started. I still have a tuns of notes eagerly waiting to get upgraded, and I don't intend to stop learning yet :sweat_smile:. 

PS: Any reaction/suggestions would be very appreciated: just drop a comment below or click on the :heart: .

See you soon :kissing_heart: 

---------
---------

---------

---------
---------
---------
---------


















---------
---------

---------

---------
---------
---------
---------
---------
---------

---------

---------
---------
---------
---------







dd

<dev class="intuitionBoxed">
:mortar_board: <span class="intuition">Intuition</span>:  <span class="intuitionText"> and this is the end of the explnation.</span>
</dev>

hh


<dev class="disadvantageBoxed">
:red_circle: <span class="disadvantage"> Disadvantage</span>:  
<span class="disadvantageText"> and this is the end of the explnation.</span>
</dev>

hh

<dev class="advantageBoxed">
:white_check_mark: <span class="advantage"> Advantage</span>:  
<span class="advantageText"> and this is the end of the explnation.</span>
</dev>

hh

<dev class="exampleBoxed">
<span class="example">Example</span>:  <span class="exampleText"> and this is the end of the explnation.</span>
</dev>

hh

<dev class="practiceBoxed">
:wrench: <span class="practice">Practical</span>:  <span class="practiceText"> and this is the end of the explnation.</span>
</dev>

| Header1 | Header2 | Header3 |
|:--------|:-------:|--------:|
|       ss | cell2   | cell3   |
| cell4   | cell5   | cell6   |
| cell1   | cell2   | cell3   |
| cell4   | cell5   | cell6   |


jj

$$e^{27}$$

*Nota Bene: these terms are not always the most important ones but important ones I have encountered since my "migration" to machine learning / computer science in September 2016.*

Thanks to [Mark Schmidt](https://www.cs.ubc.ca/~schmidtm/){:.mdLink}, my Machine Learning professor, who introduced me to this amazing field.
