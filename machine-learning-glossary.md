---
layout: page
title: "Machine Learning Glossary"
description: "Machine Learning Glossary / Cheat Sheet with a focus on intuition"
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

<details open>
  <summary>What</summary>
  
  <div markdown="1">
  I will try to **summarise important terms and concepts of machine learning**. The **focus is on intuition** but there will also be practical and theoretical notes. 
  
  **Target Audience,** this is for you if you:

* Have a decent understanding of a concept but **want more intuition.**
* Are **switching machine learning subdomains.**
* **Knew a term, but want to refresh** your knowledge as it's hard to remember everything (that's me :sweat_smile: ).
* Need to **learn the important concepts in an efficient way**. Students cramming for an exam: that's for you :four_leaf_clover: !
</div>
</details> 
<p></p>

* any text here
{:toc}

<!-- 
Why dropdown: start closed
 -->
 <details>
  <summary>Why</summary>
  
  <div markdown="1">
  Having a bad memory but beeing (at least considering myself to be :sweat_smile: ) a philomath who loves machine learning, I developped the habit of taking notes, then summarizing and finally making a cheat sheet for every new ML domain I encounter. There are multiple reasons I want to switch to a webpage: 
  
<ul>
    <li>Paper is <b>not practical</b> and prone to loss. </li>
    <li>Thinking that someone I don't know (I'm talking about you :raising_hand: ) might read this post <b>makes me write higher quality notes</b> .</li>
    <li>I'm forever gratefull to people that spend time on forums and open source projects. <b>I now want to give back to the community</b> (The contribution isn't comparable, but I have to start somewhere :innocent: ).</li>
    <li>Taking notes on a computer is a necessary step for my migragtion to CS :sweat_smile: .</li>
    <li>As a wise man once said: <blockquote> You do not really understand something unless you can explain it to your grandmother. <cite> - Albert Einstein </cite>
    	   </blockquote> 
    	   My grandma's are awesome :heart: but not really into ML (yet). You have thus been designated "volunteer" to temporarily replace them.</li>
  </ul> 
</div>
</details> 

<!-- 
How dropdown: start closed
 -->
<p>
<details>
  <summary>How</summary>
  
  <div markdown="1">
  To make it easier to search the relevant information in the Glossary here is the color coding I will be using:  
 
  <ul style="list-style: none;">
    <li class="col-xs-6"> :bulb: <span class="intuition"> Intuition </span> </li>
    <li class="col-xs-6"> :wrench: <span class="practice"> Practical </span> </li>
    <li class="col-xs-6"> :x: <span class="disadvantage"> Disadvantage </span> </li>
    <li class="col-xs-6"> :white_check_mark: <span class="advantage"> Advantage </span> </li>
    <li class="col-xs-6"> :school_satchel: <span class="example"> Example </span> </li>
    <li class="col-xs-6"> :mag: <span class="note"> Side notes </span> </li>
    <li class="col-xs-6"> :wavy_dash: <span class="compare"> Compare to </span> </li>
    <li style="position:relative;left:15px;"> :information_source: <span class="ressource"> Ressources </span><br /> </li>
  </ul> 
</div>
</details>
</p>


**Disclaimer**: 
* This is my **first post ever** :bowtie:, I would love to get your [feedback](#disqus_thread){:.mdLink}.
* I'm bad at spelling: **Apologies in advance** (feel free to correct me).    
* **Check out the [ressources](/ressources/){:.mdLink}** from where I got most of this information. 
* ML subdomains overlap **A LOT**. I'll try not to make the separations too artificial. Any suggestions would be appreciated :relaxed: . Note that I separate domains both by *learning style* and by *algorithm similarity*. 
* This is not meant to be a post read in order, but rather used as a "cheat-sheet". Use the [table of content](#markdown-toc){:.mdLink} or `Ctrl+f`.

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
#### Evalutaion Metrics
##### Classification Metrics
###### Single Metric
* **TP** / **TN** / **FN** / **FP:** The best way to understand these is to look at a $$2*2$$ [confusion matrix](#visual-metrics){:.mdLink}.

![confusion matrix](/img/blog/confusion-matrix.png)

* **Accuracy:** fraction of observation correctly classified. 
	*  $ Acc = \frac{Real Positives}{Total} = \frac{TP+FN}{TP+FN+TN+FP}$
	* :bulb: <span class="intuitionText"> In general, how much can we trust the predictions ? </span>
	* :wrench: <span class="practiceText"> Use if no class inbalance and cost of error is the same for both types  </span>
* **Precision** fraction of the observation predicted as positive that were actually positive. 
	* $ Prec = \frac{TP}{Predicted Positives} = \frac{TP}{TP+FP}$
	* :bulb: <span class="intuitionText"> How much can we trust positive predictions ? </span>
	* :wrench: <span class="practiceText"> Use if FP are the worst  </span>
* **Recall** fraction of the positive observation that have been correctly predictied. 
	* $ Rec = \frac{TP}{Actual Positives} = \frac{TP}{TP+FN}$
	* :bulb: <span class="intuitionText"> How many actual positives will we find? </span>
	* :wrench: <span class="practiceText"> Use if FN are the worst  </span>
* **F1-Score** harmonic mean (good for averaging rates) of recall and precision.
	* $F1 = \frac{2}{\frac{1}{Recall} + \frac{1}{Precision}}$
	* :bulb: <span class="intuitionText"> How well much can we trust our algorithms for the positive class</span>
	* :wrench: <span class="practiceText"> Use if the positive class is more improtant and/or if there are less positive observations  </span>
* **Specificity** like recall but for negatives. $ Spec = \frac{TN}{Actual Negatives} = \frac{TN}{TN+FP}$
* **Log-Loss** measures performance when model outputs a probability between 0 and 1
	* Also called **Cross entropy loss** or **logistic loss**
	* $logLoss = - \frac{1}{N} \sum^N_{i=1} \sum^K_{c=1} y_{ic} \ln(p_{ic})$
	* Use the natural logarithm for consistency
	* 
	* :bulb: <span class="intuitionText"> Penalizes more if confident but wrong (see graph below)</span>
	* :wrench: <span class="practiceText"> Use when your model outputs a probability of beeing in a class </span>
	* The graph below shows the log loss depending on the confidence of the algorithm that an observation should be classed in the correct category. For multiple onservation we compute the log loss of each and then average them.

![log loss](/img/blog/log-loss.png)
* **AUC**

###### Visual Metrics
* **Confusion Matrix**
* **Roc Curve**
###### Other
* **Cohen's Kappa**
* **Kolmogorov-Smirnov**
* **Gini**
* ...

:information_source: <span class="ressource"> Addiditional Ressources </span>: [Additional scores based on confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix){:.mdLink}, [Additional scores based on confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix){:.mdLink}
##### Regression Metrics
#### Evalutaion / Model Selection
#### Hyperparameter Optimisation
#### Entropy related terms
##### Entropy

<details open>
  <summary>Long Story Short</summary>
  <div markdown="1">
* $$H(p) = \sum_i^N p_i \ \log(\frac{1}{p_i}) = - \sum_i^N p_i\  log(p_i)$$
* :bulb: <span class="intuition"> Intuition </span>:
	* The entropy of a random variable is intuitively the <span class="intuitionText"> expected amount of suprise you would have by observing it  </span>. We often say <span class="intuitionText"> it is a measure of "information" </span> in the sense that if something is not suprising to you, then you didn't learn much by seeing. So it didn't convey much new information. 
	* <span class="intuitionText"> Entropy is the expected number of bits (for $log_2$) used to encode an obrservation from a (discrete) random variable under the optimal coding scheme </span>. 

* Don't confuse the "information" in information theory with the everyday word which refers to "meaningfull information". <span class="exampleText"> A book with random letters will have more information because each new letter would be a suprise to you. But it will definitely not have more meaning than a book with english words </span>.

* :mag: <span class="note"> Side notes </span> :
	* $H(X) \geq 0$
	* Entropy is maximized when all events occure with uniform probability. If $X$ can take $n$ values then $max(H) = log(n)$
</div>
</details>

<p></p>


<details>
  <summary>Long Story Long</summary>
  <div markdown="1">
  
The simple concept of entropy is central in both thermodynamics and information theory, and I find that quite amazing. It orginally comes from statistical thermodynamics and is so central there, that it is carved on Ludwig Blotzmann's grave (one of the father of this field). You will often hear:

* **Thermodynamics**: *Entropy is a measure of disorder*
* **Information Theory**: *Entropy is a measure of information*

These 2 way of thinking may seem different but in reality they are exactly the same. They essentially answer: <span class="intuitionText"> how hard is it to describe this thing? </span>

I will focus here on the information theory point of view, because its interpretation is more intuitive for machine learning. Also I don't want to spend to much time thinking about thermodynamics, as [people that do often commit suicide](http://www.eoht.info/page/Founders+of+thermodynamics+and+suicide){:.mdLink} :flushed:.

$$H(p) = \sum_i p_i \ \log(\frac{1}{p_i}) = - \sum_i p_i\  log(p_i)$$

 In information theory there are 2 intuitive way of thinking of entropy. These are best explained through an <span class="example"> example </span> : 

<div class="exampleBoxed">
<div markdown="1">
:school_satchel: Imagine that my friend [Claude](https://en.wikipedia.org/wiki/Claude_Shannon){:.mdLink} offers me to go see a NBA game (Cavaliers vs Spurs) with him tonight. Unfortunately I can't come but ask him to record who scored each field goals. Claude is very geeky and uses a binary phone which can only write 0 and 1. As he doesn't have much memory left, he wants to use the smallest possible number of bits.

1. From previous games, Claude knows that Lebron James will very likely score more than the old (but awesome :basketball: ) Manu Ginobili. Will he use the same number of bits to indicate that Lebron scored, than he will for Ginobili ? Of course not, he will allocate less bits for Lebron as he will be writing it down more often. He's essentally exploiting his knowledge about the distribution of field goals to reduce the expected number of bits to write down. It turns out that if he knew the probability $p_i$ of each player $i$ to score he should encode their name with $nBit(p_i)=log_2(1/p_i)$ bits. This has been intuitively constructed by Claude (Shannon) himself as it is the only measure (up to a constant) that satisfies axioms of information measure. The intuition behind this is the following:
	*  <span class="intuitionText"> Multiplying probabilities of 2 players scoring should result in adding their bits. </span> Indeed imagine Lebron and Ginobili have respectively 0.25 and 0.0625 probability of scoring the next field goal. Then, the probability that Lebron scores the 2 next field goals would be the same than Ginobili scoring a single one ($lebron*lebron = 0.25 * 0.25 = 0.0625 = Ginobili$). We should thus allocate 2 times less bits for Lebron, so that on average we always add the same number of bits per observation. $nBit(p_{Lebron}) = \frac{1}{2} * nBit(p_{Ginobili}) = \frac{1}{2} * nBit(p^2_{Lebron})$. From this we quickly realize that we need to use logarithms and that the simplest H will be of the form: $H(p_i) = \alpha * \log(p_i) + \beta $
	* <span class="intuitionText"> Players that have higher probability of scoring should be encoded by a lower number of bits </span>. I.e H should decrease when $p_i$ increases: $H(p_i) = - \alpha * \log(p_i) + \beta, \alpha > 0  $
	* <span class="intuitionText"> If Lebron had $100%$ probability of scoring, why would I have bothered asking Claude to write anything down ? I would have known everything *a priori* </span>. I.e H should be $0$ for $p_i = 1$ : $H(p_i) = - \alpha * \log(p_i), \alpha > 0  $

2. Now Claude sends me the message containing information about who scored. Seeying that Lebron scored will suprise me less than Ginobili. I.e Claude's message gives me more information when telling me that Ginobili scored. If I wanted to quantify my suprise for each field goal, I should make a measure that satisfies the following conditions:
	* <span class="intuitionText">The lower the probability of a player to score, the more suprised I will be </span>. The measure of suprise should thus be a decreasing function of probability: $suprise(p_i) = -f(p_i) * \alpha, \alpha > 0$.
	* Supposing that players scoring are independent of one another, it's reasonable to ask that my suprise if Lebron and Ginobili scored in a row should be the same than the sum of my suprise if Lebron scored and my suprsie if Ginobili scored. <span class="intuitionText"> Multiplying independant probabilities should sum the suprise </span>: $suprise(p_i * p_j) = suprise(p_i) + suprise(p_j)$.
	* Finally, <span class="intuitionText"> the measure should be continuous given probabilities </span>. $suprise(Lebron) = -\log(p_{Lebron}) * \alpha, \alpha > 0$

Taking $\alpha = 1 $ for simplicity, we get $suprise(p_i) = -log(p_i) =  nBit(p_i)$. We thus derived a formula for computing the suprise associated with event $i$ and the optimal number of bits that should be used to encode that event. <span class="intuitionText">In order to get the average suprise / number of bits associated with a random variable $X$ we simply have to take the expectation over all possible events</span> (i.e average weighted by probability of event). This gives us the entropy formula $H(p) = \sum_i p_i \ \log(\frac{1}{p_i}) = - \sum_i p_i\  log(p_i)$

</div>
</div>

From the example above we see that entropy corresponds to : 
<div class="intuitionText">
<div markdown="1">
* **to the expected number of bits to optimally encode a message**
* **the average amount of information gained by observing a random variable** 
</div>
</div>

:mag: <span class="note"> Side notes </span> :
* From our derivation we see that the function is defined up to a constant term $\alpha$. This is the reason why the formula works equally well for any logarithmic base, indeed changing the base is the same as multiplying by a constant. In the context of information theory we use $log_2$.
* Entropy is the reason (second law of thermodynamics) why putting an ice cube in your *moscow mule* (yes that is my go-to drink) doesn't normally make your ice cube colder and your cocktail warmer. I say "normally" because it is possible but very unprobable : ponder about this next time your sipping your own go-to drink :smirk: ! 

:information_source: <span class="ressource"> Addiditional Ressources </span>: Exellent explantion of the link between [entropy in thermodynamics and information theory](http://www.askamathematician.com/2010/01/q-whats-the-relationship-between-entropy-in-the-information-theory-sense-and-the-thermodynamics-sense/){:.mdLink}, friendly [ introduction to entropy related concepts](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/){:.mdLink}

</div>
</details>
<p></p>

##### Differential Entropy
Differetial entropy (= continuous entropy), is the generalization of entropy for continuous random variables.

Given a continuous random variable $X$ with a probability density function $f(x)$:

$$h(X) = - \int_{-\infty}^{\infty} f(x) \log {f(x)} \ dx$$

If you had to make a guess, which distribution maximizes entropy for a given variance ? You guessed it : it's the **Gaussian distribution**.

:mag: <span class="note"> Side notes </span> : Differential entropy can be negative.

##### Cross Entropy
We [saw that](#entropy){:.mdLink} entropy is the expected number of bits used to encode an observation of $X$ under the optimal coding scheme. In contrast <span class="intuitionText"> cross entropy is the expected number of bits to encode an observation of $X$ under the wrong coding scheme</span>. Let's call $q$ the wrong probability distribution that is used to make a coding scheme. Then we will use $-log(q_i)$ bits to encode the $i^{th}$ possible values of $X$. Although we are using $q$ as a wrong probability distribution, the observations will still be distributed based on $p$. We thus have to take the expected value over $p$ :

$$H(p,q) = - \sum_i^N p \log(q)$$

From this interpretation it naturally follows that:
* $H(p,q) > H(p), \forall q \neq p$
* $H(p,p) = H(p)$

:mag: <span class="note"> Side notes </span> : Log loss is often called cross entropy loss, indeed it is the cross-entropy between 

##### Kullback-Leibler Divergence

##### Machine Learning and Entropy
Well all this is interesting, but why are we talking about information theory concepts in machine learning :sweat_smile: ? Well it turns our that many ML algorithms can be interpreted with entropy related concepts.

The 2 major ways we see entropy in machine learning are through:
* **Maximising information gain** (i.e entropy) at each step of our algorithm. 

* When building decision trees you greedily select split which maximizes information gain (i.e the difference of entropy before and after the split). Intuitively you want to minimize the number of splits you should do afterwards to correctly classify the observation.

* **Minimizing the KL divergence between the actual unkown probability distribution of observations $p$ and the predicted one $q$ **. 

* **Minimizing the KL divergence between the computationally intractable $p$ and a simpler approximation $q$ **. Indeed machine learning is not only about theory but also about how to make something work in practice.



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

**Hyperparameter vs Parameter**

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
*The classification problem consists of assigning a set of classes/categories to an observation. I.e* $$x \mapsto y,\ y \in \{0,1,...,K\}$$

Classification problems can be further separated into:

* **Binary:** There are 2 possible classes. $$K=2,\ y \in \{0,1\}$$
* **Multi-Class:** There are more than 2 possible classes. $$K>2$$
* **Multi-Label:** If labels are not mutually exclusive. Often replaced by $$K$$ binary classification specifiying whether an observation should be assigned to each class.

Common evaluation metrics:
* **Accuraccy**
* **Confusion Matrix**
* **Accuraccy**
* **F1-Score**
* ****

:wavy_dash: <span class="compare"> Compare to </span> : 
[Regression](#regression){:.mdLink}

#### Discriminative Classifiers

:wavy_dash: <span class="compare"> Compare to </span> : 
[Regression](#generative-classifiers){:.mdLink}
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
### Model Free Reinforcement Learning
### Model Based Reinforcement Learning

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

## Bayesian Optimisation

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
