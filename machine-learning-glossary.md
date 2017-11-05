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
  I will try to **summarize important terms and concepts of machine learning**. The **focus is on intuition** but there will also be practical and theoretical notes. 
  
  **Target Audience,** this is for you if you:

* Have a decent understanding of a concept but **want more intuition.**
* Are **switching machine learning sub-domains.**
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


<div>
<details>
  <summary>Why</summary>
  
  <div markdown="1">
  Having a bad memory but being (at least considering myself to be :sweat_smile: ) a philomath who loves machine learning, I developed the habit of taking notes, then summarizing and finally making a cheat sheet for every new ML domain I encounter. There are multiple reasons I want to switch to a web-page: 

  <ul>
    <li>Paper is <b>not practical</b> and prone to loss. </li>
    <li>Thinking that someone I don't know (I'm talking about you :raising_hand: ) might read this post <b>makes me write higher quality notes</b> .</li>
    <li>I'm forever grateful to people that spend time on forums and open source projects. <b>I now want to give back to the community</b> (The contribution isn't comparable, but I have to start somewhere :innocent: ).</li>
    <li>Taking notes on a computer is a necessary step for my migration to CS :sweat_smile: .</li>
    <li>As a wise man once said: <blockquote> You do not really understand something unless you can explain it to your grandmother. <cite> - Albert Einstein </cite>
    	   </blockquote> 
    	   My grandma's are awesome :heart: but not really into ML (yet). You have thus been designated "volunteer" to temporarily replace them.</li>
  </ul> 
  </div>
</details>
</div> 

<!-- 
How drop-down: start closed
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
    <li class="col-xs-6"> :mag: <span class="note"> Side Notes </span> </li>
    <li class="col-xs-6"> :wavy_dash: <span class="compare"> Compare to </span> </li>
    <li style="position:relative;left:15px;"> :information_source: <span class="ressource"> Resources </span><br /> </li>
  </ul> 
</div>
</details>
</p>

**first post ever**

**Disclaimer**: 
* This is my **first post ever** :bowtie:, I would love to get your [feedback](#disqus_thread){:.mdLink}.
* I'm bad at spelling: **Apologies in advance** (feel free to correct me).    
* **Check out the [resources](/ressources/){:.mdLink}** from where I got most of this information. 
* ML sub-domains overlap **A LOT**. I'll try not to make the separations too artificial. Any suggestions would be appreciated :relaxed: . Note that I separate domains both by *learning style* and by *algorithm similarity*. 
* This is not meant to be a post read in order, but rather used as a "cheat-sheet". Use the [table of content](#markdown-toc){:.mdLink} or `Ctrl+f`.

Enough talking: let's get going :rocket: ! 


## General Machine Learning Terms
### Fundamental Concepts
#### No Free Lunch theorem
#### Fundamental Trade-off (bias-variance)
#### Parametric vs Non Parametric
#### Curse of Dimensionality
#### Frequentist vs Bayesian
#### Online Learning
#### Overfitting
#### Evaluation Metrics
##### Classification Metrics
###### Single Metrics

:mag: <span class='notes'> Side Notes </span> : I will mostly focus on binary classification but most scores can be generalized to the multi-class setting. Often this is achieved by only considering "correct class" and "incorrect class" in order to make it a binary classification, then you average (weighted by the proportion of observation in the class) the score for each classes.

* **TP** / **TN** / **FN** / **FP:** The best way to understand these is to look at a $$2*2$$ [confusion matrix](#visual-metrics){:.mdLink}.

![confusion matrix](/img/blog/confusion-matrix.png)

* **Accuracy:** fraction of observation correctly classified. 
	*  $ Acc = \frac{Real Positives}{Total} = \frac{TP+FN}{TP+FN+TN+FP}$
	* :bulb: <span class="intuitionText"> In general, how much can we trust the predictions ? </span>
	* :wrench: <span class="practiceText"> Use if no class imbalance and cost of error is the same for both types  </span>
* **Precision** fraction of the observation predicted as positive that were actually positive. 
	* $ Prec = \frac{TP}{Predicted Positives} = \frac{TP}{TP+FP}$
	* :bulb: <span class="intuitionText"> How much can we trust positive predictions ? </span>
	* :wrench: <span class="practiceText"> Use if FP are the worst  </span>
* **Recall** fraction of the positive observation that have been correctly predicted. 
	* $ Rec = \frac{TP}{Actual Positives} = \frac{TP}{TP+FN}$
	* :bulb: <span class="intuitionText"> How many actual positives will we find? </span>
	* :wrench: <span class="practiceText"> Use if FN are the worst  </span>
* **F1-Score** harmonic mean (good for averaging rates) of recall and precision.
	* $F1 = 2 \frac{Precision * Recall}{Precision + Recall}$
    * If recall is $\beta$ time more important than precision use $F_{\beta} = (1+\beta^2) \frac{Precision * Recall}{\beta^2 Precision + Recall}$
	* :bulb: <span class="intuitionText"> How well much can we trust our algorithms for the positive class</span>
	* :wrench: <span class="practiceText"> Use if the positive class is more important (want a *detector* more than a *classifier*)</span>

* **Specificity** like recall but for negatives. $ Spec = \frac{TN}{Actual Negatives} = \frac{TN}{TN+FP}$

* **Cohen's Kappa** Improvement of your classifier over always guessing the most probable class 
    * $\kappa = \frac{accuracy - percentageMaxClass}{1 - percentageMaxClass}$
    * More generally can compare 2 raters (ex: 2 humans): $\kappa = \frac{p_o- p_e}{1 - p_e}$ where $p_o$ is the observed agreement and $p_e$ is the expected agreement due to chance.
    * $ \kappa \leq 1$ (if $<0$ then useless).
    * :bulb: <span class='intuitionText'>Accuracy improvement weighted by class imbalance </span> .
    * wrench: <span class='practiceText'> Use when high class imbalance and all classes are ~the same importance</span>
    
* **Log-Loss** measures performance when model outputs a probability $\hat{y_ic}$ that observation $i$ is in class $c$
	* Also called **Cross entropy loss** or **logistic loss**
	* $logLoss = - \frac{1}{N} \sum^N_{i=1} \sum^K_{c=1} y_{ic} \ln(\hat{y}_{ic})$
	* Use the natural logarithm for consistency
	* Incorporates the idea of probabilistic confidence
  * Log Loss is the metric that is minimized through [Logistic Regression](#logistic-regression){:.mdLink} and more generally [Softmax](#softmax){:.mdLink}
  * :bulb: <span class="intuitionText"> Penalizes more if confident but wrong (see graph below)</span>
  * :bulb: <span class="intuitionText"> Log-loss is the</span>  [cross entropy](#cross-entropy){:.mdLink} <span class="intuitionText"> between the distribution of the true labels and the predictions</span> 
  * :wrench: <span class="practiceText"> Use when you are interested in outputting confidence of results </span>
  * The graph below shows the log loss depending on the confidence of the algorithm that an observation should be classed in the correct category. For multiple observation we compute the log loss of each and then average them.

![log loss](/img/blog/log-loss.png)

* **AUC** **A**rea **U**nder the **C**urve. Summarizes curves in a simple single metric.
  * It normally refers to the [ROC](#visual-metrics){:.mdLink} curve. Can also be used for other curves like the precision-recall one.
  * :bulb: <span class='intuitionText'> Probability that a randomly selected positive has a higher score than a randomly selected negative observation </span> .
  * :mag: <span class='noteText'> AUC evaluates results at all possible cut-off points. It gives better insights about how well the classifier is able to separate between classes </span>. This makes it very different from the other metrics above that depend on the cut-off threshold.
  * :wrench: <span class='practiceText'> Use when building a classifier for users that will have different needs (they could tweak the cut-off point)</span> . From my experience AUC is often used in statistics (~go-to metric in bio-statistics) but less so in machine learning.
  * Random predictions: $AUC = 0.5$. Perfect predictions: $AUC=1$.
 
###### Visual Metrics
* **Confusion Matrix** a $K*K$ matrix which shows the number of observation of class $c$ that have been labeled $c'$ $\forall c,c' \in 1,...,K$
    * :mag: <span class='noteText'> Be careful: People are not consistent with the axis :you can find real-predicted and predicted-real  </span> .
    * This is best understood through an example:

![Multi Confusion Matrix](/img/blog/multi-confusion-matrix.png)

* **ROC Curve** **R**eceiver **O**perating **C**haracteristic
  * Plot showing the TP rate vs the FP rate, over a varying threshold.
  * This plot from [wikipedia](https://commons.wikimedia.org/wiki/File:ROC_curves.svg){:.mdLink} shows it well:
  
![ROC curve](/img/blog/ROC.png)

:information_source: <span class="ressource"> Additional Resources </span>: [Additional scores based on confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix){:.mdLink}

##### Regression Metrics
#### Evaluation and Model Selection
#### Hyperparameter Optimization
#### Entropy Related Terms
##### Entropy

<details open>
  <summary>Long Story Short</summary>
  <div markdown="1">
* $$H(p) = \sum_i^N p_i \ \log(\frac{1}{p_i}) = - \sum_i^N p_i\  log(p_i)$$
* :bulb: <span class="intuition"> Intuition </span>:
	* The entropy of a random variable is intuitively the <span class="intuitionText"> expected amount of surprise you would have by observing it  </span>. We often say <span class="intuitionText"> it is a measure of "information" </span> in the sense that if something is not surprising to you, then you didn't learn much by seeing. So it didn't convey much new information. 
	* <span class="intuitionText"> Entropy is the expected number of bits (for $log_2$) used to encode an observation from a (discrete) random variable under the optimal coding scheme </span>. 

* Don't confuse the "information" in information theory with the everyday word which refers to "meaningful information". <span class="exampleText"> A book with random letters will have more information because each new letter would be a surprise to you. But it will definitely not have more meaning than a book with English words </span>.

* :mag: <span class="note"> Side notes </span> :
	* $H(X) \geq 0$
	* Entropy is maximized when all events occur with uniform probability. If $X$ can take $n$ values then : $max(H) = H(p_{uniform})= \sum_i^n \frac{1}{n} \log(\frac{1}{ 1/n} ) = \log(n)$

</div>
</details>

<p></p>


<details>
  <summary>Long Story Long</summary>
  <div markdown="1">
  
The simple concept of entropy is central in both thermodynamics and information theory, and I find that quite amazing. It originally comes from statistical thermodynamics and is so central there, that it is carved on Ludwig Boltzmann's grave (one of the father of this field). You will often hear:

* **Thermodynamics**: *Entropy is a measure of disorder*
* **Information Theory**: *Entropy is a measure of information*

These 2 way of thinking may seem different but in reality they are exactly the same. They essentially answer: <span class="intuitionText"> how hard is it to describe this thing? </span>

I will focus here on the information theory point of view, because its interpretation is more intuitive for machine learning. Also I don't want to spend to much time thinking about thermodynamics, as [people that do often commit suicide](http://www.eoht.info/page/Founders+of+thermodynamics+and+suicide){:.mdLink} :flushed:.

$$H(p) = \sum_i p_i \ \log(\frac{1}{p_i}) = - \sum_i p_i\  log(p_i)$$

 In information theory there are 2 intuitive way of thinking of entropy. These are best explained through an <span class="example"> example </span> : 

<div class="exampleBoxed">
<div markdown="1">
:school_satchel: Imagine that my friend [Claude](https://en.wikipedia.org/wiki/Claude_Shannon){:.mdLink} offers me to go see a NBA game (Cavaliers vs Spurs) with him tonight. Unfortunately I can't come but ask him to record who scored each field goals. Claude is very geeky and uses a binary phone which can only write 0 and 1. As he doesn't have much memory left, he wants to use the smallest possible number of bits.

1. From previous games, Claude knows that Lebron James will very likely score more than the old (but awesome :basketball: ) Manu Ginobili. Will he use the same number of bits to indicate that Lebron scored, than he will for Ginobili ? Of course not, he will allocate less bits for Lebron as he will be writing it down more often. He's essentially exploiting his knowledge about the distribution of field goals to reduce the expected number of bits to write down. It turns out that if he knew the probability $p_i$ of each player $i$ to score he should encode their name with $nBit(p_i)=log_2(1/p_i)$ bits. This has been intuitively constructed by Claude (Shannon) himself as it is the only measure (up to a constant) that satisfies axioms of information measure. The intuition behind this is the following:
	*  <span class="intuitionText"> Multiplying probabilities of 2 players scoring should result in adding their bits. </span> Indeed imagine Lebron and Ginobili have respectively 0.25 and 0.0625 probability of scoring the next field goal. Then, the probability that Lebron scores the 2 next field goals would be the same than Ginobili scoring a single one ($lebron*lebron = 0.25 * 0.25 = 0.0625 = Ginobili$). We should thus allocate 2 times less bits for Lebron, so that on average we always add the same number of bits per observation. $nBit(p_{Lebron}) = \frac{1}{2} * nBit(p_{Ginobili}) = \frac{1}{2} * nBit(p^2_{Lebron})$. From this we quickly realize that we need to use logarithms and that the simplest H will be of the form: $H(p_i) = \alpha * \log(p_i) + \beta $
	* <span class="intuitionText"> Players that have higher probability of scoring should be encoded by a lower number of bits </span>. I.e H should decrease when $p_i$ increases: $H(p_i) = - \alpha * \log(p_i) + \beta, \alpha > 0  $
	* <span class="intuitionText"> If Lebron had $100%$ probability of scoring, why would I have bothered asking Claude to write anything down ? I would have known everything *a priori* </span>. I.e H should be $0$ for $p_i = 1$ : $H(p_i) = - \alpha * \log(p_i), \alpha > 0  $

2. Now Claude sends me the message containing information about who scored. Seeing that Lebron scored will surprise me less than Ginobili. I.e Claude's message gives me more information when telling me that Ginobili scored. If I wanted to quantify my surprise for each field goal, I should make a measure that satisfies the following conditions:
	* <span class="intuitionText">The lower the probability of a player to score, the more surprised I will be </span>. The measure of surprise should thus be a decreasing function of probability: $surprise(p_i) = -f(p_i) * \alpha, \alpha > 0$.
	* Supposing that players scoring are independent of one another, it's reasonable to ask that my surprise if Lebron and Ginobili scored in a row should be the same than the sum of my suprise if Lebron scored and my surprise if Ginobili scored. <span class="intuitionText"> Multiplying independent probabilities should sum the surprise </span>: $surprise(p_i * p_j) = surprise(p_i) + surprise(p_j)$.
	* Finally, <span class="intuitionText"> the measure should be continuous given probabilities </span>. $surprise(Lebron) = -\log(p_{Lebron}) * \alpha, \alpha > 0$

Taking $\alpha = 1 $ for simplicity, we get $surprise(p_i) = -log(p_i) =  nBit(p_i)$. We thus derived a formula for computing the surprise associated with event $i$ and the optimal number of bits that should be used to encode that event. <span class="intuitionText">In order to get the average surprise / number of bits associated with a random variable $X$ we simply have to take the expectation over all possible events</span> (i.e average weighted by probability of event). This gives us the entropy formula $H(p) = \sum_i p_i \ \log(\frac{1}{p_i}) = - \sum_i p_i\  log(p_i)$

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
* Entropy is the reason (second law of thermodynamics) why putting an ice cube in your *Moscow Mule* (yes that is my go-to drink) doesn't normally make your ice cube colder and your cocktail warmer. I say "normally" because it is possible but very improbable : ponder about this next time your sipping your own go-to drink :smirk: ! 

:information_source: <span class="ressource"> Additional Resources </span>: Excellent explanation of the link between [entropy in thermodynamics and information theory](http://www.askamathematician.com/2010/01/q-whats-the-relationship-between-entropy-in-the-information-theory-sense-and-the-thermodynamics-sense/){:.mdLink}, friendly [ introduction to entropy related concepts](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/){:.mdLink}

</div>
</details>
<p></p>

##### Differential Entropy
Differential entropy (= continuous entropy), is the generalization of entropy for continuous random variables.

Given a continuous random variable $X$ with a probability density function $f(x)$:

$$h(X) = - \int_{-\infty}^{\infty} f(x) \log {f(x)} \ dx$$

If you had to make a guess, which distribution maximizes entropy for a given variance ? You guessed it : it's the **Gaussian distribution**.

:mag: <span class="note"> Side notes </span> : Differential entropy can be negative.

##### Cross Entropy
We [saw that](#entropy){:.mdLink} entropy is the expected number of bits used to encode an observation of $X$ under the optimal coding scheme. In contrast <span class="intuitionText"> cross entropy is the expected number of bits to encode an observation of $X$ under the wrong coding scheme</span>. Let's call $q$ the wrong probability distribution that is used to make a coding scheme. Then we will use $-log(q_i)$ bits to encode the $i^{th}$ possible values of $X$. Although we are using $q$ as a wrong probability distribution, the observations will still be distributed based on $p$. We thus have to take the expected value over $p$ :

$$H(p,q) = - \sum_i p_i \log(q_i)$$

From this interpretation it naturally follows that:
* $H(p,q) > H(p), \forall q \neq p$
* $H(p,p) = H(p)$

:mag: <span class="note"> Side notes </span> : Log loss is often called cross entropy loss, indeed it is the cross-entropy between 

##### Kullback-Leibler Divergence
The Kullback-Leibler Divergence (= relative entropy = information gain) from q to p is simply the difference between the cross entropy and the entropy:

$$
\begin{align*} 
D_{KL}(p\|q) &= H(p,q) - H(p) \\
&= [- \sum_i p_i \log(q_i)] - [- \sum_i p_i \log(p_i)] \\
&= \sum_i p_i \log(\frac{p_i}{q_i})
\end{align*} 
$$

* :bulb: <span class="intuition"> Intuition </span>
	* KL divergence corresponds to the number of additional bits you will have to use when using an encoding scheme based on the wrong probability distribution $q$ compared to the real $p$ .
	* KL divergence says in average how much more surprised you will be by rolling a loaded dice but thinking it's fair, compared to the surprise of knowing that it's loaded.
	* KL divergence is often called the **information gain** achieved by using $p$ instead of $q$
	* KL divergence can be thought as the "distance" between 2 probability distribution. Mathematically it's not a distance as it's none symmetrical. It is thus more correct to say that it is a measure of how a probability distribution $q$ diverges from an other one $p$.
	
KL divergence is often used with probability distribution of continuous random variables. In this case the expectation involves integrals:

$$D_{KL}(p\|q) = \int_{- \infty}^{\infty} p(x) \log(\frac{p(x)}{q(x)}) dx$$

In order to understand why KL divergence is not symmetrical, it is useful to think of a simple example of a dice and a coin (let's indicate head and tails by 0 and 1). Both are fair and thus their PDF is uniform. Their entropy is trivially: $H(p_{coin})=log(2)$ and $H(p_{dice}=log(6))$. Let's first consider $D_{KL}(p_{coin}\|p_{dice})$. The 2 possible events of $X_{dice}$ are 0,1 which are also possible in the dice. The average number of bits to encode a coin observation under the dice encoding, will thus simply be $log(6)$, and the KL divergence is of $log(6)-log(2)$ additional bits. Now let's consider the problem the other way around: $D_{KL}(p_{dice}\|p_{coin})$. We will use $log(2)=1$ bit to encode the events of 0 and 1. But how many bits will we use to encode $3,4,5,6$ ? Well the optimal encoding for the dice doesn't have any encoding for these as they will never happen in his world. The KL divergence is thus not defined (division by 0). With this example you should clearly understand that the additional bits required to encode $p$ with $q$ is not the same as encoding $q$ with $p$. The KL divergence is thus not symmetric and cannot be a distance.

:mag: <span class="note"> Side notes </span> : Minimizing cross entropy with respect to $1$ is the same as minimizing $D_{KL}(p\|q)$. Indeed the 2 equations are equivalent up to an additive constant (the entropy of $p$) which doesn't depend on $q$.

##### Machine Learning and Entropy
This is all interesting, but why are we talking about information theory concepts in machine learning :sweat_smile: ? Well it turns our that many ML algorithms can be interpreted with entropy related concepts.

The 2 major ways we see entropy in machine learning are through:

* **Maximizing information gain** (i.e entropy) at each step of our algorithm. <span class="exampleText">Example</span>:
	
	* When building <span class="exampleText">decision trees you greedily select split which maximizes information gain</span> (i.e the difference of entropy before and after the split). Intuitively you want to minimize the number of splits you should do afterwards to correctly classify the observation.

* **Minimizing KL divergence between the actual unknown probability distribution of observations $p$ and the predicted one $q$**. <span class="exampleText">Example</span>:

	* The Maximum Likelihood Estimator (MLE) of our parameters <span class="exampleText"> $\hat{ \theta }_{MLE}$ are also the parameter which minimizes the KL divergence between our predicted distribution and the actual unknown one </span> . I.e 

$$\hat{ \theta }_{MLE} = argmin_{ \theta } \ NLL= argmin_{ \theta } \ D_{KL}(p(X| \theta )\|p(X| \theta ))$$

* **Minimizing  KL divergence between the computationally intractable $p$ and a simpler approximation $q$**. Indeed machine learning is not only about theory but also about how to make something work in practice.<span class="exampleText">Example</span>:

	* This is the whole point of <span class="exampleText"> **Variational Inference** (= variational Bayes) which approximates posterior probabilities of unobserved variables that are often intractable due to the integral in the denominator. Thus turning the inference problem to an optimization one</span>. These methods are an alternative to Monte Carlo sampling methods for inference (ex: Gibbs Sampling). In general sampling methods are slower but asymptotically exact.


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

**Hyper-parameter vs Parameter**

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
* **Multi-Label:** If labels are not mutually exclusive. Often replaced by $$K$$ binary classification specifying whether an observation should be assigned to each class.

Common evaluation metrics:
* **Accuracy**
* **Confusion Matrix**
* **Accuracy**
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
##### Support Vector Machines (SVM)
##### Artificial Neural Networks
#### Generative Classifiers
##### Naive Bayes

## Unsupervised Learning

### Clustering

Nota Bene: Careful when ensemble learning to label switching
#### K-Nearest Neighbors
#### K-Means
#### Density Based Clustering
#### Hierarchical Clustering
#### Spectral Clustering

### Latent Factor Model
#### Dimensionality reduction
##### Principal Component Analysis (PCA)
##### Independent Component Analysis (ICA)
##### Sammon Mapping
##### Multidimensional Scaling (MDS)
##### Projection Pursuit
##### Linear Discriminant Analysis (LDA)
##### ISOMAP
##### T-SNE
##### Autoencoders

### Density Estimation
#### Collaborative Filtering

Recommender systems (also content based or hybrid)


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

### Undirected Graphical Models

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
## Computational Neuroscience
### Spiking Neural Networks

## Optimization
### Evolutionary Methods

## Bayesian Optimization

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
