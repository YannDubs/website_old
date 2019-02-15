
## General Machine Learning Concepts
### Bias Variance Trade-off
#### Regularization
### Common Distributions
#### Discrete
* Bernouilli 
* Binomial
* Multinouilli 
* Multinomial
* Poisson
* Empirical

#### Continuous
* Gaussian
* Student t
* Laplace 
* Gamma
* Beta
* Pareto
* Dirichlet
### Curse of Dimensionality
----
### Decision Theory
### Ensemble Learning
#### Boosting
#### Bootstrapped Aggregation (Bagging)
#### Stacked Generalization (Blending)
#### Averaging Generalization 
### Estimators
#### Maximum Likelihood Estimation
#### Maximum A Posteriori Estimation (MAP)
#### Bayesian Modeling
### Evaluation Metrics
#### Classification Metrics
----
#### Regression Metrics
### Frequentist vs Bayesian
* :mag: <span class='note'> Side Notes </span> :

    * Don't get mistaken: using Bayes rule doesn't make you a Bayesian. As my previous ML professor [Mark Schmidt](https://www.cs.ubc.ca/~schmidtm/){:.mdLink} used to say: "If you're not integrating, you're not a Bayesian". :sweat_smile:

    * If you understood well the point of view of frequentist, you might be surprised of seeing something like $p(x\| \Theta)$, which means the "conditional distribution of *x* given $\Theta$". Indeed for frequentists $\Theta$ is not a random variable and thus conditioning on it makes no sense (there's a single value for $\Theta$ which may be unknown but is still fixed: it's value is thus not a condition). Frequentists would thus write such distributions: $p(x;\Theta)$ which means "the distribution of *x* parameterized by $\Theta$". In statistics and machine learning, most people use $\|$ for both cases. Mathematicians tend to differentiate between the notations. In this blog, I will use $\|$ for both cases in order to keep the same notation as other ML resources you will find. 

### Generative vs Discriminative 
-----
### Information Theory
-----
### Model Selection
#### Cross Validation
#### Hyperparameter Optimization
### Monte Carlo Estimation
### No Free Lunch Theorem
----
### Parametric vs Non Parametric
-----
### Quick Definitions
**Capacity**
**Convex functions**
**Exponential Family**
**Inference**
**Kernels**
**Norms**
**Online learning**
**Surrogate Loss Function**







## Supervised Learning
### Classification
#### Decision Trees
----
#### Logistic Regression 
<div class="container-fluid">
  <div class="row text-center">
    <div class="col-xs-12 col-sm-6">
        <a href="#supervised-learning" class="infoLink">Supervised</a>
    </div>
    <div class="col-xs-12 col-sm-6">
            <a href="#supervised-learning" class="infoLink">Classification</a>
        </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#generative-vs-discriminative" class="infoLink">Discriminative</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#parametric-vs-non-parametric" class="infoLink">Parametric</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#discriminative-classifiers" class="infoLink">Probabilistic</a>
    </div>
  </div>
</div>
#### Softmax 
<div class="container-fluid">
  <div class="row text-center">
    <div class="col-xs-12 col-sm-6">
        <a href="#supervised-learning" class="infoLink">Supervised</a>
    </div>
    <div class="col-xs-12 col-sm-6">
            <a href="#supervised-learning" class="infoLink">Classification</a>
        </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#generative-vs-discriminative" class="infoLink">Discriminative</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#parametric-vs-non-parametric" class="infoLink">Parametric</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#discriminative-classifiers" class="infoLink">Probabilistic</a>
    </div>
  </div>
</div>
#### Support Vector Machines (SVM)
<div class="container-fluid">
  <div class="row text-center">
    <div class="col-xs-12 col-sm-6">
        <a href="#supervised-learning" class="infoLink">Supervised</a>
    </div>
    <div class="col-xs-12 col-sm-6">
            <a href="#supervised-learning" class="infoLink">Classification</a>
        </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#generative-vs-discriminative" class="infoLink">Discriminative</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#parametric-vs-non-parametric" class="infoLink">Parametric</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#discriminative-classifiers" class="infoLink">Non-Probabilistic</a>
    </div>
  </div>
</div>

* :mag: <span class='note'> Side Notes </span> :
    * There are extensions such as [Platt scaling](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639){:.mdLink} to interpret SVM in a probabilistic manner.

#### Gaussian Process
  
----
Now the Generative classifiers

----

#### Gaussian Discriminant Analysis
#### Gaussian Mixture Model
#### Latent Dirichlet Allocation 


### Regression
#### Linear Models
##### Linear Regression
Normally use OLS but could use something else to estimate
##### Ordinary Least Squares
estimation technique: often used in Linear regression
#### Decision Trees
----
#### Gaussian Processes

### Ranking
## Unsupervised Learning

### Clustering
Nota Bene: Careful when ensemble learning to label switching
#### Density Based Clustering
#### Hierarchical Clustering
#### K-Means
#### K-Nearest Neighbors (KNN)
#### Spectral Clustering

### Density Estimation
#### Collaborative Filtering
Recommender systems (also content based or hybrid)

### Dimensionality Reduction
#### Autoencoders
#### Independent Component Analysis (ICA)
#### ISOMAP
#### Linear Discriminant Analysis (LDA)
#### Multidimensional Scaling (MDS)
#### Principal Component Analysis (PCA)
#### Projection Pursuit
#### Sammon Mapping
#### Self-Organizing Maps
#### T-SNE


### Outlier Detection
Nota Bene: distinguish global and local outliers
##### Cluster-Based
##### Distance-Based
##### Graphical Approaches
##### Model-Based


## Reinforcement Learning
### Model Based RL
### Model Free RL


## Partially supervised learning
non t called like that
### Active Learning
### Semi-supervised learning


## Bayesian Methods


## Deep Learning
### Backpropagation
### Feed-forward Neural Networks
### Convolution Neural Network
### Recurrent Neural Network
### Autoencoders
### Regularization
* Early Stopping
* Dropout
* Multi-task Learning
* Norm Penalties

### Deep Generative Models
#### Variational Autoencoders
#### Generative Adversarial Networks
#### Neural Autoregressive 
#### Flow-based deep generative models


## Graphical Models
### Directed Graphical Models (Bayes Networks)
### Undirected Graphical Models (Markov Networks)
#### Pair Wise Undirected Graphical Models
#### Gibbs Networks
#### Conditional Random FIelds
- https://prateekvjoshi.com/2013/02/23/what-are-conditional-random-fields/
- generalization of multinomial listic regression to seuence prediction (i.e y's are dependent)
- https://www.youtube.com/watch?v=GF3iSJkgPbA
- http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/
- discriminative probabilisticc
- advantage over HMM for classification because learn p(y|x) instead of joint
- use for labeling sequences
- using conditional probabilities is udes to relax the conditional independence of naive bayes (like logistic regression) or the conditional indep required in HMM
- HMM are subcase of CRF  where constant probabilities are used to model state transitions. 


## Natural Language Processing


## Time Series


## Optimization
### Evolutionary Methods
Compare to RL : If the space of policies is sufficiently small, or can be structured so that good policies are common or easy to find or if a lot of time is available for the search then evolutionary methods can be effective. In addition, evolutionary methods have advantages on problems in which the learning agent cannot sense the complete state of its environment.


## Computational Neuroscience
### Spiking Neural Networks


## Other
### Causal Learning
#### Counter Factuals
### State Space Models
### Optimal control 
https://vincentherrmann.github.io/blog/wasserstein/
we look at the minimal amount of energy to move one distribution to an other (it's similar to KL divergence but instead of looking at the difference in the y axis (i.e difference of probability given to each point in both distribution) you look at the distance in x axis (i.e how much energy does it take to make one one distribution turn itno an other). For example if there are 2 very non linear pdf that are the same but shifted by a small amout, the KL might be very big (because difference of probability given to each point might be big) but not the earth moving disatnce) 










*Nota Bene: these terms are not always the most important ones but important ones I have encountered since my "migration" to machine learning / computer science in September 2016.*

Thanks to [Mark Schmidt](https://www.cs.ubc.ca/~schmidtm/){:.mdLink}, my Machine Learning professor, who introduced me to this amazing field.



-----
Additional ----
#### Discriminative Classifiers

I devoted a [section](#generative-vs-discriminative){:.mdLink} to discriminative classifiers but in summary these are the algorithms that directly learn a (decision) boundary between classes.

As a reminder these can be either:
* **Probabilistic**: the algorithm has a probabilistic interpretation: it tries to model $p(y\|x)$.
* **Non-Probabilistic**: the model cannot me interpreted with probabilities: it simply "draws" a boundary which you will simply use to predict. 


-----
Other concepts ??
- Maximum Mean Discrepancy 
