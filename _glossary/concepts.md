### Evaluation Metrics
#### Classification Metrics
##### Single Metrics
{:.no_toc}

:mag: <span class='note'> Side Notes </span> : I will mostly focus on binary classification but most scores can be generalized to the multi-class setting. Often this is achieved by only considering "correct class" and "incorrect class" in order to make it a binary classification, then you average (weighted by the proportion of observation in the class) the score for each classes.

* **TP** / **TN** / **FN** / **FP:** The best way to understand these is to look at a $$2*2$$ [confusion matrix](#visual-metrics){:.mdLink}.

<div class="mediumWrap" markdown="1">
![confusion matrix](/img/blog/confusion-matrix.png)
</div>

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

  <div class="mediumWrap" markdown="1">
  ![log loss](/img/blog/log-loss.png)
  </div>

* **AUC** **A**rea **U**nder the **C**urve. Summarizes curves in a simple single metric.
  * It normally refers to the [ROC](#visual-metrics){:.mdLink} curve. Can also be used for other curves like the precision-recall one.
  * :bulb: <span class='intuitionText'> Probability that a randomly selected positive has a higher score than a randomly selected negative observation </span> .
  * :mag: <span class='noteText'> AUC evaluates results at all possible cut-off points. It gives better insights about how well the classifier is able to separate between classes </span>. This makes it very different from the other metrics above that depend on the cut-off threshold.
  * :wrench: <span class='practiceText'> Use when building a classifier for users that will have different needs (they could tweak the cut-off point)</span> . From my experience AUC is often used in statistics (~go-to metric in bio-statistics) but less so in machine learning.
  * Random predictions: $AUC = 0.5$. Perfect predictions: $AUC=1$.
 
##### Visual Metrics
{:.no_toc}

* **Confusion Matrix** a $K*K$ matrix which shows the number of observation of class $c$ that have been labeled $c'$ $\forall c,c' \in 1,...,K$
    * :mag: <span class='noteText'> Be careful: People are not consistent with the axis :you can find real-predicted and predicted-real  </span> .
    * This is best understood through an example:

    <div class="mediumWrap" markdown="1">
    ![Multi Confusion Matrix](/img/blog/multi-confusion-matrix.png)
    </div>


* **ROC Curve** **R**eceiver **O**perating **C**haracteristic
  * Plot showing the TP rate vs the FP rate, over a varying threshold.
  * This plot from [wikipedia](https://commons.wikimedia.org/wiki/File:ROC_curves.svg){:.mdLink} shows it well:
  
<div class="mediumWrap" markdown="1">
![ROC curve](/img/blog/ROC.png)
</div>

:information_source: <span class="resources"> Resources </span>: [Additional scores based on confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix){:.mdLink}

### Generative vs Discriminative 
These two major types of models, distinguish themselves by the approach they are taking to learn. Although these distinctions are not specific to a particular task, you will most often hear about the distinction between [generative](#generative-classifiers){:.mdLink} and [discriminative](#discriminative-classifiers){:.mdLink} [classifiers](#classification){:.mdLink}.

#### Differences
{:.no_toc}

In [classification](#classification){:.mdLink}, the task is to identify the category $y$ of an observation, given its features $x$. In mathematical notation we are looking for $y\|x$. There are 2 approaches, to this problem:

* **Discriminative** learn the *boundaries* between classes, called the decision boundaries.
    * :bulb: <span class='intuitionText'> Simply tell me in which class is this observation given past data</span>. 
    * Can be **probabilistic** or **non-probabilistic**. If probabilistic, it tries to model **$p(y\|x)$** and give label $y$ for which $p(y\|x)$ is maximum. If non probabilistic: simply "draws" a boundary between classes, if on one side then class A if on the other then B (easily generalizes for multiple class).
    * Directly models what we care about: **$y\|x$**.
    * :school_satchel: As an example for detecting languages from a conversation, the  discriminative model would learn to <span class='exampleText'>distinguish between languages from their sound but wouldn't understand anything</span>.

* **Generative** model the *distribution* of each classes.
    * :bulb: <span class='intuitionText'> First understand what this data means, then use your knowledge to classify</span>. 
    * First, model the joint distribution **$p(y,x)$** (normally through $p(y,x)=p(x\|y)p(y)$). Then find the conditional probability we are looking for, through Bayes theorem: $p(y\|x)=\frac{p(y,x)}{p(x)}$. Finally find $y$ which maximizes $p(y\|x)$ (same as discriminative).
    * Computes more information than discriminative classifiers. Thus more general.
    * :school_satchel: To continue with the previous example, the generative model would first <span class='exampleText'>learn how to speak the language and then say from which language the words come from</span>.

#### Pros / Cons
{:.no_toc}

Please note that some of advantages / disadvantages mean the same thing but are worded differently.

* **Discriminative**:
    <ul style="list-style: none;">
      <li > :white_check_mark: <span class="advantageText"> Less bias => better if more data.</span> </li>
      <li > :white_check_mark: <span class="advantageText"> Less model assumptions</span>  as it's tackling an easier problem. </li>
      <li > :x:<span class="disadvantageText"> Slower convergence rate </span>. Logistic Regression requires $O(d)$ observations. </li>
      <li > :x: <span class="disadvantageText"> Prone to over-fitting </span> when there's less data, as it doesn't make assumptions to constrain it from finding inexistent patterns.  </li>
      <li > :x: <span class="disadvantageText"> More variance. </span> </li>
      <li > :x: <span class="disadvantageText"> Hard to update the model </span> with new data (online learning). </li>
      <li > :x: <span class="disadvantageText"> Have to retrain model when adding new classes. </span> </li>
      <li > :x: <span class="disadvantageText"> In practice needs additional regularization / kernel / penalty functions.</span> </li>
    </ul >


* **Generative** 
  <ul style="list-style: none;">
      <li > :white_check_mark: <span class="advantageText"> Faster convergence rate => better if less data </span>. Naive Bayes only requires $O(\log(d))$ observations. </li>
      <li > :white_check_mark: <span class="advantageText"> Less variance. </span> </li>
      <li > :white_check_mark: <span class="advantageText"> Can easily update the model  </span> with new data (online learning).  </li>
      <li > :white_check_mark: <span class="advantageText"> Can generate new data </span> by looking at $p(x|y)$.  </li>
      <li > :white_check_mark: <span class="advantageText"> Can handle missing features</span> .  </li>
      <li > :white_check_mark: <span class="advantageText"> You don't need to retrain model when adding new classes </span>  as the parameters of classes are fitted independently.</li>
      <li > :white_check_mark: <span class="advantageText"> Easy to extend to the semi-supervised case. </span>  </li>
      <li > :x: <span class="disadvantageText"> More Biais. </span> </li>
      <li > :x: <span class="disadvantageText"> Prone to under-fitting </span> when more there's data because of the multiple assumptions. </li>
      <li > :x: <span class="disadvantageText"> Uses computational power to compute something we didn't ask for.</span> </li>

    </ul >

:wrench: <span class='practice'> Rule of thumb </span>: If your problem is only to train the best classifier on a large data set, use a **discriminative model**. If your task involves more constraints (online learning, semi supervised learning, small data set, ...) use a **generative model**.

<div class="exampleBoxed" markdown="1">

Let's illustrate the advantages and disadvantage of both methods with an <span class='exampleText'> example </span> . Imagine we are asked to make a classifier for the "true distribution" below. As a training set, we are once given a "small sample" and an other time a "large sample".


<div class="col-xs-4" markdown="1">
![discriminative vs generative true distribution](/img/blog/discriminative-generative-true.png)
</div>

<div class="col-xs-4" markdown="1">
![discriminative vs generative small sample](/img/blog/discriminative-generative-small.png)
</div>

<div class="col-xs-4" markdown="1">
![discriminative vs generative large sample](/img/blog/discriminative-generative-large.png)
</div>

How well will the algorithms distinguish the classes in each case ?

* **Small Sample**:
    * The *discriminative* model never saw any examples at the bottom of the blue ellipse. It has no chance of finding the correct decision boundary there.
    * The *generative* model assumes that the data follows a normal distribution (ellipse). It will therefore be able to infer the correct decision boundary without ever having seen data points there!

<div class="col-xs-6" markdown="1">
![small sample discriminative](/img/blog/small-discriminative.png)
</div>

<div class="col-xs-6" markdown="1">
![small sample generative](/img/blog/small-generative.png)
</div>

.

* **Large Sample**:
    * The *discriminative* model doesn't have any assumption that restricts it of finding the small red cluster inside the blue one.
    * The *generative* model still assumes that the data follows a normal distribution (ellipse). It will therefore not be able to find the small red cluster.

<div class="col-xs-6" markdown="1">
![large sample discriminative](/img/blog/large-discriminative.png)
</div>

<div class="col-xs-6" markdown="1">
![large sample generative](/img/blog/large-generative.png)
</div>

Please note that this is simply an example. Some generative models would find the small red cluster: it all depends on the assumptions they are making. (I hope that) It still gives you a good idea of the advantages and disadvantages.
</div>

#### Examples of Algorithms
{:.no_toc}

##### Discriminative
* [Logistic Regression](#logistic-regression){:.mdLink}
* [Softmax](#softmax){:.mdLink}
* Traditional Neural Networks
* Conditional Random Fields
* Maximum Entropy Markov Model
* [Decision Trees](#decision-trees){:.mdLink}


##### Generative
* [Naives Bayes](#naive-bayes){:.mdLink}
* Gaussian Discriminant Analysis
* Latent Dirichlet Allocation
* Restricted Boltzmann Machines
* Gaussian Mixture Models
* Hidden Markov Models 
* Sigmoid Belief Networks
* Bayesian networks
* Markov random fields

##### Hybrid
* Generative Adversarial Networks
* "Discriminative Training" [from this recent paper](:information_source:){:.mdLink}

:information_source: <span class='resources'> Resources </span> : A. Ng and M. Jordan have a [must read paper](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf){:.mdLink} on the subject, T. Mitchell summarizes very well these concepts in [his slides](http://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/07_GenDiscr2_2-4-2015.pdf){:.mdLink}, and section 8.6 of [K. Murphy's book](https://www.cs.ubc.ca/~murphyk/MLbook/){:.mdLink} has a great overview of pros and cons, which strongly influenced the devoted section above.

### Information Theory

#### Entropy

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


<details open>
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

:information_source: <span class="resources"> Resources </span>: Excellent explanation of the link between [entropy in thermodynamics and information theory](http://www.askamathematician.com/2010/01/q-whats-the-relationship-between-entropy-in-the-information-theory-sense-and-the-thermodynamics-sense/){:.mdLink}, friendly [ introduction to entropy related concepts](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/){:.mdLink}

</div>
</details>
<p></p>

#### Differential Entropy
Differential entropy (= continuous entropy), is the generalization of entropy for continuous random variables.

Given a continuous random variable $X$ with a probability density function $f(x)$:

$$h(X) = - \int_{-\infty}^{\infty} f(x) \log {f(x)} \ dx$$

If you had to make a guess, which distribution maximizes entropy for a given variance ? You guessed it : it's the **Gaussian distribution**.

:mag: <span class="note"> Side notes </span> : Differential entropy can be negative.

#### Cross Entropy
We [saw that](#entropy){:.mdLink} entropy is the expected number of bits used to encode an observation of $X$ under the optimal coding scheme. In contrast <span class="intuitionText"> cross entropy is the expected number of bits to encode an observation of $X$ under the wrong coding scheme</span>. Let's call $q$ the wrong probability distribution that is used to make a coding scheme. Then we will use $-log(q_i)$ bits to encode the $i^{th}$ possible values of $X$. Although we are using $q$ as a wrong probability distribution, the observations will still be distributed based on $p$. We thus have to take the expected value over $p$ :

$$H(p,q) = - \sum_i p_i \log(q_i)$$

From this interpretation it naturally follows that:
* $H(p,q) > H(p), \forall q \neq p$
* $H(p,p) = H(p)$

:mag: <span class="note"> Side notes </span> : Log loss is often called cross entropy loss, indeed it is the cross-entropy between the distribution of the true labels and the predictions.

#### Kullback-Leibler Divergence
The Kullback-Leibler Divergence (= relative entropy = information gain) from $q$ to $p$ is simply the difference between the cross entropy and the entropy:

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

#### Machine Learning and Entropy
This is all interesting, but why are we talking about information theory concepts in machine learning :sweat_smile: ? Well it turns our that many ML algorithms can be interpreted with entropy related concepts.

The 2 major ways we see entropy in machine learning are through:

* **Maximizing information gain** (i.e entropy) at each step of our algorithm. <span class="exampleText">Example</span>:
	
	* When building <span class="exampleText">decision trees you greedily select to split on the attribute which maximizes information gain</span> (i.e the difference of entropy before and after the split). Intuitively you want to chose to know the value of the attribute, which would decrease the randomness in your data by the largest amount.

* **Minimizing KL divergence between the actual unknown probability distribution of observations $p$ and the predicted one $q$**. <span class="exampleText">Example</span>:

	* The Maximum Likelihood Estimator (MLE) of our parameters <span class="exampleText"> $\hat{ \theta }_{MLE}$ are also the parameter which minimizes the KL divergence between our predicted distribution and the actual unknown one </span> . I.e 

$$\hat{ \theta }_{MLE} = argmin_{ \theta } \ NLL= argmin_{ \theta } \ D_{KL}(p(X| \theta )\|p(X| \theta ))$$

* **Minimizing  KL divergence between the computationally intractable $p$ and a simpler approximation $q$**. Indeed machine learning is not only about theory but also about how to make something work in practice.<span class="exampleText">Example</span>:

	* This is the whole point of <span class="exampleText"> **Variational Inference** (= variational Bayes) which approximates posterior probabilities of unobserved variables that are often intractable due to the integral in the denominator. Thus turning the inference problem to an optimization one</span>. These methods are an alternative to Monte Carlo sampling methods for inference (ex: Gibbs Sampling). In general sampling methods are slower but asymptotically exact.

  ### Parametric vs Non Parametric
  These 2 types of methods distinguish themselves based on their answer to the following question: "Will I use the same amount of memory to store the model trained on $100$ examples than to store a model trained on $10 000$ of them ? "
  If yes then you are using a *parametric model*. If not, you are using a *non-parametric model*.

  * **Parametric**:
      * :bulb: <span class='intuitionText'> The memory used to store a model trained on $100$ observations is the same as for a model trained on $10 000$ of them  </span>. 
      * I.e: The number of parameters is fixed.
      * :white_check_mark: <span class='advantageText'> Computationally less expensive </span> to store and predict.
      * :white_check_mark: <span class='advantageText'> Less variance. </span> 
      * :x: <span class='disadvantageText'> More bias.</span> 
      * :x: <span class='disadvantageText'> Makes more assumption on the data</span> to fit less parameters.
      * :school_satchel: <span class='example'> Example </span> : [K-Means](#k-means){:.mdLink} clustering, [Linear Regression](#linear-regression){:.mdLink}:
      
      <div class="smallWrap" markdown="1">
      ![Linear Regression](/img/blog/Linear-regression.png)
      </div>


  * **Non Parametric**: 
      * :bulb: <span class='intuitionText'> I will use less memory to store a model trained on $100$ observation than for a model trained on $10 000$ of them  </span>. 
      * I.e: The number of parameters is grows with the training set.
      * :white_check_mark: <span class='advantageText'> More flexible / general.</span> 
      * :white_check_mark: <span class='advantageText'> Makes less assumptions. </span> 
      * :white_check_mark: <span class='advantageText'> Less bias. </span> 
      * :x: <span class='disadvantageText'> More variance.</span> 
      * :x: <span class='disadvantageText'> Bad if test set is relatively different than train set.</span> 
      * :x: <span class='disadvantageText'> Computationally more expensive </span> as it has to store and compute over a higher number of "parameters" (unbounded).
      * :school_satchel: <span class='example'> Example </span> : [K-Nearest Neighbors](#k-nearest-neighbors){:.mdLink} clustering, RBF Regression:

      <div class="smallWrap" markdown="1">
      ![RBF Regression](/img/blog/RBF-regression.png)
      </div>

  :wrench: <span class='practice'> Practical </span> : <span class='practiceText'>Start with a parametric model</span>. It's often worth trying a non-parametric model if: you are doing <span class='practiceText'>clustering</span>, or the training data is <span class='practiceText'>not too big but the problem is very hard</span>.

  :mag: <span class='note'> Side Note </span> : Strictly speaking any non-parametric model could be seen as a infinite-parametric model. So if you want to be picky: next time you hear a colleague talking about non-parametric models, tell him it's in fact parametric. I decline any liability for the consequence on your relationship with him/her :sweat_smile: . 

