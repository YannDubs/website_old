*Supervised learning tasks tackle problems that have labeled data.*

:bulb: <span class='intuition'> Intuition </span>: It can be thought of a teacher who corrects a multiple choice exam. At the end you will get the average result as well as a detailed result saying which answer was wrong what was the correct answer.

Supervised learning can be further separated into two broad type of problems:
* **Classification**: here the output variable $y$ is categorical. We are basically trying to assign one or multiple classes to an observation. Example: is it a cat or not ?
* **Regression**: here the output variable $y$ is continuous. Example : how tall is this person ?

### Classification
*The classification problem consists of assigning a set of classes/categories to an observation. I.e* $$x \mapsto y,\ y \in \{0,1,...,K\}$$

Classification problems can be further separated into:

* **Binary:** There are 2 possible classes. $$K=2,\ y \in \{0,1\}$$
* **Multi-Class:** There are more than 2 possible classes. $$K>2$$
* **Multi-Label:** If labels are not mutually exclusive. Often replaced by $$K$$ binary classification specifying whether an observation should be assigned to each class.

Common evaluation metrics include Accuracy, F1-Score, AUC... I have a [section devoted for these classification metrics](#classification-metrics){:.mdLink}.

:wavy_dash: <span class="compare"> Compare to </span> : 
[Regression](#regression){:.mdLink}


#### Decision Trees

<div>
<details open>
<summary>Overview</summary>

<div class="container-fluid">
  <div class="row text-center">
    <div class="col-xs-12 col-sm-6">
        <a href="#supervised-learning" class="infoLink">Supervised</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#supervised-learning" class="infoLink">Classification or Regression</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#generative-vs-discriminative" class="infoLink">Discriminative</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#parametric-vs-non-parametric" class="infoLink">Non-Parametric</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#discriminative-classifiers" class="infoLink">Non-Probabilistic</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <span class="info">Piecewise Linear Decision Boundary</span>
    </div>
  </div>
</div>

<div markdown='1'>
* :bulb: <span class='intuition'> Intuition </span> :
    * Split the training data based on “the best” question you can ask (ex: is he older than 27 ?). Recursively do the above while not happy with the classification results.
    * Decision trees are basically the algorithm to use for the "20 question" game. [Akinator](http://en.akinator.com/){:.mdLink} is a great algorithm that could have been implemented with decision trees. Akinator is probably based on fuzzy logic expert systems (as it can work with wrong answers) but you could do a simpler version with decision trees.
    * "Optimal" splits are found are found by maximization of [information gain](#machine-learning-and-entropy){:.mdLink} or similar methods.
* :wrench: <span class='practice'> Practical </span> :
    * "Use when you need a simple and interpretable model but the relationship between $y$ and $x$ is complex".
    * Training Complexity : <span class='practiceText' markdown='1'> $O(mnd + nd\log(n) )$ </span> . 
    * Testing Complexity : <span class='practiceText' markdown='1'> $O(mt)$ </span> .
    * Notation Used : $m=depth$ ; $$n= \#_{train}$$ ; $$d= \#_{features}$$ ; $$t= \#_{test}$$.
* :white_check_mark: <span class='advantage'> Advantage </span> :
    * <span class='advantageText'>  Interpretable </span> .
    * Few hyper-parameters.
    * Needs less data cleaning :
        * No normalization needed.
        * Can handle missing values.
        * Handles numerical and categorical variables.
    * Robust to outliers.
    * Doesn't make assumptions regarding the data distribution.
    * Performs feature selection.
    * Scales well.
* :x: <span class='disadvantage'> Disadvantage </span> :
    * Generally poor accuracy because greedy selection.
    * <span class='disadvantageText'> High variance</span> because if the top split changes, everything does.
    * Splits are parallel to features axes => need multiple splits to separate 2 classes with a 45° decision boundary.
    * No online learning.
</div>
</details>
</div> 
<p></p>

The basic idea behind building a decision tree is to :
1. Find an optimal split (feature + threshold). I.e the split which minimizes the impurity (maximizes information gain). 
2. Partition the dataset into 2 subsets based on the split above.
3. Recursively apply $1$ and $2$ this to each new subset until a stop criterion is met.
4. To avoid over-fitting: prune the nodes which "aren't very useful". 

Here is a little gif showing these steps: 
<div class="mediumWrap" markdown="1">
![Building Decision Trees CLassification](/img/blog/decision-tree-class.gif)
</div>

Note: For more information, please see the "*details*" and "*Pseudocode and Complexity*" drop-down below.

<div>
<details>
<summary>Details</summary> 
<div markdown='1'>
The idea behind decision trees is to partition the input space into multiple region. Ex: region of men who are more than 27 years old. Then predict the most probable class for each region, by assigning the mode of the training data in this region. Unfortunately, finding an optimal partitioning is usually computationally infeasible ([NP-complete](https://people.csail.mit.edu/rivest/HyafilRivest-ConstructingOptimalBinaryDecisionTreesIsNPComplete.pdf){:.mdLink}) due to the combinatorially large number of possible trees. In practice the different algorithms thus use a greedy approach. I.e each split of the decision tree tries to maximize a certain criterion regardless of the next splits. 

*How should we define an optimality criterion for a split?* Let's define an impurity (error) of the current state, which we'll try to minimize. Here are 3 possibilities of state impurities:

* **Classification Error**:  
    * :bulb: <span class='intuitionText'> The accuracy error : $1-Acc$</span> of the current state. I.e the error we would do by stopping at the current state.
    * $$ClassificationError = 1 - \max p_c$$

* **[Entropy](#entropy){:.mdLink}**:  
    * :bulb: <span class='intuitionText'> How unpredictable are the classes</span> of the current state. 
    * Minimize the entropy corresponds to maximizing the [information gain](#machine-learning-and-entropy){:.mdLink}.
    * $$Entropy = - \sum_c^K p_c log_2 p_c$$

* **Gini Impurity**:  
    * :bulb: <span class='intuitionText'> How unpredictable are the classes</span> of the current state. 
    * Minimize the entropy corresponds to maximizing the [information gain](#machine-learning-and-entropy){:.mdLink}.
    * $$ClassificationError =  \sum_c^K p_c (1-p_c) = 1- \sum_c^K p_c^2$$

Here is a quick graph showing the impurity depending on a class distribution in a binary setting:

<div class="mediumWrap" markdown="1">
![Impurity Measure](/img/blog/impurity.png)
</div>

:mag: <span class='note'> Side Notes </span>: 

* Classification error may seem like a natural choice, but don't get fooled by the appearances: it's generally worst than the 2 other methods:
    *  It is "more" greedy than the others. Indeed, it only focuses on the current error, while Gini and Entropy try to make a purer split which will make subsequent steps easier. <span class='exampleText'> Suppose we have a binary classification with 100 observation in each class $(100,100)$. Let's compare a split which divides the data into $(20,80)$ and $(80,20)$, to an other split which would divide it into $(40,100)$ and $(60,0)$. In both case the accuracy error would be of $0.20\%$. But we would prefer the second case, which is **pure** and will not have to be split further. Gini impurity and the Entropy would correctly chose the latter. </span> 
    *  Classification error doesn't look at all classes, only the most probable one. So having a split with 2 extremely probable classes would be the same than having a split with one extremely probable class and many improbable ones.
* Gini Impurity and Entropy [give very similar results](https://www.garysieling.com/blog/sklearn-gini-vs-entropy-criteria#footnote_4_2094){:.mdLink} as you can see in the plot above. Chose one and stick with it.

*But when should we stop ?* If possible, is important to stop soon to decrease over-fitting. There are a variety of different heuristics to determine when to stop. Here are a few:

* When the number of training examples is small enough.
* When the depth reaches a threshold.
* If the impurity is low enough.
* If the purity gain due to the split is too small.

One problem with these, are the multiple problem-dependent thresholds (hyperparameters) to chose. Furthermore, some heuristics might yield relatively bad results. For example decision trees might have to split the data without any purity gain, to reach high purity gain at the following step. For these reasons, it is common to grow large trees using only the number of training example in a leaf node as a stopping criterion. To avoid over-fitting, the algorithm would prune back the resulting tree. In CART, the pruning criterion $C_{pruning}(T)$ tries to balance impurity and model complexity by regularization. The regularized variable is often the number of leaf nodes $\|T\|$, as below:

$$C_{pruning}(T) = \sum^{|T|}_{v=1} I(T,v) + \lambda |T|$$

Where $\lambda$ determines the trade-off between impurity and model complexity, for a given tree $T$, with leaf nodes $v=1...\|T\|$ using Impurity measure $I$. Then you would simply chose $\lambda$ using [cross validation](#cross-validation){:.mdLink}.

**Variants**: there are multiple different decision tree methods. They mostly differ with regards to the following points:

* Splitting Criterion ? Gini / Entropy.
* Technique to Reduce Over-fitting ?
* How many variables can be used in a split ?
* Do they build Binary Trees ?
* How they handle Missing Values ?
* Do they handle Regression?
* Are they susceptible to outliers?

The most famous variants are:
* **ID3**: first decision tree implementation. Not used in practice. 
* **C4.5**: Improvement over ID3 by the same developer. Error based pruning. Uses entropy. Handles missing values. Susceptible to outliers. Can create empty branches.
* **CART**: Uses Gini.  Cost complexity pruning. Binary trees. Handles missing values. Handles regression. Not susceptible to outliers.
* **CHAID**: FInds a splitting variable using Chi-squared to test the dependency between a variable and a response. No pruning. Seems better for describing the data, but worst for predicting.

Other variants include : C5.0 (next version of C4.5, probably less used because patented), MARS.

:information_source: <span class='resources'> Resources </span> : A comparative study of [different decision tree methods](http://www.academia.edu/34100170/Comparative_Study_Id3_Cart_And_C4.5_Decision_Tree_Algorithm_A_Survey){:.mdLink}.
</div>
</details>
</div> 
<p></p>

<div>
<details>
<summary>Pseudocode and Complexity</summary>
<div markdown='1'>

* **Pseudocode**
The simple version of a decision tree can be written in a few lines of python pseudocode:

```python
def buildTree(X,Y):
    if stop_criteria(X,Y) :
        # if stop then store the majority class
        tree.class = mode(X) 
        return Null

    minImpurity = infinity
    bestSplit = None
    for j in features:
        for T in thresholds:
            if impurity(X,Y,j,T) < minImpurity:
                bestSplit = (j,T)
                minImpurity = impurity(X,Y,j,T) 

    X_left,Y_Left,X_right,Y_right = split(X,Y,bestSplit)

    tree.split = bestSplit # adds current split
    tree.left = buildTree(X_left,Y_Left) # adds subsequent left splits
    tree.right buildTree(X_right,Y_right) # adds subsequent right splits

return tree

def singlePredictTree(tree,xi):
    if tree.class is not Null:
        return tree.class

    j,T = tree.split
    if xi[j] >= T:
        return singlePredictTree(tree.right,xi)
    else:
        return singlePredictTree(tree.left,xi)

def allPredictTree(tree,Xt):
    t,d = Xt.shape
    Yt = vector(d)
    for i in t:
        Yt[i] = singlePredictTree(tree,Xt[i,:])

    return Yt
```

* **Complexity**
I will be using the following notation: $$m=depth$$ ; $$T=\#_{thresholds}$$ ; $$n = \#_{train}$$ ; $$d = \#_{features}$$ ; $$t = \#_{test}$$ . 

Let's first think about the complexity for building the first decision stump (first function call):

* In a decision stump, we loop over all features and thresholds $O(td)$, then compute the impurity. The impurity depends solely on class probabilities. Computing probabilities means looping over all $X$ and count the $Y$ : $O(n)$. With this simple pseudocode, the time complexity for building a stump is thus $O(tdn)$. 
* In reality, we don't have to look for arbitrary thresholds, only for the unique values taken by at least an example. Ex: no need of testing $feature_j>0.11$ and $feature_j>0.12$ when all $feature_j$ are either $0.10$ or $0.80$. Let's replace the number of possible thresholds $t$ by training set size $n$. $O(n^2d)$
* Currently we are looping twice over all $X$, once for the threshold and once to compute the impurity. Instead, if the data was sorted by the current feature, the impurity could simply be updated as we loop through possible thresholds. Ex: when considering the rule $feature_j>0.8$ after having already considered $feature_j>0.7$, we do not have to recompute all the class probabilities: we can simply take the probabilities from $feature_j>0.7$ and make the adjustments knowing the number of example with $feature_j==0.7$. For each feature $j$ we should first sort all data $O(n\log(n))$ then loop once in $O(n)$, the final would be in $O(dn\log(n))$.

We now have the complexity of a decision stump. You could think that finding the complexity of building a tree would be multiplying it by by the number of function calls: Right ? Well not really, that would be an over-estimate. Indeed, at each function call, the training data size $n$ would have decreased. The intuition for the result we are looking for, is that at each level $l=1...m$ the sum of the training data in each function is still $n$. Multiple function working in parallel take the same time as a single function would, with the whole training set $n$. The complexity at each level is thus still $O(dn\log(n))$ so the complexity for building a tree of depth $m$ is $O(mdn\log(n))$. If you you want a proof that the work at each level stays constant:

At each iterations the dataset is split into $\nu$ subsets of $k_i$ element and a set of $n-\sum_{i=1}^{\nu} k_i$. At every level, the total cost would therefore be (using the well known log propriety and the fact that $k_i \le n$ ) : 

$$
\begin{align*}
cost &= O(k_1d\log(k_1)) + ... + O((n-\sum_{i=1}^{\nu} k_i)d\log(n-\sum_{i=1}^{\nu} k_i))\\
    &\le O(k_1d\log(n)) + ... + O((n-\sum_{i=1}^{\nu} k_i)d\log(n))\\
    &= O(((n-\sum_{i=1}^{\nu} k_i)+\sum_{i=1}^{\nu} k_i)d\log(n)) \\
    &= O(nd\log((n))   
\end{align*} 
$$

The last possible adjustment I see, is to sort everything once, store it and simply use this precomputed data at each level. The final training complexity is therefore <span class='practiceText'> $O(mdn + nd\log(n))$ </span> .

The time complexity of making predictions is straightforward: for each $t$ examples, go through a question at each $m$ levels. I.e <span class='practiceText'> $O(mt)$ </span> .
</div>
</details>
</div> 
<p></p>



### Regression
#### Decision Trees
Decision trees are more often used for classification problems. I thus talk at length about them [here](#decision-trees-1){:.mdLink}.

The 2 differences with decision trees for classification are:
* **What error to minimize for an optimal split?** This replaces the impurity measure in the classification setting. An easy error function for regression which has the nice interpretation of being linked to variance, is the [sum of squared error](#mean-squared-error){:.mdLink}. Note that we don't use the mean squared error because when computing the error/variance reduction wafter a split we want to subtract the sum of errors after the split from the error before the split. Sum of squared error for region $R$:

$$Error = \sum_{x_i \in R} (y_i - \bar{y_{R}})^2$$

* **What to predict for a given space region?** Before we were predicting the mode of the subset of training data in this space. Taking the mode doesn't make sense for a continuous variable. Now that we've defined an error function above, we would like to predict a value which minimizes this sum of squares error function. The values which minimizes this error function is simply the **average** of the values of the points in this region. Thankfully, predicting the mean is intuitively what we would have done. 

The rest is the same as in the classification setting. Let's look at a simple plot to get a better idea of the algorithm:

<div class="mediumWrap" markdown="1">
![Building Decision Trees Regression](/img/blog/decision-tree-reg.gif)
</div>

:x: Besides the disadvantages seen in the [decision trees for classification](#decision-trees-1){:.mdLink}, decision trees for regression suffer from the fact that it predicts a <span class='disadvantageText'> non smooth function  </span>.