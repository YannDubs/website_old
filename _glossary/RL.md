*In Reinforcement Learning (RL), the sequential decision-making algorithm (an agent) **interacts** with an environment it is **uncertain** about. The agent learns to map situations to actions to maximize a long term reward. During training, the action it choses are evaluated rather than instructed.*

:school_satchel: <span class='example'> Example</span>: Games can very naturally be framed in a RL framework. For example, when playing tennis you are not told how good every movement you make is, but you are given a certain reward if you win the whole game.  

:mag: <span class='note'> Side Notes</span>: Games could also be framed in a supervised problem. The training set would consist in many different states of the environment and the optimal action to take in each of those. Creating such a dataset is not possible for most applications as it requires to enumerate the exponential number of states and to know the associated best action (*e.g.* exact rotation of all your joints when you play tennis). Note that during supervised training, the feedback indicates the correct action independently to the chosen action. The RL framework is a lot more natural as the agent is trained by playing the game. Importantly, the agent interacts with the environment such that the states that it will visit depend on previous actions. So it is a chicken-egg problem where it will unlikely reach good states before being trained, but it has to reach good states to get reward and train effectively. This leads to training curves that start with very long plateaus of low reward until it reaches a good state (somewhat by chance) and then learn quickly. In contrast, supervised methods have very steep loss curves at the start.

:information_source: <span class='resources'> Resources </span> : The link and differences between supervised and RL is described in details by [A. Barto and T. Dietterich](http://www-anw.cs.umass.edu/pubs/2004/barto_d_04.pdf){:.mdLink}. 

In RL, future states depend on current actions, thus requiring to model indirect consequences of actions and planning. Furthermore, the agent often has to take actions in real-time while planning for the future.  All of the above makes it very similar to how humans learn, and is thus widely used in psychology and neuroscience. 

:information_source: <span class='resources'> Resources </span>  : All this section on RL is highly influenced by [Sutton and Barto's introductory book](http://incompleteideas.net/book/the-book-2nd.html){:.mdLink}.


### Markov Decision Process

Markov Decision Processes (MDPs) are a mathematical idealized form of the RL problem that suppose that states follow the **Markov Property**. *I.e.* that future states are conditionally independent of past ones given the present state: $S_{t+1} \perp \\{S_{i}\\}_{i=1}^{t-1} \vert S_t$.

Before diving into the details, it is useful to visualize how simple a MDP is (image taken from [Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html){:.mdLink}):

<div class="mediumWrap" markdown="1">
![Markov Decision Process](/img/blog/MDP.png)
</div>

Important concepts:

* **Agent**: learner and decision maker. This corresponds to the *controller* in [classical control theory](https://en.wikipedia.org/wiki/Classical_control_theory){:.mdLink}.
* **Environment**: everything outside of the agent. *I.e.* what it interacts with. It corresponds to the *plant* in classical control theory. Note that in the case of a human / robot, the body should be considered as the environment rather as the agent because it cannot be modified arbitrarily (the boundary is defined by the lack of possible control rather than lack of knowledge).
* **Time step** *t*: discrete time at which the agent and environment interact. Note that it doesn't have to correspond to fix real-time intervals. Furthermore, it can be extended to the continuous setting.
* **State** $S_t = s \in \mathcal{S}$ : information available to the agent about the environment.
* **Action** $A_t=a \in \mathcal{A}$ : action that the agent decides to take. It corresponds to the *control signal* in classical control theory.
* **Reward** $R_{t+1} = r \in \mathcal{R} \subset \mathbb{R}$: a value which is returned at each step by a (deterministic or stochastic) *reward signal* function depending on the previous $S_t$ and $A_t$. Intuitively, <span class='intuitionText'> the reward corresponds to current (short term) pain / pleasure that the agent is feeling</span>. Although the reward is computed inside the agent / brain, we consider them to be external (given by the environment) as they cannot be modified arbitrarily by the agent. <span class='noteText'> The reward signal we chose should truly represent *what* we want to accomplish </span>, not *how* to accomplish it (such prior knowledge can be added in the initial policy or value function). <span class='exampleText'> For example, in chess, the agent should only be rewarded if it actually wins the game. Giving a reward for achieving subgoals (*e.g.* taking out an opponent's piece) could result in *reward hacking* (*i.e.* the agent might found a way to achieve large rewards without winning)</span>.
* **Return** $G_t := \sum_{\tau=1}^{T-t} \gamma^{\tau-1} R_{t+\tau} \text{, with } \gamma \in [0,1[$: the expected discounted cumulative reward, which has to be maximized by the agent. Note that the *discounting factor* $\gamma$, has a two-fold use. First and foremost, it enables to have a finite $G_t$ even for *continuing tasks* $T=\infty$ (opposite of *episodic tasks*) assuming that $R_t$ is bounded $\forall t$. It also enables to encode the preference for rewards in the near future, and is a parameter that can be tuned to select the "farsightedness" of your agent ("myopic" agent with $\gamma=0$, "farsighted" as $\gamma \to 1$). <span class='noteText'> Importantly, the return can be defined in a recursive manner </span> : 

$$
\begin{aligned}
G_t &:= \sum_{\tau=1}^{T-t} \gamma^{\tau-1} R_{t+\tau} \\
&= R_{t+1} + \sum_{\tau=2}^{T-t} \gamma^{\tau-1} R_{t+\tau} \\
&= R_{t+1} + \sum_{\tau'=1}^{T-(t+1)} \gamma^{\tau' - 1+1} R_{t + 1+ \tau'} & & \tau' := \tau - 1 \\
&= R_{t+1} + \gamma G_{t+1} & & \text{Factorize a } \gamma
\end{aligned}
$$ 

* The **dynamics** of the MDP: $p(s', r\vert s, a) := P(S_{t+1}, R_{t+1} \vert S_t=s, A_t=a)$. In a MDP, this probability completely characterizes the network dynamics due to the Markov Property. Some useful functions that can be derived from it are:
    - *State-transition probabilities*
    
    $$p(s' \vert s, a) = \sum_{r} p(s',r \vert s,a)$$

    - *Expected rewards for state-actions pairs*:

    $$
    \begin{aligned}
    r(s, a) &:= \mathbb{E}[R_t \vert S_{t-1}=s, A_{t-1}=a] \\
    &= \sum_{r} r \sum_{s'} p(s',r \vert s,a)
    \end{aligned}
    $$ 

    - *Expected rewards for state-actions-next state triplets*: 

    $$
    \begin{aligned}
    r(s, a, s') &:= \mathbb{E}[R_t \vert S_{t-1}=s, A_{t-1}=a, S_{t}=s'] \\
    &= \sum_{r} r   p(s',r \vert s,a)
    \end{aligned}
    $$ 

* **Policy** $\pi(a\vert s)$: a mapping from states to probabilities over actions. Intuitively, it corresponds to a "<span class='intuitionText'>The behavioral function</span>" . Often called *stimulus-response* in psychology.
* **State-value function** for policy $\pi$, $v_\pi(s)$: expected return for an agent that follows a policy $\pi$ and starts in state $s$.   Intuitively, it corresponds <span class='intuitionText'> to how good it is to be in a certain state $s$ (long term)</span>. This function is not given by the environment but is often predicted by the agent. Similar to the return, the value function can also be defined recursively, by the very important **Bellman equation** : 

$$
\begin{aligned}
v_\pi(s) &:=\mathbb{E}[G_t \vert S_{t}=s]  \\
&=\mathbb{E}[R_{t+1} + \gamma G_{t+1} \vert S_{t}=s] & & \text{Recursive def. of return} \\
&= \sum_{s'} \sum_r \sum_{g_{t+1}} \sum_{a} p(s',r,g_{t+1},a \vert s) \left[ r + \gamma g_{t+1} \right] & & \text{Expectation over all R.V.} \\
&= \sum_{s'} \sum_r \sum_{g_{t+1}} \sum_{a} \pi(a \vert s) p(s', r\vert s, a) p(g_{t+1} \vert s', r, a, s) \left[ r + \gamma g_{t+1} \right] & & \text{Conditional indep.} \\
&= \sum_{s'} \sum_r \sum_{g_{t+1}} \sum_{a} \pi(a \vert s) p(s', r\vert s, a) p(g_{t+1} \vert s') \left[ r + \gamma g_{t+1} \right] & & \text{MDP assumption} \\
&= \sum_{a} \pi(a \vert s) \sum_{s'} \sum_r p(s', r\vert s, a) \left[ r  + \gamma  \sum_{g_{t+1}}   p(g_{t+1} \vert s')   g_{t+1} \right] \\
&= \sum_{a} \pi(a \vert s) \sum_{s'} \sum_r p(s', r\vert s, a) \left[ r  + \gamma  \mathbb{E}[G_{t+1} \vert S_{t+1}=s'] \right] \\
&= \sum_{a} \pi(a \vert s) \sum_{s'} \sum_r p(s', r\vert s, a) \left[ r  + \gamma  v_\pi(s') \right] \\
&= \mathbb{E}[R_{t+1} + \gamma  v_\pi(S_{t+1}) \vert S_{t}=s]  
\end{aligned}
$$ 

* **Action-value function** (*Q function*) for policy $\pi$, $q_\pi(s,a)$: expected total reward and agent can get starting from a state $s$. Intuitively, it corresponds <span class='intuitionText'> to how good it is to be in a certain state $s$ and take  specific action $a$ (long term)</span>. A **Bellman equation** can be derived in a similar way to the value function:

$$
\begin{aligned}
q_\pi(s,a) &:=\mathbb{E}[G_t \vert S_{t}=s, A_{t}=a]  \\
&= \sum_{s'} \sum_r p(s', r\vert s, a) \left[ r  + \gamma  \sum_{g_{t+1}}   p(g_{t+1} \vert s')   g_{t+1} \right] \\
&=  \sum_{s'} \sum_r p(s', r\vert s, a) \left[ r  + \gamma  v_\pi(s') \right]\\
&=  \sum_{s'} \sum_r p(s', r\vert s, a) \left[ r  + \gamma  \sum_{a'} \pi(a' \vert s') q_\pi(s',a') \right] \\
&= \mathbb{E}[R_{t+1} + \gamma \sum_{a'} \pi(a' \vert S_{t+1}) q_\pi(S_{t+1}, a') \vert S_{t}=s, A_{t}=a]  
\end{aligned}
$$ 


* **Model of the environment**: an internal model in the agent to predict the dynamic of the environment (*e.g.* probability of getting a certain reward or getting in a certain state for each action). This is only used by some RL agents for **planning**.


:mag: <span class='note'> Side Notes </span> : We usually assume that we have a **finite** MDP. *I.e.* that $\mathcal{R},\mathcal{A},\mathcal{S}$ are finite. Dealing with continuous state and actions pairs requires approximations. One possible way of converting a continuous problem to a finite one, is quantize the state and actions space.

#### Optimality Equations

Solving the RL tasks, consists in finding a good policy. A policy $\pi'$ is defined to be better than $\pi$ iff $v_{\pi'}(s) \geq v_{\pi}(s), \ \forall s \in \mathcal{S}$. The optimal policy $\pi_{\*}$ has an associated state *optimal value-function* and *optimal action-value function*:

$$v_*(s)=v_{\pi_*}(s):= \max_\pi v_\pi(s)$$

$$
\begin{aligned}
q_*(s,a) &= q_{\pi_*}(s,a) \\
&= \max_\pi q_\pi(s, a) \\
&= \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \vert S_{t}=s, A_{t}=a] 
\end{aligned}
$$ 

A special recursive update (the **Bellman optimality equations**) can be written for the optimal functions $v_{\*}$, $q_{\*}$ by taking the best action at each step instead of marginalizing:

$$
\begin{aligned}
v_*(s) &= \max_a q_{\pi_*}(s, a) \\
&= \max_a \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}, a') \vert S_{t}=s, A_{t}=a] \\
&= \max_a \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \vert S_{t}=s, A_{t}=a] \\
&= \max_a \sum_{s'} \sum_r p(s', r \vert s, a) \left[r + \gamma v_*(s') \right]
\end{aligned}
$$

$$
\begin{aligned}
q_*(s,a) &= \mathbb{E}[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \vert S_{t}=s, A_{t}=a]  \\
&= \sum_{s'} \sum_{r} p(s', r \vert s, a) \left[r + \gamma \max_{a'} q_*(s', a')\right]  
\end{aligned}
$$

The optimal Bellman equations are a set of equations ($\vert \mathcal{S} \vert$ equations and unknowns) with a unique solution. But these equations are now non-linear (due to the max). If the dynamics of the environment were known, you could solve the non-linear system of equation to get $v_{\*}$, and follow $\pi_{\*}$ by greedily choosing the action that maximizes the expected $v_{\*}$. By solving for $q_{\*}$, you would simply chose the action $a$ that maximizes $q_{\*}(s,a)$, which doesn't require to know anything about the system dynamics.

In practice, the optimal Bellman equations can rarely be solved due to three major problems:

* They require the **dynamics of the environment** which is rarely known.
* They require **large computational resources** (memory and computation) to solve as the number of states $\vert \mathcal{S} \vert$ might be huge (or infinite). This is nearly always an issue.
* The Markov Property.

Practical RL algorithms thus settle for approximating the optimal Bellman equations. Usually they parametrize functions and focus mostly on states that will be frequently encountered to make the computations possible.

### Dynamic Programming

Dynamic programming (DP), derives update rules for $v_\pi$ or $q_\pi$ from the Bellman equations to give rise to **iterative** methods that solve **exactly** the optimal control problem of finding $\pi_{\*}$.  These are thus a form of planning.

As discussed in the previous section they assume that the dynamics of the environment are known, that the environment is a finite MDP, and can only be used in practice if the number of states is not too large.

:wrench: <span class='practice'> Practical</span>:
* DP algorithms are guaranteed to find the optimal policy in polynomial time with respect to $\vert \mathcal{S} \vert$ and $$\vert \mathcal{A} \vert$$, even-though the number of possible deterministic policies is $\vert \mathcal{A} \vert ^ {\vert \mathcal{S} \vert}$. This exponential speedup comes from the MDP assumption.
* In practice, DP algorithm converge much faster than the theoretical worst case scenario. 

:mag: <span class='note'> Side Notes</span>: DP algorithms use the Bellman equations to update estimates based on other estimates (typically the value function at the next state). This general idea is called **Bootstrapping**.

#### Policy Iteration

The high-level idea is to iteratively: evaluate $v_\pi$ for the current policy $\pi$ (*policy evaluation* 1), use $v_\pi$ to improve $\pi'$ (*policy improvement* 1), evaluate $v_{\pi'}$ (*policy evaluation* 2)... Thus obtaining a sequence of strictly monotonically improving policies and value functions (except when converged to $\pi_{\*}$). As a finite MDP has only a finite number of possible policies, this is guaranteed to converge in a finite number of steps. This idea is often visualized using a 1d example taken from [Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html){:.mdLink}):

<div class="mediumWrap" markdown="1">
![Generalized Policy Iteration](/img/blog/generalized_policy_iteration.png)
</div>

This simplified diagram shows that although the policy improvement and policy evaluation "pull in opposite directions", the 2 processes still converge to find a single joint solution. <span class='noteText'> Almost all RL algorithms can be described using 2 interacting processes (for approximating the value and the policy), which are often called *Generalized Policy Iteration*</span>.

In python pseudo-code:

```python
def policy_iteration(environment):
    # v initialized arbitrarily for all states except V(terminal)=0
    v, pi = initialize()
    is_converged = False
    while not is_converged:
        v = policy_evaluation(pi, environment, v)
        pi, is_converged = policy_improvement(v, environment, pi)
    return pi
```

##### Policy Evaluation

The first step is to evaluate $v_\pi$ for a given $\pi$. This can be done by solving the Bellman equation :

$$
v_\pi(s) = \sum_{a} \pi(a \vert s) \sum_{s'} \sum_r p(s', r\vert s, a) \left[ r  + \gamma  v_\pi(s') \right] 
$$ 

Solving the equation can be done by either:

* **Linear System**: This is a set of linear equations ($\vert \mathcal{S} \vert$ equations and unknowns) with a unique solution (if $\gamma <1$ or if it is an episodic task). Note that we would to solve for these equations at every step of the policy iteration and $\vert \mathcal{S} \vert$ is often very large.

* **Iterative method**: Modify the Bellman equation to become an iterative method that is guaranteed to converge when $k\to \infty$ if $v_\pi$ exists. This is done by realizing that $v_\pi$ is a fixed point of:

$$
v_{k+1}(s) = \sum_{a} \pi(a \vert s) \sum_{s'} \sum_r p(s', r\vert s, a) \left[ r  + \gamma  v_{k}(s') \right], \ \forall s \in \mathcal{S}
$$ 

:mag: <span class='note'> Side Notes </span> : In the iterative method, we would have to keep to keep 2 arrays $v_k(s)$ and $v_{k+1}(s)$. At each iteration $k$ we would update $v_{k+1}(s)$ by looping through all states. Importantly, the algorithm also converges to $v_\pi(s)$ if we keep in memory a single array that would be updated "in-place" (often converges faster as it updates the value for some states using the latest available values). <span class='practiceText'> The order of state updates has a significant influence on the convergence in the "in-plase" case</span> .

We solve for an approximation $V \approx v_\pi$ by halting the algorithm when the change for every state is "small". 

In python pseudo-code:

```python
def q(s, a, v, environment, gamma):
    """Computes the action-value function `q` for a given state and action."""
    dynamics, states, actions, rewards = environment
    return sum(dynamics(s_p,r,s,a) * (r + gamma * v(s_p))
               for s_p in states
               for r in rewards)

def policy_evaluation(pi, environment, v_init, threshold=..., gamma=...):
    dynamics, states, actions, rewards = environment
    V = v_init # don't restart from scratch policy evaluation
    delta = 0
    while delta < threshold :
        delta = 0
        for s in states:
            v = V(s)
            # Bellman update
            V(s) = sum(pi(a, s) * q(a,s, V, environment, gamma) for s_p in states)
            # stop when change for any state is small
            delta = max(delta, abs(v-V(s)))
    return v
```

##### Policy Improvement

Now that we have an (estimate of) $v_\pi$ which says how good it is to be in $s$ when following $\pi$, we want to know whether changing the policy would yield a higher return. 

We have previously defined a policy $\pi'$ to be better than $\pi$ if $v_{\pi'}(s) > v_{\pi}(s), \forall s$. One simple way of improving the current policy $\pi$ would thus be to use a $\pi'$ which is identical to $\pi$ at each state besides one $s_{update}$ for which it will take a better action. Let's assume that we have a deterministic policy $\pi$ that we follow at every step besides one step when in $s_{update}$ at which we follow the new $\pi'$ (and continue with $\pi$). By construction : 

$$q_\pi(s, {\pi'}(s)) = v_\pi(s), \forall s \in \mathcal{S} \setminus \{s_{update}\}$$

$$q_\pi(s_{update}, {\pi'}(s_{update})) > v_\pi(s_{update})$$

Then it can be proved that (**Policy Improvement
Theorem**) :

$$v_{\pi'}(s) \geq v_\pi(s), \forall s \in \mathcal{S} \setminus \{s_{update}\}$$

$$v_{\pi'}(s_{update}) > v_\pi(s_{update})$$

*I.e.* if such policy $\pi'$ can be constructed, then it is a better than $\pi$. 

The same hold if we extend the update to all actions and all states and stochastic policies. The general **policy improvement** algorithm is thus :

$$
\begin{aligned}
\pi'(s) &= arg\max_a q_\pi(s,a) \\
&= \sum_{s'} \sum_r arg\max_a p(s', r\vert s, a) \left[ r  + \gamma  v_\pi(s') \right]
\end{aligned}
$$

$\pi'$ is always going to be better than $\pi$ except if $\pi=\pi_{\*}$, in which case the update equations would turn into the Bellman optimality equation.

In python pseudo-code:

```python
def policy_improvement(v, environment, pi):
    dynamics, states, actions, rewards = environment
    is_converged = False
    while not is_converged :
        is_converged = True
        for s in states:
            old_action = pi(s)
            # q defined in `policy_evaluation` pseudo-code
            pi(s) = argmax(q(s,a, v, environment, gamma) for a in actions)
            if old_action != pi(s):
                is_converged = False
    return pi, is_converged
```

#### Value Iteration

In policy iteration, the bottleneck is the policy evaluation which requires multiple loops over the state space (convergence only for an infinite numebr of loops). Importantly, the same convergence guarantees as with policy iteration hold when doing a single policy evaluation step. Policy iteration with a single evaluation step, is called **value iteration** and can be written as a simple update step that combines the truncated policy evaluation and the policy improvement steps:

$$
v_{k+1}(s) = \max_{a} \sum_{s'} \sum_r p(s', r\vert s, a) \left[ r  + \gamma  v_{k}(s') \right], \ \forall s \in \mathcal{S}
$$ 

This is guaranteed to converge for arbitrary $v_0$ if $v_*$ exists. <span class='intuitionText'> Value iteration corresponds to the update rule derived from the Bellman optimality equation </span> . The formula, is very similar to policy evaluation, but it maximizes instead of marginalizing over actions. 

Like policy evaluation, value iteration needs an infinite number of steps to converge to $v_*$. In practice we stop whenever the change for all actions is small.

In python pseudo-code:

```python
def value_to_policy(v, environment, gamma):
    """Makes a deterministic policy from a value function."""
    dynamics, states, actions, rewards = environment
    pi = dict()
    for s in states:
        # q defined in `policy_evaluation` pseudo-code
        pi(s) = argmax(q(s,a, v, environment, gamma) for a in actions)
    return pi

def value_iteration(environment, threshold=..., gamma=...):
    dynamics, states, actions, rewards = environment
    # v initialized arbitrarily for all states except V(terminal)=0
    V = initialize()
    delta = 0
    while delta < threshold :
        delta = 0
        for s in states:
            v = V(s)
            # Bellman optimal update
            # q defined in `policy_evaluation` pseudo-code
            V(s) = max(q(s, a, v, environment, gamma) for a in actions)
            # stop when change for any state is small
            delta = max(delta, abs(v-V(s)))
    pi = value_to_policy(v, environment, gamma)
    return pi
```


:wrench: <span class='practice'> Practical </span> : Faster convergence is often achieved by doing a couple policy evaluation sweeps (instead of a single one in the value iteration case) between each policy improvement. The entire class of truncated policy iteration converges. Truncated policy iteration can be schematically seen as using the modified generalized policy iteration diagram:

<div class="mediumWrap" markdown="1">
![Value Iteration](/img/blog/value_iteration.png)
</div>

As seen above, truncated policy iteration uses only approximate value functions. This often increases the number of required policy evaluation and iteration steps, but greatly decreases the number of steps per policy iteration making the overall algorithm usually quicker.

#### Asynchronous Dynamic Programming

A drawback of DP algorithms, is that they require to loop over all states for a single sweep. *Asynchronous* DP algorithms are in-place iterative methods that update the value of each state in any order. Such algorithms are still guaranteed to converge as long as it can't ignore any state after some points (for the episodic case it is also easy to avoid the few orderings that do not converge).

By carefully selecting the states to update, we can often improve the convergence rate. Furthermore, asynchronous DP enables to update the value of the states as the agent visits them. This is very useful in practice, and focuses the computations on the most relevant states. 

:wrench: <span class='practice'> Practical </span> : Asynchronous DP are usually preferred for problems with large state-spaces
