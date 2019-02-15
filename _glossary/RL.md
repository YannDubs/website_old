*In Reinforcement Learning (RL), the sequential decision-making algorithm (an agent) **interacts** with an environment it is **uncertain** about and learns to map situations to actions to maximize a long term reward. During training, the action it choses are evaluated rather than instructed.*

:school_satchel: <span class='example'> Example </span> : Games can very naturally be framed in a RL framework. For example, when playing tennis you are not told how good every movement you make is, but you are given a certain reward if you win the whole game.  

:mag: <span class='note'> Side Notes </span> : Games could also be framed in a supervised problem. The training set would consist in many different states of the environment and the optimal action to take in each of those. Creating such a dataset is not possible for most applications as it requires to enumerate the exponential number of states and to know the associated best action (*e.g.* exact rotation of all your joints when you play tennis). Note that during supervised training, the feedback indicates the correct action independently to the chosen action. The RL framework is a lot more natural and human-like as the agent is trained by playing the game. Importantly, the agent interacts with the environment such that the states that it will visit depend on previous actions. So it is some a chicken-egg problem where it will unlikely reach good states before being trained, but it has to reach good states to get reward and train effectively. This leads to training curves that start with very long plateaus of low reward until it reaches a good state (somewhat by chance) and then learn quickly. In contrast, supervised methods have very steep loss curves at the start.

:information_source: <span class='resources'> Resources </span> : The link and differences between supervised and RL is described in details by [A. Barto and T. Dietterich](http://www-anw.cs.umass.edu/pubs/2004/barto_d_04.pdf){:.mdLink}. 

In RL, future states depend on current actions, thus requiring to model indirect consequences of actions and planning. Furthermore, the agent often has to take actions in real-time while planning for the future.  All of the above makes it very similar to how humans learn, and is thus widely used in psychology and neuroscience. 

:information_source: <span class='resources'> Resources </span>  : All this section on RL is highly influenced by [Sutton and Barto's introductory book](http://incompleteideas.net/book/the-book-2nd.html){:.mdLink}.


### Markov Decision Process

Markov Decision Processes (MDPs) are a mathematical idealized form of the RL problem that suppose that states follow the **Markov Property**. *I.e.* that future states are conditionally independent of past ones given the present state: $S_{t+1} \perp \\{S_{i}\\}_{i=1}^{t-1} \vert S_t$.

Before diving into the details, it is useful to visualize how simple a MDP is (image taken from [Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html){:.mdLink}):

<div class="mediumWrap" markdown="1">
![log loss](/img/blog/MDP.png)
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

* **Policy** $\pi(a\vert s)$: a mapping from states to probabilities over actions. Intuitively, it corresponds to a "<span class='intuitionText'> The behavioral function </span>" . Often called *stimulus-response* in psychology.
* **State-value function** for policy $\pi$ $v_\pi(s)$: expected return for an agent that follows a policy $\pi$ and starts in state $s$.   Intuitively, it corresponds <span class='intuitionText'> to how good it is to be in a certain state (long term)</span>. Similar to the return, the value function can also be defined recursively, by the very important **Bellman equation** : 

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

* **Action-value function** (*Q function*) for policy $\pi$ $q_\pi(s,a)$: expected total reward and agent can get starting from a state $s_0$. the value of a state is not given by the environment and has to be predicted to learn a good policy. Intuitively, it corresponds <span class='intuitionText'> to how good it is to be in a certain state (long term)</span>. A **Bellman equation** can be derived in a similar way to the value function:

$$
\begin{aligned}
q_\pi(s,a) &:=\mathbb{E}[G_t \vert S_{t}=s, A_{t}=a]  \\
&= \sum_{s'} \sum_r p(s', r\vert s, a) \left[ r  + \gamma  \sum_{g_{t+1}}   p(g_{t+1} \vert s')   g_{t+1} \right] \\
&=  \sum_{s'} \sum_r p(s', r\vert s, a) \left[ r  + \gamma  v_\pi(s') \right]\\
&=  \sum_{s'} \sum_r p(s', r\vert s, a) \left[ r  + \gamma  \sum_{a'} \pi(a' \vert s') q_\pi(s',a') \right] \\
&= \mathbb{E}[R_{t+1} + \gamma \sum_{a'} q_\pi(S_{t+1}, a') \vert S_{t}=s, A_{t}=a]  
\end{aligned}
$$ 


* **Model of the environment**: an internal model in the agent to predict the dynamic of the environment (*e.g.* probability of getting a certain reward or getting in a certain state for each action). This is only used by some RL agents for **planning**.


:mag: <span class='note'> Side Notes </span> : 
* the Bellman equations given a policy $\pi$ are a system of linear equations. For the value function, solving this exactly would take $O(\vert \mathcal{S} \vert ^2)$, while it would take $O(\vert \mathcal{S} \vert \cdot \vert \mathcal{A} \vert)$ for the Q function.
* We usually assume that we have a **finite** MDP. *I.e.* that $\mathcal{R},\mathcal{A},\mathcal{S}$ are finite. Dealing with continuous state and actions pairs requires approximations. One possible way of converting a continuous problem to a finite one, is quantize the state and actions space.

#### Optimality Equations

Solving the RL tasks, consists in finding a good policy. A policy $\pi$ is defined to be better than $\pi'$ iff $v_\pi(s) \geq v_{\pi'}(s), \ \forall s \in \mathcal{S}$. The optimal policy $\pi_*$ has an associated state *optimal value-function* and *optimal action-value function*:

$$v_*(s)=v_{\pi_*}(s):= \max_\pi v_\pi(s)$$

$$
\begin{aligned}
q_*(s,a) &= q_{\pi_*}(s,a) \\
&= \max_\pi q_\pi(s, a) \\
&= \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \vert S_{t}=s, A_{t}=a] 
\end{aligned}
$$ 

A special recursive update (the **Bellman optimality equations**) can be written for the optimal functions $v_*$, $q_*$ by taking the best action at each step instead of marginalizing:

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
q_*(s) &= \mathbb{E}[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \vert S_{t}=s, A_{t}=a]  \\
&= \sum_{s'} \sum_{r} p(s', r \vert s, a) \left[r + \gamma \max_{a'} q_*(s', a')\right]  
\end{aligned}
$$

As for the non-optimal Bellman equations, the optimal ones are a set of equations ($\vert \mathcal \vert$ equations and unknowns) with a unique solution. But these equations are now non-linear. If the dynamics of the environment were known, you could thus solve the non-linear system of equation to get $v_*$, and follow $\pi_*$ by greedily choosing the action that maximizes the expected $v_*$. By solving for $q_*$, you would simply chose the action $a$ that maximizes $q_*(s,a)$, which doesn't require to know anything about the system dynamics.

In practice, the optimal Bellman equations can rarely be solved due to three major problems:

* They require the **dynamics of the environment** which is rarely known.
* They require **large computational resources** (memory and computation) to solve as the number of states $\vert \mathcal{S} \vert$ might be huge (or infinite). This is nearly always an issue.
* The Markov Property.

Practical RL algorithms thus settle for approximating the optimal Bellman equations. Usually they parametrize functions and focus mostly on states that will be frequently encountered to make the computations possible.

