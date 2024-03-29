---
layout: post
title: "Notes on the 2nd edition of Reinforcement Learning: An Introduction"
date: 2024-03-28
categories: RL ML DL
usemathjax: true
---
$$
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\argmax}{arg\,max}
$$

Here are some notes I took when reading the second edition of the <a href="http://acfpeacekeeper.github.io/github-pages/docs/literature/books/RLbook2020.pdf" onerror="this.href='http://localhost:4000/docs/literature/books/RLbook2020.pdf'">Reinforcement Learning: An Introduction</a> book.\\
If you want to get into Reinforcement Learning, or are just interested in Artificial Intelligence in general, I highly recommend that you read this book! It does require some mathematical background to read and understand everything (mostly Linear Algebra, Probabilities, Statistics, and some Calculus), but it is overall one of the best and most exhaustive introductory books about Reinforcement Learning out there.
# Chapter Index
1. [Introduction](#chapter-1-introduction)
2. [Multi-armed Bandits](#chapter-2-multi-armed-bandits)
	1. [A k-armed Bandit Problem](#ch-21-a--armed-bandit-problem)
	2. [Action-value Methods](#ch-22-action-value-methods)
	3. [The 10-armed Test-bed](#ch-23-the-10-armed-test-bed)
	4. [Incremental Implementation](#ch-24-incremental-implementation)
	5. [Tracking a Non-stationary Problem](#ch-25-tracking-a-non-stationary-problem)
	6. [Optimistic Initial Values](#ch-26-optimistic-initial-values)
	7. [Upper-Confidence-Bound Action Selection](#ch-27-upper-confidence-bound-action-selection)
	8. [Gradient Bandit Algorithms](#ch-28-gradient-bandit-algorithms)
	9. [Associative Search (Contextual Bandits)](#ch-29-associative-search-contextual-bandits)
	10. [Summary](#ch-210-summary)
3. [Finite Markov Decision Processes](#chapter-3-finite-markov-decision-processes)
	1. [The Agent-Environment Interface](#ch-31-the-agent-environment-interface)
	2. [Goals and Rewards](#ch-32-goals-and-rewards)
	3. [Returns and Episodes](#ch-33-returns-and-episodes)
	4. [Unified Notation for Episodic and Continuing Tasks](#ch-34-unified-notation-for-episodic-and-continuing-tasks)
	5. [Policies and Value Functions](#ch-35-policies-and-value-functions)
	6. [Optimal Policies and Optimal Value Functions](#ch-36-optimal-policies-and-optimal-value-functions)
	7. [Optimality and Approximation](#ch-37-optimality-and-approximation)
	8. [Summary](#ch-38-summary)
4. [Dynamic Programming](#chapter-4-dynamic-programming)
	1. [Policy Evaluation (Prediction)](#ch-41-policy-evaluation-prediction)
	2. [Policy Improvement](#ch-42-policy-improvement)
	3. [Policy Iteration](#ch-43-policy-iteration)
	4. [Value Iteration](#ch-44-value-iteration)
	5. [Asynchronous Dynamic Programming](#ch-45-asynchronous-dynamic-programming)
	6. [Generalized Policy Iteration](#ch-46-generalized-policy-iteration)
	7. [Efficiency of Dynamic Programming](#ch-47-efficiency-of-dynamic-programming)
	8. [Summary](#ch-48-summary)
5. [Monte Carlo Methods](#chapter-5-monte-carlo-methods)
	1. [Monte Carlo Prediction](#ch-51-monte-carlo-prediction)
	2. [Monte Carlo Estimation of Action Values](#ch-52-monte-carlo-estimation-of-action-values)
	3. [Monte Carlo Control](#ch-53-monte-carlo-control)
	4. [Monte Carlo Control without Exploring Starts](#ch-54-monte-carlo-control-without-exploring-starts)
	5. [Off-policy Prediction via Importance Sampling](#ch-55-off-policy-prediction-via-importance-sampling)
	6. [Incremental Implementation](#ch-56-incremental-implementation)
	7. [Off-policy Monte Carlo Control](#ch-57-off-policy-monte-carlo-control)
	8. [*Discounting-aware Importance Sampling](#ch-58-discounting-aware-importance-sampling)
	9. [*Per-decision Importance Sampling](#ch-59-per-decision-importance-sampling)
	10. [Summary](#ch-510-summary)
6. [Temporal-Difference Learning](#chapter-6-temporal-difference-learning)
	1. [TD Prediction](#ch-61-td-prediction)
	2. [Advantages of TD Prediction Methods](#ch-62-advantages-of-td-prediction-methods)
	3. [Optimality of TD(0)](#ch-63-optimality-of-td0)
	4. [Sarsa: On-policy TD Control](#ch-64-sarsa-on-policy-td-control)
	5. [Q-learning: Off-policy TD Control](#ch-65-q-learning-off-policy-td-control)
	6. [Expected Sarsa](#ch-66-expected-sarsa)
	7. [Maximization Bias and Double Learning](#ch-67-maximization-bias-and-double-learning)
	8. [Games, Afterstates, and Other Special Cases](#ch-68-games-afterstates-and-other-special-cases)
	9. [Summary](#ch-69-summary)
7. [n-step Bootstrapping](#chapter-7--step-bootstrapping)
	1. [n-step TD Prediction](#ch-71--step-td-prediction)
	2. [n-step Sarsa](#ch-72--step-sarsa)
	3. [n-step Off-policy Learning](#ch-73--step-off-policy-learning)
	4. [*Per-decision Methods with Control Variates](#ch-74-per-decision-methods-with-control-variates)
	5. [Off-policy Learning Without Importance Sampling: The $n$-step Tree Backup Algorithm](#ch-75-off-policy-learning-without-importance-sampling-the--step-tree-backup-algorithm)
	6. [*A Unifying Algorithm: n-step Q(sigma)](#ch-76-a-unifying-algorithm--step)
	7. [Summary](#ch-77-summary)
8. [Planning and Learning with Tabular Methods](#chapter-8-planning-and-learning-with-tabula-methods)
	1. [Models and Planning](#ch-81-models-and-planning)
	2. [Dyna: Integrated Planning, Acting, and Learning](#ch-82-dyna-integrated-planning-acting-and-learning)
	3. [When the Model Is Wrong](#ch-83-when-the-model-is-wrong)
	4. [Prioritized Sweeping](#ch-84-prioritized-sweeping)
	5. [Expected vs. Sample Updates](#ch-85-expected-vs-sample-updates)
	6. [Trajectory Sampling](#ch-86-trajectory-sampling)
	7. [Real-Time Dynamic Programming](#ch-87-real-time-dynamic-programming)
	8. [Planning at Decision Time](#ch-88-planning-at-decision-time)
	9. [Heuristic Search](#ch-89-heuristic-search)
	10. [Rollout Algorithms](#ch-810-rollout-algorithms)
	11. [Monte Carlo Tree Search](#ch-811-monte-carlo-tree-search)
	12. [Summary of the Chapter](#ch-812-summary-of-the-chapter)
	13. [Summary of Part I: Dimensions](#ch-813-summary-of-part-i-dimensions)
9. [On-policy Prediction with Approximation](#chapter-9-on-policy-prediction-with-approximation)
	1. [Value-function Approximation](#ch-91-value-function-approximation)
	2. [The Prediction Objective (VE)](#ch-92-the-prediction-objective)
	3. [Stochastic-gradient and Semi-gradient Methods](#ch-93-stochastic-gradient-and-semi-gradient-methods)
	4. [Linear Methods](#ch-94-linear-methods)
	5. [Feature Construction for Linear Methods](#ch-95-feature-construction-for-linear-methods)
		1. [Polynomials](#ch-951-polynomials)
		2. [Fourier Basis](#ch-952-fourier-basis)
		3. [Coarse Coding](#ch-953-coarse-coding)
		4. [Tile Coding](#ch-954-tile-coding)
		5. [Radial Basis Functions](#ch-955-radial-basis-functions)
	6. [Selecting Step-Size Parameters Manually](#ch-96-selecting-step-size-parameters-manually)
	7. [Non-linear Function Approximation: Artificial Neural Networks](#ch-97-non-linear-function-approximation-artificial-neural-networks)
	8. [Least-Squares TD](#ch-98-least-squares-td)
	9. [Memory-based Function Approximation](#ch-99-memory-based-function-approximation)
	10. [Kernel-based Function Approximation](#ch-910-kernel-based-function-approximation)
	11. [Looking Deeper at On-policy Learning: Interests and Emphasis](#ch-911-looking-deeper-at-on-policy-learning-interests-and-emphasis)
	12. [Summary](#ch-912-summary)
10. [On-policy Control with Approximation](#chapter-10-on-policy-control-with-approximation)
	1. [Episodic Semi-gradient Control](#ch-101-episodic-semi-gradient-control)
	2. [Semi-gradient $n$-step Sarsa](#ch-102-semi-gradient--step-sarsa)
	3. [Average Reward: A New Problem Setting for Continuing Tasks](#ch-103-average-reward-a-new-problem-setting-for-continuing-tasks)
	4. [Deprecating the Discounted Setting](#ch-104-deprecating-the-discounted-setting)
	5. [Differential Semi-gradient n-step Sarsa](#ch-105-differential-semi-gradient--step-sarsa)
	6. [Summary](#ch-106-summary)
11. [*Off-policy Methods with Approximation](#chapter-11-off-policy-methods-with-approximation)
	1. [Semi-gradient Methods](#ch-111-semi-gradient-methods)
	2. [Examples of Off-policy Divergence](#ch-112-examples-of-off-policy-divergence)
	3. [The Deadly Triad](#ch-113-the-deadly-triad)
	4. [Linear Value-function Geometry](#ch-114-linear-value-function-geometry)
	5. [Gradient Descent in the Bellman Error](#ch-115-gradient-descent-in-the-bellman-error)
	6. [The Bellman Error is Not Learnable](#ch-116-the-bellman-error-is-not-learnable)
	7. [Gradient-TD Methods](#ch-117-gradient-td-methods)
	8. [Emphatic-TD Methods](#ch-118-emphatic-td-methods)
	9. [Reducing Variance](#ch-119-reducing-variance)
	10. [Summary](#ch-1110-summary)
12. [Eligibility Traces](#chapter-12-eligibility-traces)
	1. [The lambda-return](#ch-121-the--return)
	2. [TD(lambda)](#ch-122-td)
	3. [n-step Truncated lambda-return Methods](#ch-123--step-truncated--return-methods)
	4. [Redoing Updates: Online lambda-return Algorithm](#ch-124-redoing-updates-online--return-algorithm)
	5. [True Online TD(lambda)](#ch-125-true-online-td)
	6. [*Dutch Traces in Monte Carlo Learning](#ch-126-dutch-traces-in-monte-carlo-learning)
	7. [Sarsa(lambda)](#ch-127-sarsa)
	8. [Variable lambda and gamma](#ch-128-variable-and)
	9. [Off-policy Traces with Control Variates](#ch-129-off-policy-traces-with-control-variates)
	10. [Watkin's Q(lambda) to Tree-Backup(lambda)](#ch-1210-watkins-q-to-tree-backup)
	11. [Stable Off-policy Methods with Traces](#ch-1211-stable-off-policy-methods-with-traces)
	12. [Implementation Issues](#ch-1212-implementation-issues)
	13. [Conclusions](#ch-1213-conclusions)
13. [Policy Gradient Methods](#chapter-13-policy-gradient-methods)
	1. [Policy Approximation and its Advantages](#ch-131-policy-approximation-and-its-advantages)
	2. [The Policy Gradient Theorem](#ch-132-the-policy-gradient-theorem)
	3. [REINFORCE: Monte Carlo Policy Gradient](#ch-133-reinforce-monte-carlo-policy-gradient)
	4. [REINFORCE with Baseline](#ch-134-reinforce-with-baseline)
	5. [Actor-Critic Methods](#ch-135-actor-critic-methods)
	6. [Policy Gradient for Continuing Problems](#ch-136-policy-gradient-for-continuing-problems)
	7. [Policy Parameterization for Continuous Actions](#ch-137-policy-parameterization-for-continuous-actions)
	8. [Summary](#ch-138-summary)
14. [Psychology](#chapter-14-psychology)
	1. [Prediction and Control](#ch-141-prediction-and-control)
	2. [Classical Conditioning](#ch-142-classical-conditioning)
		1. [Blocking and Higher-order Conditioning](#ch-1421-blocking-and-higher-order-conditioning)
		2. [The Rescorla-Wagner Model](#ch-1422-the-rescorla-wagner-model)
		3. [The TD Model](#ch-1423-the-td-model)
		4. [TD Model Simulations](#ch-1424-td-model-simulations)
	3. [Instrumental Conditioning](#ch-143-instrumental-conditioning)
	4. [Delayed Reinforcement](#ch-144-delayed-reinforcement)
	5. [Cognitive Maps](#ch-145-cognitive-maps)
	6. [Habitual and Goal-directed Behavior](#ch-146-habitual-and-goal-directed-behavior)
	7. [Summary](#ch-147-summary)
15. [Neuroscience](#chapter-15-neuroscience)
	1. [Neuroscience Basics](#ch-151-neuroscience-basics)
	2. [Reward Signals, Reinforcement Signals, Values, and Prediction Errors](#ch-152-reward-signals-reinforcement-signals-values-and-prediction-errors)
	3. [The Reward Prediction Error Hypothesis](#ch-153-the-reward-prediction-error-hypothesis)
	4. [Dopamine](#ch-154-dopamine)
	5. [Experimental Support for the Reward Prediction Error Hypothesis](#ch-155-experimental-support-for-the-reward-prediction-error-hypothesis)
	6. [TD Error/Dopamine Correspondence](#ch-156-td-errordopamine-correspondence)
	7. [Neural Actor–Critic](#ch-157-neural-actor–critic)
	8. [Actor and Critic Learning Rules](#ch-158-actor-and-critic-learning-rules)
	9. [Hedonistic Neurons](#ch-159-hedonistic-neurons)
	10. [Collective Reinforcement Learning](#ch-1510-collective-reinforcement-learning)
	11. [Model-based Methods in the Brain](#ch-1511-model-based-methods-in-the-brain)
	12. [Addiction](#ch-1512-addiction)
	13. [Summary](#ch-1513-summary)
16. [Applications and Case Studies](#chapter-16-applications-and-case-studies)
	1. [TD-Gammon](#ch-161-td-gammon)
	2. [Samuel's Checkers Player](#ch-162-samuels-checkers-player)
	3. [Watson's Daily-Double Wagering](#ch-163-watsons-daily-double-wagering)
	4. [Optimizing Memory Control](#ch-164-optimizing-memory-control)
	5. [Human-level Video Game Play](#ch-165-human-level-video-game-play)
	6. [Mastering the Game of Go](#ch-166-mastering-the-game-of-go)
		1. [AlphaGo](#ch-1661-alphago)
		2. [AlphaGo Zero](#ch-1662-alphago-zero)
	7. [Personalized Web Services](#ch-167-personalized-web-services)
	8. [Thermal Soaring](#ch-168-thermal-soaring)
17. [Frontiers](#chapter-17-frontiers)
	1. [General Value Functions and Auxiliary Tasks](#ch-171-general-value-functions-and-auxiliary-tasks)
	2. [Temporal Abstraction via Options](#ch-172-temporal-abstraction-via-options)
	3. [Observations and State](#ch-173-observations-and-state)
	4. [Designing Reward Signals](#ch-174-designing-reward-signals)
	5. [Remaining Issues](#ch-175-remaining-issues)
	6. [Reinforcement Learning and the Future of Artificial Intelligence](#ch-176-reinforcement-learning-and-the-future-of-artificial-intelligence)

# Notes on the Book
## Chapter 1: Introduction

Def. **Reinforcement Learning (RL)**: an agent learns how to map situations to actions through *trial-and-error* or *planned* interaction with a (possibly) uncertain environment, so as to maximize a numerical reward value (i.e., achieve his goal or goals).
- *Delayed reward* is another important characteristic of RL, since any action taken may influence (not only the immediate reward value, but also) any subsequent rewards;
- RL can be formalized as the optimal control of incompletely-known Markov Decision Processes (MDPs).

Besides RL, other **Machine Learning (ML)** paradigms include *Supervised Learning* - predicting the correct label, given the corresponding set of features - and *Unsupervised Learning* - finding hidden patterns in a collection of unlabeled features.

A challenge unique to the RL paradigm is that of the trade-off between **exploration versus exploitation**. This challenge arises due to the fact that an agent prefers to take the actions that have previously given the highest rewards (*exploitation*), but it must also try out other actions in order to have more knowledge about which actions it should select (*exploration*).

A RL system has four main sub-elements beyond the interactive agent and the environment, which are:
- A policy $$\pi_t: s \rightarrow a$$, which in stochastic cases specifies a probability for each action;
- A reward $$r(s, a)$$, an immediate signal that specifies how good it is for an agent to have chosen a certain action in a given state (may also be stochastic);
- A value function $$v(s)$$ that specifies the total reward an agent is expected to accumulate in the future if he starts at a given state, i.e., predicted long-term reward;
- A (optional) model of the environment used by model-based methods (opposed to purely trial-and-error model-free methods) for planning.

# Part I: Tabular Solution Methods

## Chapter 2: Multi-armed Bandits

*Non-associative* setting: a problem setting that involves learning to act in only 1 situation

*Associative* setting: a problem setting where the best action depends on the situation

### Ch 2.1: A $$k$$-armed Bandit Problem

Setting of the $$k$$-armed bandit learning problem (analogous to a slot machine with $$k$$ levers):
1. Choose 1 action from among $$k$$ different options;
2. Receive a (numerical) reward from a stationary probability distribution which depends on the action selected;
3. Repeat steps 1 and 2 with the purpose of maximizing the expected total reward over some time period (e.g., 1000 action selections or *time steps*).

**Value** of an action: the expected or mean reward received if that action is selected

Letting $$A_t$$ be the action taken at time step $$t$$ and $$R_t$$ the corresponding reward, then the value $$q_{*}(a)$$ of an arbitrary action $$a$$ is given by:
$$
    q_{*} (a) \doteq \mathbb{E} [R_t | A_t = a].
$$

Since we do not know the true value of each action, we need to estimate them in such a way that the estimates are close to the real values. The estimated value of an action $$a$$ at time step $$t$$ is denoted by $$Q_t (a)$$.

**Greedy** action: the action with the highest estimated value at a given time step
- Choosing this action equates to the agent **exploiting** his current knowledge of the values of the actions;
- Selecting 1 of the non-greedy actions enables the agent to improve his estimates of the non-greedy action's value, i.e., **exploration**;
- Exploitation maximizes the reward on 1 step, but it needs to be intercalated with exploration steps so as to maximize the greater total reward in the long term.

### Ch 2.2: Action-value Methods

Def. **Action-value Methods**: methods used to estimate the values of actions and to use those estimates to select an action to take at a given time step.

Letting $$\mathbb{1}_{predicate}$$ be the random variable which equals 1 if the $predicate$ is true and 0 otherwise, the value of an action can be estimated by averaging the rewards received: <!-- TODO: check if equations inside text inside equations need double $ signs-->
$$
Q_t (a) \doteq \frac{\text{sum of rewards when $a$ taken prior to $t$}}{\text{number of times $a$ taken prior to $t$}} = \frac{\sum_{i = 1}^{t - 1} R_i \cdot \mathbb{1}_{A_i = a}}{\sum_{i = 1}^{t - 1} \mathbb{1}_{A_i = a}}.
$$

If the denominator is zero (action has never been taken), then $$Q_t(a)$$ is defined as an arbitrary default value (e.g., zero). By the law of large numbers, as the denominator goes to infinity, $$Q_t(a)$$ converges to $$q_{*}(a)$$. This is called the *sample-average* method for estimating action values.

The simplest action selection rule is to always select a greedy action and - if there is more than 1 action with the same highest value - to break ties in some arbitrary way (e.g., randomly). This action selection method can be written as:
$$
A_t = \argmax_a Q_t (a).
$$
This selection method never performs exploration. A simple alternative that does so is to select the greedy action most of the time (probability $$1 - \epsilon$$) and (with probability $$\epsilon$$) to randomly select any possible action with equal probability. Methods that use this near-greedy action selection rule are dubbed $$\epsilon$$-greedy methods.

### Ch 2.3: The 10-armed Test-bed

**Non-stationary** setting: problem setting where the true values of the actions (or the reward probabilities) change over time

Given a set of 2000 randomly generated $$k$$-armed bandit problems (with $$k = 10$$), for each problem in the set, the action values $$q_{*}(a), \ a = \{1, 2, \dots, 10\},$$ were selected from a normal (Gaussian) distribution with $$\mu = 0, \  \sigma^2 = 1$$. When a learning method is applied to this problem selects action $$A_t$$ at time step $$t$$, the actual reward ($$R_t$$) was drawn from a normal distribution with $$\mu = q_{*}(A_t), \ \sigma^2 = 1$$. 
The performance of the learning methods is measured as it improves with experience over 1000 time steps of the bandit problem, which makes up a single run. To obtain an accurate measure of the learning algorithms' behavior, 2000 runs are performed and the results for the bandit problems are averaged.

A greedy action selection method is compared against 2 $$\epsilon$$-greedy methods (with $$\epsilon = 0.01 \lor \epsilon = 0.1$$). All methods begin with initial action-value estimates of zero and update these estimates using the sample-average technique.
While the greedy method improved slightly faster than the other 2, it converged to a reward-per-step of 1, which is lower than the best value of around 1.54 achieved by the $$\epsilon$$-greedy method (with $$\epsilon = 0.1$$). The method with $$\epsilon = 0.1$$ improved faster than the method with $$\epsilon = 0.01$$, since it explored more earlier. However, the method with $$\epsilon = 0.01$$ converges to a higher reward-per-step in the long run, since the method with $$\epsilon = 0.1$$ never selects the optimal action more than 91% of the time. 
It is possible to perform $$\epsilon$$ annealing to try to get fast learning at the start combined with convergence to a higher reward average.

It takes more exploration to find the optimal actions in cases with noisy rewards (i.e., high reward variance), meaning that $$\epsilon$$-greedy methods perform even better in those cases, when compared to the greedy method. Also, although the greedy method is theoretically optimal in the deterministic case (i.e., with $$\sigma^2 = 0$$), this property does not hold in non-stationary bandit problems, making exploration a necessity even in deterministic settings.

### Ch 2.3: The 10-armed Test-bed

**Non-stationary** setting: problem setting where the true values of the actions (or the reward probabilities) change over time

Given a set of 2000 randomly generated $$k$$-armed bandit problems (with $$k = 10$$), for each problem in the set, the action values $$q_{*}(a), \ a = \{1, 2, \dots, 10\},$$ were selected from a normal (Gaussian) distribution with $$\mu = 0, \  \sigma^2 = 1$$. When a learning method is applied to this problem selects action $$A_t$$ at time step $$t$$, the actual reward ($$R_t$$) was drawn from a normal distribution with $$\mu = q_{*}(A_t), \ \sigma^2 = 1$$. 
The performance of the learning methods is measured as it improves with experience over 1000 time steps of the bandit problem, which makes up a single run. To obtain an accurate measure of the learning algorithms' behavior, 2000 runs are performed and the results for the bandit problems are averaged.

A greedy action selection method is compared against 2 $$\epsilon$$-greedy methods (with $$\epsilon = 0.01 \lor \epsilon = 0.1$$). All methods begin with initial action-value estimates of zero and update these estimates using the sample-average technique.
While the greedy method improved slightly faster than the other 2, it converged to a reward-per-step of 1, which is lower than the best value of around 1.54 achieved by the $$\epsilon$$-greedy method (with $$\epsilon = 0.1$$). The method with $$\epsilon = 0.1$$ improved faster than the method with $$\epsilon = 0.01$$, since it explored more earlier. However, the method with $$\epsilon = 0.01$$ converges to a higher reward-per-step in the long run, since the method with $$\epsilon = 0.1$$ never selects the optimal action more than 91% of the time. 
It is possible to perform $$\epsilon$$ annealing to try to get fast learning at the start combined with convergence to a higher reward average.

It takes more exploration to find the optimal actions in cases with noisy rewards (i.e., high reward variance), meaning that $$\epsilon$$-greedy methods perform even better in those cases, when compared to the greedy method. Also, although the greedy method is theoretically optimal in the deterministic case (i.e., with $$\sigma^2 = 0$$), this property does not hold in non-stationary bandit problems, making exploration a necessity even in deterministic settings.

### Ch 2.4: Incremental Implementation

For a single action, let $$R_i$$ denote the reward received after the $$i^{th}$$ selection of *this action* and $$Q_n$$ the estimate of its action value after it has been selected $$n - 1$$ times, written as:
$$
Q_n \doteq \frac{R_1 + R_2 + \dots + R_{n - 1}}{n - 1}.
$$
Instead of maintaining a record of all the rewards and performing the computation for the estimated value whenever needed (resulting in the growth of both computational and memory requirements), we can devise incremental formulas to update the averages with a small and constant computation to process each new reward. Given $$Q_n$$ and the $$n^{th}$$ reward $$R_n$$, the new average of all $$n$$ rewards can be computed as:
$$
\begin{align}
	Q_{n + 1} &= \frac{1}{n} \sum_{i = 1}^n R_i \\
	&= \frac{1}{n}(R_n + \sum_{i = 1}^{n - 1} R_i) \\
	&= \frac{1}{n}(R_n + (n - 1) \cdot \frac{1}{n - 1} \cdot \sum_{i = 1}^{n - 1} R_i) \\
	&= \frac{1}{n} (R_n + (n - 1) \cdot Q_n) \\
	&= \frac{1}{n} (R_n + n \cdot Q_n - Q_n) \\
	&= Q_n + \frac{1}{n} [R_n - Q_n], \ n > 1 \\
	Q_2 &= R_1, \ Q_1 \in \mathbb{R}.
\end{align}
$$
This implementation only needs memory for $$Q_n$$ and $$n$$, and only performs a small computation for each new reward. 
The general form of the previous update rule is given by:
$$
NewEstimate \leftarrow OldEstimate + StepSize [Target - OldEstimate],
$$
where $$[Target - OldEstimate]$$ is an *error* in the estimate, which is reduced by taking a step towards the (possibly noisy) target value.
The step-size parameter is generally denoted by $$\alpha$$ or $$\alpha_t (a)$$.

### Ch 2.5: Tracking a Non-stationary Problem

When reward probabilities change over time, it makes sense to give more weight to recent rewards than to those receive long ago. This can be done by using a constant step-size parameter, e.g., for updating an average $$Q_n$$ of the $$n - 1$$ past rewards w.h.t.:
$$
Q_{n + 1} \doteq Q_n + \alpha [R_n - Q_n],
$$
where the step-size parameter $$\alpha \in \  ]0, 1]$$ is constant. Given this, $$Q_{n + 1}$$ becomes a weighted average (since the sum of weights = 1) of the past rewards and initial estimate $$Q_1$$:
$$
\begin{align}
	Q_{n + 1} &= Q_n + \alpha [R_n - Q_n] \\
	&= \alpha \cdot R_n + (1- \alpha) Q_n \\
	&= \alpha \cdot R_n + (1 - \alpha) [\alpha \cdot R_{n - 1} + (1 - \alpha)Q_{n - 1}] \\
	&= \alpha \cdot R_n + (1 - \alpha) \cdot \alpha \cdot R_{n - 1} + (1 - \alpha)^2 Q_{n - 1} \\
	&= \alpha \cdot R_n + (1 - \alpha) \cdot \alpha \cdot R_{n - 1} + \dots + (1 - \alpha)^{n - 1} \cdot \alpha \cdot R_1 + (1 - \alpha)^n \cdot Q_1 \\
	&= (1 - \alpha)^n \cdot Q_1 + \sum_{i = 1}^n \alpha \cdot (1 - \alpha)^{n - i} \cdot R_i.
\end{align}
$$
Since $$1 - \alpha < 1$$, the weight given to $$R_i$$ decreases as the number of intervening rewards increases. Also, the weight decays exponentially in proportion to the exponent on $$1 - \alpha$$ and, if $$1 - \alpha = 0$$, the entire weight goes onto the very last reward $$R_n$$. This method is sometimes called an *exponential recency-weighted average*. 

Letting $$\alpha_n (a)$$ denote the step-size parameter to process the reward obtained after the $$n^{th}$$ selection of action $$a$$, for the sample-average method, w.h.t. $$\alpha_n (a) = 1/n$$, whose convergence to the true action values is guaranteed by the law of large numbers. However, convergence is **NOT** guaranteed for all choices of the $$\{\alpha_n (a)\}$$ sequence. Through a result in stochastic approximation theory, we obtain the conditions required to assure convergence with probability 1:
$$
\sum_{n = 1}^{\infty} \alpha_n (a) = \infty \quad \land \quad \sum_{n = 1}^{\infty} \alpha_n^2 (a) < \infty,
$$
where the first condition is required to guarantee that the steps are big enough to overcome any initial conditions or random fluctuations that would otherwise result in getting stuck at saddle points, and the second condition guarantees that the steps will eventually become small enough to assure convergence.
The second condition is not met for the constant step-size parameter case, i.e., $$\alpha_n (a) = \alpha$$. This means that the estimates will never completely converge, which is actually a desirable property for non-stationary problems (the most common type of problem in RL), since the estimates continue to vary in response to the most recently received rewards, accounting for the changes in reward probabilities over time. Also, the sequences of step-size parameters that meet both of the above conditions often lead to slow convergence rates, meaning that these are seldomly used in applications and empirical research.

### Ch 2.6: Optimistic Initial Values

All previous methods are somewhat dependent on the initial action-value estimates $$Q_1 (a)$$, i.e., they are *biased* by their initial estimates. This bias decreases over time as various actions are selected. However, while for sample-average methods the bias eventually disappear after all actions have been taken at least once, the bias is permanent for methods with a constant $$\alpha$$. This property means that, when using methods with a constant $$\alpha$$, the user must select the values for the initial estimates, which provides a way to supply some prior knowledge about the expected rewards, at the possible cost of being harder to tune.

By selecting optimistic initial action-values, i.e., $$Q_1 (a) >> R_1 (a), \forall a$$, the agent will always be disappointed since the rewards will always be far less than the first estimates, regardless of which actions are selected. This encourages exploration, as the agent will select all possible actions before the value estimates converge, even if greedy actions are selected at every single time step.

This technique for encouraging exploration is named *optimistic initial values* and is a simple, yet effective trick when used on stationary problems (e.g., with $$Q_1(a) = 5$$ it outperforms a $$\epsilon$$-greedy method with $$Q_1(a) = 0$$ and $$\epsilon = 0.1$$). However, since the drive for exploration is dependent on the initials conditions and disappears after a certain time, it cannot adequately deal with non-stationary problems, where exploration is always required due to the dynamic nature of the reward probabilities. This drawback is present in all methods that treat the beginning of time as a special event (e.g., the sample-average methods).

### Ch 2.7: Upper-Confidence-Bound Action Selection

While $$\epsilon$$-greedy methods encourage exploration, they do so equally, without any preference for whether the action selected is nearly greedy or particularly uncertain. However, it is possible to select the non-greedy actions while taking into account both how close their value estimates are to the maximal action-value and the estimation uncertainty. An effective way of doing this is to select actions according to the following equation:
$$
A_t \doteq \argmax_a [Q_t(a) + c \cdot \sqrt{\frac{\ln t}{N_t(a)}}],
$$
where $$N_t(a)$$ denotes the number of times the action $$a$$ has been selected prior to time $$t$$ and the number $$c > 0$$ controls the degree of exploration. If $$N_t(a) = 0$$, then $$a$$ is considered to be a maximizing action.

This **upper confidence bound (UCB)** action selection is based on the idea that the square-root term is a measure of the uncertainty or variance of action $$a$$ value's estimate. As such, the max'ed over quantity becomes a sort of upper bound on the possible true value of action $$a$$ with $$c$$ determining the confidence level, and thus the uncertainty is reduced each time the action $$a$$ is selected. The natural logarithm results in smaller increases over time, meaning that actions with lower value estimates or that have been frequently selected, will be selected with decreasing frequency.

UCB often performs better than $$\epsilon$$-greedy action selection (except in the first $$k$$ steps), but it is harder to extend beyond bandits into the general RL settings. This is due to its difficulties in dealing with more advanced settings, such as non-stationary problems and (function approximation) with large state spaces.

### Ch 2.8: Gradient Bandit Algorithms

Beyond using action-value estimates to select actions, it is also possible to learn a numerical *preference* for each action $$a$$, denoted $$H_t(a) \in \mathbb{R}$$, which has no interpretation w.r.t. reward. As such, only the relative preference of 1 action over another is important. The action probabilities are determined according to a softmax distribution as follows:
$$
Pr\{A_t = a\} \doteq \frac{\exp(H_t(a))}{\sum_{b = 1}^k \exp(H_t(b))} \doteq \pi_t (a),
$$
where $$\pi_t(a)$$ is the probability of taking action $$a$$ at time $$t$$. All actions have an equal probability of being selected at first (i.e., $$H_1(a) = 0, \forall a$$).

There exists a natural learning algorithm for softmax action preferences based on the idea of **Stochastic Gradient Ascent (SGA)**, where, at each time step, after selecting action $$A_t$$, and receiving the reward $$R_t$$, the action preferences are updated as follows:
$$
\begin{align}
	H_{t + 1}(A_t) &\doteq H_t(A_t) + \alpha (R_t - \bar{R_t}) (1 - \pi_t(A_t)), &\text{and} \\
	H_{t + 1}(a) &\doteq H_t(a) - \alpha (R_t \bar{R_t}) \pi_t(a), &\forall a \neq A_t,
\end{align}
$$
where $$\alpha > 0$$ is a step-size parameter and $$\bar{R}_t \in \mathbb{R}$$ - which serves as baseline to compare against the reward - is the average of the rewards up to but not including time $$t$$ (with $$\bar{R}_1 \doteq R_1$$). If $$R_t > \bar{R}_t, \  t \neq 1$$, then the probability of taking $$A_t$$ in the future is increased, otherwise, the probability of taking $$A_t$$ is decreased if $$R_t < \bar{R}_t$$. Also, the unselected actions probabilities are updated in the opposite direction.

Since only the relative preferences are taken into account, adding an arbitrary constant value to all the action preferences has no effect on the action probabilities. Also, since the reward baseline term instantaneously adapts to new values of the mean, shifting the mean (e.g., $$\mu_{new} = \mu_{old} + 4$$) of the distribution (while keeping the unit variance) has no effect on the gradient bandit algorithm. However, omitting the baseline term results in a significantly degraded performance.

#### The Bandit Gradient Algorithm as SGA

In exact **Gradient Ascent (GA)**, each action preference $$H_t(a)$$ would be incremented in proportion to the increment's effect on performance, given by:
$$
H_{t + 1} (a) \doteq H_t(a) + \alpha \frac{\partial \mathbb{E}[R_t]}{\partial H_t (a)},
$$
where the measure of performance is the expected reward:
$$
\mathbb{E}[R_t] = \sum_x \pi_t (x) \cdot q_{*} (x),
$$
and the measure of the increment's effect is the *partial derivative* of this performance measure w.r.t. the action preference. Since $$q_{*}(x)$$ is not known, it is impossible to use exact GA. As such, the updates will instead take the form of those used in SGA.

The exact performance gradient can be written as:
$$
\frac{\partial \mathbb{E} [R_t]}{\partial H_t(a)} = \frac{\partial [\sum_x \pi_t(x) \cdot q_{*}(x)]}{\partial H_t(a)} = \sum_x q_{*}(x) \frac{\partial \pi_t(x)}{\partial H_t(a)} = \sum_x (q_{*}(x) - B_t) \frac{\partial \pi_t(x)}{\partial H_t(a)},
$$
where the baseline $$B_t$$ can be any scalar value that doesn't depend on $$x$$. Since the sum of probabilities is always one, the sum of the changes $$\sum_x \frac{\partial \pi_t (x)}{\partial H_t (a)} = 0$$, and the baseline can be added without changing the equality.
We continue by multiplying each term of the sum by $$\pi_t(x)/\pi_t(x)$$, as follows:
$$
\begin{align}
	\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} &= \sum_x \pi_t(x) \cdot (q_{*}(x) - B_t) \cdot \frac{\partial \pi_t (x)}{\partial H_t(x)} / \pi_t(x) \\
	&= \mathbb{E} [(q_{*}(A_t) - B_t) \cdot \frac{\partial \pi_t (A_t)}{\partial H_t(a)}/\pi_t(A_t)] \\
	&= \mathbb{E}[(R_t - \bar{R_t}) \cdot \frac{\partial \pi_t(A_t)}{\partial H_t(a)}/\pi_t(A_t)] \\
	&= \mathbb{E} [(R_t - \bar{R}_t) \cdot \pi_t(A_t) \cdot (\mathbb{1}_{a = A_t} - \pi_t(a))/\pi_t(A_t)] \\
	&= \mathbb{E}[(R_t - \bar{R}_t) \cdot (\mathbb{1}_{a = A_t} - \pi_t(a))],
\end{align}
$$
where the chosen baseline is $$B_t = \bar{R_t}$$ and $$R_t$$ is substituted for $$q_{*}(A_t)$$, which is allowed since $$\mathbb{E}[R_t|A_t] = q_{*}(A_t)$$. 
By substituting a sample of the expectation above for the performance gradient, w.h.t.:
$$
H_{t + 1}(a) = H_t (a) + \alpha \cdot (R_t - \bar{R}_t) \cdot (\mathbb{1}_{a = A_t} - \pi_t(a)), \quad \forall a,
$$
which is equivalent to the original algorithm.
By recalling the standard quotient rule for derivatives
$$
\frac{\partial}{\partial x}[\frac{f(x)}{g(x)}] = \frac{\frac{\partial f(x)}{\partial x}g(x) - f(x)\frac{\partial g(x)}{\partial x}}{g(x)^2},
$$
we can then write
$$
\begin{align}
	\frac{\partial \pi_t(x)}{\partial H_t(a)} &= \frac{\partial}{\partial H_t(a)} \pi_t(x) \\
	&= \frac{\partial}{\partial H_t(a)} [\frac{\exp(H_t(x))}{\sum_{y=1}^k \exp(H_t(y))}] \\
	&= \frac{\frac{\partial \exp(H_t(x))}{\partial H_t(a)} \sum_{y=1}^k \exp(H_t(y)) - \exp(H_t(x)) \cdot \exp (H_t(a))}{(\sum_{y=1}^k \exp(H_t(y)))^2} \\
	&= \frac{\mathbb{1}_{a=x \exp(H_t(x))} \sum_{y=1}^k \exp(H_t(y)) - \exp(H_t(x)) \exp(H_t(a))}{(\sum_{y=1}^k \exp(H_t(y)))^2} \\
	&= \frac{\mathbb{1}_{a = x \exp(H_t(x))}}{\sum_{y=1}^k \exp(H_t(y))} - \frac{\exp(H_t(y)) \exp(H_t(a))}{(\sum_{y=1}^k \exp(H_t(y)))^2} \\
	&= \mathbb{1}_{a = x} \pi_t(x) - \pi_t(x) \pi_t(a) \\
	&= \pi_t(x) (\mathbb{1}_{a=x} - \pi_t(a)),
\end{align}
$$
thus showing that the expected updated of the gradient bandit algorithm is equivalent to the gradient of the expected reward, making the the algorithm a instance of SGA.

### Ch 2.9: Associative Search (Contextual Bandits)

*Associative search* tasks - which involve learning about which actions are the best through trial-and-error and associating these actions with which situations they work the best in - are often called *contextual bandits*. These tasks serve as an intermediate between the $$k$$-armed bandit problem and the full RL problem, since each action affects only the immediate reward - like the former - and also involves learning a policy, like the latter.

An example of an associative task is a one composed of several $$k$$-armed bandit problems, each identified by a given color, where at each step you are confronted with one of the $$k$$-armed bandit problems at random. If the action values change as the color changes, you can then learn a policy that maps a color to the best associated actions.

### Ch 2.10: Summary

W.r.t. performance (average reward) in the $$k$$-bandit problem, with $$k = 10$$ and taking into account the first 1000 steps, w.h.t. UCB $$\geq$$ Greedy with optimistic initialization $$\alpha = 0.1 \ \geq$$ Gradient bandit $$\geq$$ $$\epsilon$$-greedy. 

Another approach to balance exploration and exploitation in $$k$$-armed bandit problems is the Bayesian method known as *Gittins* index. It assumes a known prior distribution over the actions values and then updates the distribution after each step (assuming that the true action values are stationary).