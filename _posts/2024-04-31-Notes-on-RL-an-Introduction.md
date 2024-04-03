---
layout: post
title: "Notes on Reinforcement Learning: An Introduction (2nd edition)"
date: 2024-03-31
categories: ML RL DL
usemathjax: true
---
$$
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\argmax}{arg\,max}
$$

Here are some notes I took when reading the second edition of the <a href="http://acfpeacekeeper.github.io/github-pages/docs/literature/books/RLbook2020.pdf" onerror="this.href='http://localhost:4000/docs/literature/books/RLbook2020.pdf'">Reinforcement Learning: An Introduction</a> book.\\
If you want to get into Reinforcement Learning, or are just interested in Artificial Intelligence in general, I highly recommend that you read this book!\\
It does require some mathematical background to read and understand everything (mostly Linear Algebra, Probabilities, Statistics, and some Calculus), but it is overall one of the best - and most exhaustive - introductory books about Reinforcement Learning out there.
# Chapter Index
1. [Chapter 1: Introduction](#chapter-1-introduction)
2. [Part I: Tabular Solution Methods](#part-i-tabular-solution-methods)
	1. [Chapter 2: Multi-armed Bandits](#chapter-2-multi-armed-bandits)
		1. [Section 2.1: A k-armed Bandit Problem](#section-21-a--armed-bandit-problem)
		2. [Section 2.2: Action-value Methods](#section-22-action-value-methods)
		3. [Section 2.3: The 10-armed Test-bed](#section-23-the-10-armed-test-bed)
		4. [Section 2.4: Incremental Implementation](#section-24-incremental-implementation)
		5. [Section 2.5: Tracking a Non-stationary Problem](#section-25-tracking-a-non-stationary-problem)
		6. [Section 2.6: Optimistic Initial Values](#section-26-optimistic-initial-values)
		7. [Section 2.7: Upper-Confidence-Bound Action Selection](#section-27-upper-confidence-bound-action-selection)
		8. [Section 2.8: Gradient Bandit Algorithms](#section-28-gradient-bandit-algorithms)
		9. [Section 2.9: Associative Search (Contextual Bandits)](#section-29-associative-search-contextual-bandits)
		10. [Section 2.10: Summary](#section-210-summary)
	2. [Chapter 3: Finite Markov Decision Processes](#chapter-3-finite-markov-decision-processes)
		1. [Section 3.1: The Agent-Environment Interface](#section-31-the-agent-environment-interface)
		2. [Section 3.2: Goals and Rewards](#section-32-goals-and-rewards)
		3. [Section 3.3: Returns and Episodes](#section-33-returns-and-episodes)
		4. [Section 3.4: Unified Notation for Episodic and Continuing Tasks](#section-34-unified-notation-for-episodic-and-continuing-tasks)
		5. [Section 3.5: Policies and Value Functions](#section-35-policies-and-value-functions)
		6. [Section 3.6: Optimal Policies and Optimal Value Functions](#section-36-optimal-policies-and-optimal-value-functions)
		7. [Section 3.7: Optimality and Approximation](#section-37-optimality-and-approximation)
		8. [Section 3.8: Summary](#section-38-summary)
	3. [Chapter 4: Dynamic Programming](#chapter-4-dynamic-programming)
		1. [Section 4.1: Policy Evaluation (Prediction)](#section-41-policy-evaluation-prediction)
		2. [Section 4.2: Policy Improvement](#section-42-policy-improvement)
		3. [Section 4.3: Policy Iteration](#section-43-policy-iteration)
		4. [Section 4.4: Value Iteration](#section-44-value-iteration)
		5. [Section 4.5: Asynchronous Dynamic Programming](#section-45-asynchronous-dynamic-programming)
		6. [Section 4.6: Generalized Policy Iteration](#section-46-generalized-policy-iteration)
		7. [Section 4.7: Efficiency of Dynamic Programming](#section-47-efficiency-of-dynamic-programming)
		8. [Section 4.8: Summary](#section-48-summary)
	4. [Chapter 5: Monte Carlo Methods](#chapter-5-monte-carlo-methods)
		1. [Section 5.1: Monte Carlo Prediction](#section-51-monte-carlo-prediction)
		2. [Section 5.2: Monte Carlo Estimation of Action Values](#section-52-monte-carlo-estimation-of-action-values)
		3. [Section 5.3: Monte Carlo Control](#section-53-monte-carlo-control)
		4. [Section 5.4: Monte Carlo Control without Exploring Starts](#section-54-monte-carlo-control-without-exploring-starts)
		5. [Section 5.5: Off-policy Prediction via Importance Sampling](#section-55-off-policy-prediction-via-importance-sampling)
		6. [Section 5.6: Incremental Implementation](#section-56-incremental-implementation)
		7. [Section 5.7: Off-policy Monte Carlo Control](#section-57-off-policy-monte-carlo-control)
		8. [Section 5.8: *Discounting-aware Importance Sampling](#section-58-discounting-aware-importance-sampling)
		9. [Section 5.9: *Per-decision Importance Sampling](#section-59-per-decision-importance-sampling)
		10. [Section 5.10: Summary](#section-510-summary)
	5. [Chapter  6: Temporal-Difference Learning](#chapter-6-temporal-difference-learning)
		1. [Section 6.1: TD Prediction](#section-61-td-prediction)
		2. [Section 6.2: Advantages of TD Prediction Methods](#section-62-advantages-of-td-prediction-methods)
		3. [Section 6.3: Optimality of TD(0)](#section-63-optimality-of-td0)
		4. [Section 6.4: Sarsa: On-policy TD Control](#section-64-sarsa-on-policy-td-control)
		5. [Section 6.5: Q-learning: Off-policy TD Control](#section-65-q-learning-off-policy-td-control)
		6. [Section 6.6: Expected Sarsa](#section-66-expected-sarsa)
		7. [Section 6.7: Maximization Bias and Double Learning](#section-67-maximization-bias-and-double-learning)
		8. [Section 6.8: Games, Afterstates, and Other Special Cases](#section-68-games-afterstates-and-other-special-cases)
		9. [Section 6.9: Summary](#section-69-summary)
	6. [Chapter 7: n-step Bootstrapping](#chapter-7--step-bootstrapping)
		1. [Section 7.1: n-step TD Prediction](#section-71--step-td-prediction)
		2. [Section 7.2: n-step Sarsa](#section-72--step-sarsa)
		3. [Section 7.3: n-step Off-policy Learning](#section-73--step-off-policy-learning)
		4. [Section 7.4: *Per-decision Methods with Control Variates](#section-74-per-decision-methods-with-control-variates)
		5. [Section 7.5: Off-policy Learning Without Importance Sampling: The n-step Tree Backup Algorithm](#section-75-off-policy-learning-without-importance-sampling-the--step-tree-backup-algorithm)
		6. [Section 7.6: *A Unifying Algorithm: n-step Q(sigma)](#section-76-a-unifying-algorithm--step)
		7. [Section 7.7: Summary](#section-77-summary)
	7. [Chapter 8: Planning and Learning with Tabular Methods](#chapter-8-planning-and-learning-with-tabula-methods)
		1. [Section 8.1: Models and Planning](#section-81-models-and-planning)
		2. [Section 8.2: Dyna: Integrated Planning, Acting, and Learning](#section-82-dyna-integrated-planning-acting-and-learning)
		3. [Section 8.3: When the Model Is Wrong](#section-83-when-the-model-is-wrong)
		4. [Section 8.4: Prioritized Sweeping](#section-84-prioritized-sweeping)
		5. [Section 8.5: Expected vs. Sample Updates](#section-85-expected-vs-sample-updates)
		6. [Section 8.6: Trajectory Sampling](#section-86-trajectory-sampling)
		7. [Section 8.7: Real-Time Dynamic Programming](#section-87-real-time-dynamic-programming)
		8. [Section 8.8: Planning at Decision Time](#section-88-planning-at-decision-time)
		9. [Section 8.9: Heuristic Search](#section-89-heuristic-search)
		10. [Section 8.10: Rollout Algorithms](#section-810-rollout-algorithms)
		11. [Section 8.11: Monte Carlo Tree Search](#section-811-monte-carlo-tree-search)
		12. [Section 8.12: Summary of the Chapter](#section-812-summary-of-the-chapter)
		13. [Section 8.13: Summary of Part I: Dimensions](#section-813-summary-of-part-i-dimensions)
3. [Part II: Approximate Solution Methods](#part-ii-approximate-solution-methods)
	1. [Chapter 9: On-policy Prediction with Approximation](#chapter-9-on-policy-prediction-with-approximation)
		1. [Section 9.1: Value-function Approximation](#section-91-value-function-approximation)
		2. [Section 9.2: The Prediction Objective (VE)](#section-92-the-prediction-objective)
		3. [Section 9.3: Stochastic-gradient and Semi-gradient Methods](#section-93-stochastic-gradient-and-semi-gradient-methods)
		4. [Section 9.4: Linear Methods](#section-94-linear-methods)
		5. [Section 9.5: Feature Construction for Linear Methods](#section-95-feature-construction-for-linear-methods)
			1. [Section 9.5.1: Polynomials](#section-951-polynomials)
			2. [Section 9.5.2: Fourier Basis](#section-952-fourier-basis)
			3. [Section 9.5.3: Coarse Coding](#section-953-coarse-coding)
			4. [Section 9.5.4: Tile Coding](#section-954-tile-coding)
			5. [Section 9.5.5: Radial Basis Functions](#section-955-radial-basis-functions)
		6. [Section 9.6: Selecting Step-Size Parameters Manually](#section-96-selecting-step-size-parameters-manually)
		7. [Section 9.7: Non-linear Function Approximation: Artificial Neural Networks](#section-97-non-linear-function-approximation-artificial-neural-networks)
		8. [Section 9.8: Least-Squares TD](#section-98-least-squares-td)
		9. [Section 9.9: Memory-based Function Approximation](#section-99-memory-based-function-approximation)
		10. [Section 9.10: Kernel-based Function Approximation](#section-910-kernel-based-function-approximation)
		11. [Section 9.11: Looking Deeper at On-policy Learning: Interests and Emphasis](#section-911-looking-deeper-at-on-policy-learning-interests-and-emphasis)
		12. [Section 9.12: Summary](#section-912-summary)
	2. [Chapter 10: On-policy Control with Approximation](#chapter-10-on-policy-control-with-approximation)
		1. [Section 10.1: Episodic Semi-gradient Control](#section-101-episodic-semi-gradient-control)
		2. [Section 10.2: Semi-gradient n-step Sarsa](#section-102-semi-gradient--step-sarsa)
		3. [Section 10.3: Average Reward: A New Problem Setting for Continuing Tasks](#section-103-average-reward-a-new-problem-setting-for-continuing-tasks)
		4. [Section 10.4: Deprecating the Discounted Setting](#section-104-deprecating-the-discounted-setting)
		5. [Section 10.5: Differential Semi-gradient n-step Sarsa](#section-105-differential-semi-gradient--step-sarsa)
		6. [Section 10.6: Summary](#section-106-summary)
	3. [Chapter 11: *Off-policy Methods with Approximation](#chapter-11-off-policy-methods-with-approximation)
		1. [Section 11.1: Semi-gradient Methods](#section-111-semi-gradient-methods)
		2. [Section 11.2: Examples of Off-policy Divergence](#section-112-examples-of-off-policy-divergence)
		3. [Section 11.3: The Deadly Triad](#section-113-the-deadly-triad)
		4. [Section 11.4: Linear Value-function Geometry](#section-114-linear-value-function-geometry)
		5. [Section 11.5: Gradient Descent in the Bellman Error](#section-115-gradient-descent-in-the-bellman-error)
		6. [Section 11.6: The Bellman Error is Not Learnable](#section-116-the-bellman-error-is-not-learnable)
		7. [Section 11.7: Gradient-TD Methods](#section-117-gradient-td-methods)
		8. [Section 11.8: Emphatic-TD Methods](#section-118-emphatic-td-methods)
		9. [Section 11.9: Reducing Variance](#section-119-reducing-variance)
		10. [Section 11.10: Summary](#section-1110-summary)
	4. [Chapter 12: Eligibility Traces](#chapter-12-eligibility-traces)
		1. [Section 12.1: The lambda-return](#section-121-the--return)
		2. [Section 12.2: TD(lambda)](#section-122-td)
		3. [Section 12.3: n-step Truncated lambda-return Methods](#section-123--step-truncated--return-methods)
		4. [Section 12.4: Redoing Updates: Online lambda-return Algorithm](#section-124-redoing-updates-online--return-algorithm)
		5. [Section 12.5: True Online TD(lambda)](#section-125-true-online-td)
		6. [Section 12.6: *Dutch Traces in Monte Carlo Learning](#section-126-dutch-traces-in-monte-carlo-learning)
		7. [Section 12.7: Sarsa(lambda)](#section-127-sarsa)
		8. [Section 12.8: Variable lambda and gamma](#section-128-variable-and)
		9. [Section 12.9: Off-policy Traces with Control Variates](#section-129-off-policy-traces-with-control-variates)
		10. [Section 12.10: Watkin's Q(lambda) to Tree-Backup(lambda)](#section-1210-watkins-q-to-tree-backup)
		11. [Section 12.11: Stable Off-policy Methods with Traces](#section-1211-stable-off-policy-methods-with-traces)
		12. [Section 12.12: Implementation Issues](#section-1212-implementation-issues)
		13. [Section 12.13: Conclusions](#section-1213-conclusions)
	5. [Chapter 13: Policy Gradient Methods](#chapter-13-policy-gradient-methods)
		1. [Section 13.1: Policy Approximation and its Advantages](#section-131-policy-approximation-and-its-advantages)
		2. [Section 13.2: The Policy Gradient Theorem](#section-132-the-policy-gradient-theorem)
		3. [Section 13.3: REINFORCE: Monte Carlo Policy Gradient](#section-133-reinforce-monte-carlo-policy-gradient)
		4. [Section 13.4: REINFORCE with Baseline](#section-134-reinforce-with-baseline)
		5. [Section 13.5: Actor-Critic Methods](#section-135-actor-critic-methods)
		6. [Section 13.6: Policy Gradient for Continuing Problems](#section-136-policy-gradient-for-continuing-problems)
		7. [Section 13.7: Policy Parameterization for Continuous Actions](#section-137-policy-parameterization-for-continuous-actions)
		8. [Section 13.8: Summary](#section-138-summary)
4. [Part III: Looking Deeper](#part-iii-looking-deeper)
	1. [Chapter 14: Psychology](#chapter-14-psychology)
		1. [Section 14.1: Prediction and Control](#section-141-prediction-and-control)
		2. [Section 14.2: Classical Conditioning](#section-142-classical-conditioning)
			1. [Section 14.2.1: Blocking and Higher-order Conditioning](#section-1421-blocking-and-higher-order-conditioning)
			2. [Section 14.2.2: The Rescorla-Wagner Model](#section-1422-the-rescorla-wagner-model)
			3. [Section 14.2.3: The TD Model](#section-1423-the-td-model)
			4. [Section 14.2.4: TD Model Simulations](#section-1424-td-model-simulations)
		3. [Section 14.3: Instrumental Conditioning](#section-143-instrumental-conditioning)
		4. [Section 14.4: Delayed Reinforcement](#section-144-delayed-reinforcement)
		5. [Section 14.5: Cognitive Maps](#section-145-cognitive-maps)
		6. [Section 14.6: Habitual and Goal-directed Behavior](#section-146-habitual-and-goal-directed-behavior)
		7. [Section 14.7: Summary](#section-147-summary)
	2. [Chapter 15: Neuroscience](#chapter-15-neuroscience)
		1. [Section 15.1: Neuroscience Basics](#section-151-neuroscience-basics)
		2. [Section 15.2: Reward Signals, Reinforcement Signals, Values, and Prediction Errors](#section-152-reward-signals-reinforcement-signals-values-and-prediction-errors)
		3. [Section 15.3: The Reward Prediction Error Hypothesis](#section-153-the-reward-prediction-error-hypothesis)
		4. [Section 15.4: Dopamine](#section-154-dopamine)
		5. [Section 15.5: Experimental Support for the Reward Prediction Error Hypothesis](#section-155-experimental-support-for-the-reward-prediction-error-hypothesis)
		6. [Section 15.6: TD Error/Dopamine Correspondence](#section-156-td-errordopamine-correspondence)
		7. [Section 15.7: Neural Actor–Critic](#section-157-neural-actor–critic)
		8. [Section 15.8: Actor and Critic Learning Rules](#section-158-actor-and-critic-learning-rules)
		9. [Section 15.9: Hedonistic Neurons](#section-159-hedonistic-neurons)
		10. [Section 15.10: Collective Reinforcement Learning](#section-1510-collective-reinforcement-learning)
		11. [Section 15.11: Model-based Methods in the Brain](#section-1511-model-based-methods-in-the-brain)
		12. [Section 15.12: Addiction](#section-1512-addiction)
		13. [Section 15.13: Summary](#section-1513-summary)
	3. [Chapter 16: Applications and Case Studies](#chapter-16-applications-and-case-studies)
		1. [Section 16.1: TD-Gammon](#section-161-td-gammon)
		2. [Section 16.2: Samuel's Checkers Player](#section-162-samuels-checkers-player)
		3. [Section 16.3: Watson's Daily-Double Wagering](#section-163-watsons-daily-double-wagering)
		4. [Section 16.4: Optimizing Memory Control](#section-164-optimizing-memory-control)
		5. [Section 16.5: Human-level Video Game Play](#section-165-human-level-video-game-play)
		6. [Section 16.6: Mastering the Game of Go](#section-166-mastering-the-game-of-go)
			1. [Section 16.6.1: AlphaGo](#section-1661-alphago)
			2. [Section 16.6.2: AlphaGo Zero](#section-1662-alphago-zero)
		7. [Section 16.7: Personalized Web Services](#section-167-personalized-web-services)
		8. [Section 16.8: Thermal Soaring](#section-168-thermal-soaring)
	4. [Chapter 17: Frontiers](#chapter-17-frontiers)
		1. [Section 17.1: General Value Functions and Auxiliary Tasks](#section-171-general-value-functions-and-auxiliary-tasks)
		2. [Section 17.2: Temporal Abstraction via Options](#section-172-temporal-abstraction-via-options)
		3. [Section 17.3: Observations and State](#section-173-observations-and-state)
		4. [Section 17.4: Designing Reward Signals](#section-174-designing-reward-signals)
		5. [Section 17.5: Remaining Issues](#section-175-remaining-issues)
		6. [Section 17.6: Reinforcement Learning and the Future of Artificial Intelligence](#section-176-reinforcement-learning-and-the-future-of-artificial-intelligence)

# Chapter 1: Introduction

Def. **Reinforcement Learning (RL)**: an agent learns how to map situations to actions through *trial-and-error* or *planned* interaction with a (possibly) uncertain environment, so as to maximize a numerical reward value (i.e., achieve his goal or goals).
- *Delayed reward* is another important characteristic of RL, since any action taken may influence (not only the immediate reward value, but also) any subsequent rewards;
- RL can be formalized as the optimal control of incompletely-known Markov Decision Processes (MDPs).

Besides RL, other **Machine Learning (ML)** paradigms include *Supervised Learning* - predicting the correct label, given the corresponding set of features - and *Unsupervised Learning* - finding hidden patterns in a collection of unlabeled features.

A challenge unique to the RL paradigm is that of the trade-off between **exploration versus exploitation**. This challenge arises due to the fact that an agent prefers to take the actions that have previously given the highest rewards (*exploitation*), but it must also try out other actions in order to have more knowledge about which actions it should select (*exploration*).

A RL system has four main elements beyond the interactive **agent** and the **environment**, which are:
- A **policy** $$\pi_t: s \rightarrow a$$, which in stochastic cases specifies a probability for each action;
- A **reward** $$r(s, a)$$, an immediate signal that specifies how good it is for an agent to have chosen a certain action in a given state (may also be stochastic);
- A **value function** $$v(s)$$ that specifies the total reward an agent is expected to accumulate in the future if he starts at a given state, i.e., predicted long-term reward;
- A (optional) **world model** used by model-based methods (opposed to purely trial-and-error model-free methods) for planning.

# Part I: Tabular Solution Methods

## Chapter 2: Multi-armed Bandits

*Non-associative* setting: a problem setting that involves learning to act in only 1 situation.

*Associative* setting: a problem setting where the best action depends on the situation.

### Section 2.1: A $$k$$-armed Bandit Problem

Setting of the $$k$$-armed bandit learning problem (analogous to a slot machine with $$k$$ levers):
1. Choose 1 action from among $$k$$ different options;
2. Receive a (numerical) reward from a stationary probability distribution which depends on the action selected;
3. Repeat steps 1 and 2 with the purpose of maximizing the expected total reward over some time period (e.g., 1000 action selections or *time steps*).

**Value** of an action: the expected or mean reward received if that action is selected

Letting $$A_t$$ be the action taken at time step $$t$$ and $$R_t$$ the corresponding reward, then the value $$q^{*}(a)$$ of an arbitrary action $$a$$ is given by:

$$
\begin{equation}
    q^{*} (a) \doteq \mathbb{E} [R_t | A_t = a].
\end{equation}
$$

Since we do not know the true value of each action, we need to estimate them in such a way that the estimates are close to the real values. The estimated value of an action $$a$$ at time step $$t$$ is denoted by $$Q_t (a)$$.

**Greedy** action: the action with the highest estimated value at a given time step
- Choosing this action equates to the agent **exploiting** his current knowledge of the values of the actions;
- Selecting 1 of the non-greedy actions enables the agent to improve his estimates of the non-greedy action's value, i.e., **exploration**;
- Exploitation maximizes the reward on 1 step, but it needs to be intercalated with exploration steps so as to maximize the greater total reward in the long term.

### Section 2.2: Action-value Methods

Def. **Action-value Methods**: methods used to estimate the values of actions and to use those estimates to select an action to take at a given time step.

Letting $$\mathbb{1}_{predicate}$$ be the random variable which equals $$1$$ if the $$predicate$$ is true and $$0$$ otherwise, the value of an action can be estimated by averaging the rewards received: <!-- TODO: check if equations inside text inside equations need double $ signs-->

$$
\begin{equation}
Q_t (a) \doteq \frac{\text{sum of rewards when $a$ taken prior to $t$}}{\text{number of times $a$ taken prior to $t$}} = \frac{\sum_{i = 1}^{t - 1} R_i \cdot \mathbb{1}_{A_i = a}}{\sum_{i = 1}^{t - 1} \mathbb{1}_{A_i = a}}.
\end{equation}
$$

If the denominator is zero (action has never been taken), then $$Q_t(a)$$ is defined as an arbitrary default value (e.g., zero). By the law of large numbers, as the denominator goes to infinity, $$Q_t(a)$$ converges to $$q^{*}(a)$$. This is called the *sample-average* method for estimating action values.

The simplest action selection rule is to always select a greedy action and - if there is more than 1 action with the same highest value - to break ties in some arbitrary way (e.g., randomly). This action selection method can be written as:

$$
\begin{equation}
A_t = \argmax_a Q_t (a).
\end{equation}
$$

This selection method never performs exploration. A simple alternative that does so is to select the greedy action most of the time (probability $$1 - \epsilon$$) and (with probability $$\epsilon$$) to randomly select any possible action with equal probability. Methods that use this near-greedy action selection rule are dubbed $$\epsilon$$-greedy methods.

### Section 2.3: The 10-armed Test-bed

**Non-stationary** setting: problem setting where the true values of the actions (or the reward probabilities) change over time.

Given a set of 2000 randomly generated $$k$$-armed bandit problems (with $$k = 10$$), for each problem in the set, the action values $$q^{*}(a), \ a = \{1, 2, \dots, 10\},$$ were selected from a normal (Gaussian) distribution with $$\mu = 0, \  \sigma^2 = 1$$. When a learning method is applied to this problem selects action $$A_t$$ at time step $$t$$, the actual reward ($$R_t$$) was drawn from a normal distribution with $$\mu = q^{*}(A_t), \ \sigma^2 = 1$$.

The performance of the learning methods is measured as it improves with experience over 1000 time steps of the bandit problem, which makes up a single run. To obtain an accurate measure of the learning algorithms' behavior, 2000 runs are performed and the results for the bandit problems are averaged.

A greedy action selection method is compared against 2 $$\epsilon$$-greedy methods (with $$\epsilon = 0.01 \lor \epsilon = 0.1$$). All methods begin with initial action-value estimates of zero and update these estimates using the sample-average technique.

While the greedy method improved slightly faster than the other 2, it converged to a reward-per-step of 1, which is lower than the best value of around 1.54 achieved by the $$\epsilon$$-greedy method (with $$\epsilon = 0.1$$). The method with $$\epsilon = 0.1$$ improved faster than the method with $$\epsilon = 0.01$$, since it explored more earlier. However, the method with $$\epsilon = 0.01$$ converges to a higher reward-per-step in the long run, since the method with $$\epsilon = 0.1$$ never selects the optimal action more than 91% of the time. 
It is possible to perform $$\epsilon$$ annealing to try to get fast learning at the start combined with convergence to a higher reward average.

It takes more exploration to find the optimal actions in cases with noisy rewards (i.e., high reward variance), meaning that $$\epsilon$$-greedy methods perform even better in those cases, when compared to the greedy method. Also, although the greedy method is theoretically optimal in the deterministic case (i.e., with $$\sigma^2 = 0$$), this property does not hold in non-stationary bandit problems, making exploration a necessity even in deterministic settings.

### Section 2.4: Incremental Implementation

For a single action, let $$R_i$$ denote the reward received after the $$i^{th}$$ selection of *this action* and $$Q_n$$ the estimate of its action value after it has been selected $$n - 1$$ times, written as:

$$
\begin{equation}
Q_n \doteq \frac{R_1 + R_2 + \dots + R_{n - 1}}{n - 1}.
\end{equation}
$$

Instead of maintaining a record of all the rewards and performing the computation for the estimated value whenever needed (resulting in the growth of both computational and memory requirements), we can devise incremental formulas to update the averages with a small and constant computation to process each new reward. Given $$Q_n$$ and the $$n^{th}$$ reward $$R_n$$, the new average of all $$n$$ rewards can be computed as:

$$
\begin{align}
	Q_{n + 1} &= \frac{1}{n} \sum_{i = 1}^n R_i \nonumber\\
	&= \frac{1}{n}(R_n + \sum_{i = 1}^{n - 1} R_i) \nonumber\\
	&= \frac{1}{n}(R_n + (n - 1) \cdot \frac{1}{n - 1} \cdot \sum_{i = 1}^{n - 1} R_i) \nonumber\\
	&= \frac{1}{n} (R_n + (n - 1) \cdot Q_n) \nonumber\\
	&= \frac{1}{n} (R_n + n \cdot Q_n - Q_n) \nonumber\\
	&= Q_n + \frac{1}{n} [R_n - Q_n], \ n > 1 \\
	Q_2 &= R_1, \ Q_1 \in \mathbb{R}.
\end{align}
$$

This implementation only needs memory for $$Q_n$$ and $$n$$, and only performs a small computation for each new reward. 
The general form of the previous update rule is given by:

$$
\begin{equation}
NewEstimate \leftarrow OldEstimate + StepSize [Target - OldEstimate],
\end{equation}
$$

where $$[Target - OldEstimate]$$ is an *error* in the estimate, which is reduced by taking a step towards the (possibly noisy) target value.
The step-size parameter is generally denoted by $$\alpha$$ or $$\alpha_t (a)$$.

For some examples of Python implementations of Bandit Problems, <a href="https://acfpeacekeeper.github.io/github-pages/ml/rl/2024/03/27/k-armed-Bandit-Problems.html" onerror="this.href='http://localhost:4000/ml/rl/2024/03/27/k-armed-Bandit-Problems.html'">see here</a>.

### Section 2.5: Tracking a Non-stationary Problem

When reward probabilities change over time, it makes sense to give more weight to recent rewards than to those receive long ago. This can be done by using a constant step-size parameter, e.g., for updating an average $$Q_n$$ of the $$n - 1$$ past rewards w.h.t.:

$$
\begin{equation}
Q_{n + 1} \doteq Q_n + \alpha [R_n - Q_n],
\end{equation}
$$

where the step-size parameter $$\alpha \in \  ]0, 1]$$ is constant. Given this, $$Q_{n + 1}$$ becomes a weighted average (since the sum of weights = 1) of the past rewards and initial estimate $$Q_1$$:

$$
\begin{align}
	Q_{n + 1} &= Q_n + \alpha [R_n - Q_n] \nonumber\\
	&= \alpha \cdot R_n + (1- \alpha) Q_n \nonumber\\
	&= \alpha \cdot R_n + (1 - \alpha) [\alpha \cdot R_{n - 1} + (1 - \alpha)Q_{n - 1}] \nonumber\\
	&= \alpha \cdot R_n + (1 - \alpha) \cdot \alpha \cdot R_{n - 1} + (1 - \alpha)^2 Q_{n - 1} \nonumber\\
	&= \alpha \cdot R_n + (1 - \alpha) \cdot \alpha \cdot R_{n - 1} + \dots + (1 - \alpha)^{n - 1} \cdot \alpha \cdot R_1 + (1 - \alpha)^n \cdot Q_1 \nonumber\\
	&= (1 - \alpha)^n \cdot Q_1 + \sum_{i = 1}^n \alpha \cdot (1 - \alpha)^{n - i} \cdot R_i.
\end{align}
$$

Since $$1 - \alpha < 1$$, the weight given to $$R_i$$ decreases as the number of intervening rewards increases. Also, the weight decays exponentially in proportion to the exponent on $$1 - \alpha$$ and, if $$1 - \alpha = 0$$, the entire weight goes onto the very last reward $$R_n$$. This method is sometimes called an *exponential recency-weighted average*. 

Letting $$\alpha_n (a)$$ denote the step-size parameter to process the reward obtained after the $$n^{th}$$ selection of action $$a$$, for the sample-average method, w.h.t. $$\alpha_n (a) = 1/n$$, whose convergence to the true action values is guaranteed by the law of large numbers. However, convergence is **NOT** guaranteed for all choices of the $$\{\alpha_n (a)\}$$ sequence. Through a result in stochastic approximation theory, we obtain the conditions required to assure convergence with probability 1:

$$
\begin{equation}
\sum_{n = 1}^{\infty} \alpha_n (a) = \infty \quad \land \quad \sum_{n = 1}^{\infty} \alpha_n^2 (a) < \infty,
\end{equation}
$$

where the first condition is required to guarantee that the steps are big enough to overcome any initial conditions or random fluctuations that would otherwise result in getting stuck at saddle points, and the second condition guarantees that the steps will eventually become small enough to assure convergence.

The second condition is not met for the constant step-size parameter case, i.e., $$\alpha_n (a) = \alpha$$. This means that the estimates will never completely converge, which is actually a desirable property for non-stationary problems (the most common type of problem in RL), since the estimates continue to vary in response to the most recently received rewards, accounting for the changes in reward probabilities over time. Also, the sequences of step-size parameters that meet both of the above conditions often lead to slow convergence rates, meaning that these are seldomly used in applications and empirical research.

### Section 2.6: Optimistic Initial Values

All previous methods are somewhat dependent on the initial action-value estimates $$Q_1 (a)$$, i.e., they are *biased* by their initial estimates. This bias decreases over time as various actions are selected. However, while for sample-average methods the bias eventually disappear after all actions have been taken at least once, the bias is permanent for methods with a constant $$\alpha$$. 

This property means that, when using methods with a constant $$\alpha$$, the user must select the values for the initial estimates, which provides a way to supply some prior knowledge about the expected rewards, at the possible cost of being harder to tune.

By selecting optimistic initial action-values, i.e., $$Q_1 (a) >> R_1 (a), \forall a$$, the agent will always be disappointed since the rewards will always be far less than the first estimates, regardless of which actions are selected. This encourages exploration, as the agent will select all possible actions before the value estimates converge, even if greedy actions are selected at every single time step.

This technique for encouraging exploration is named *optimistic initial values* and is a simple, yet effective trick when used on stationary problems (e.g., with $$Q_1(a) = 5$$ it outperforms a $$\epsilon$$-greedy method with $$Q_1(a) = 0$$ and $$\epsilon = 0.1$$). 

An important caveat is that, since the drive for exploration in the previous technique is dependent on the initials conditions and disappears after a certain time, it cannot adequately deal with non-stationary problems, where exploration is always required due to the dynamic nature of the reward probabilities. This drawback is present in all methods that treat the beginning of time as a special event (e.g., the sample-average methods).

### Section 2.7: Upper-Confidence-Bound Action Selection

While $$\epsilon$$-greedy methods encourage exploration, they do so equally, without any preference for whether the action selected is nearly greedy or particularly uncertain. However, it is possible to select the non-greedy actions while taking into account both how close their value estimates are to the maximal action-value and the estimation uncertainty. An effective way of doing this is to select actions according to the following equation:

$$
\begin{equation}
A_t \doteq \argmax_a [Q_t(a) + c \cdot \sqrt{\frac{\ln t}{N_t(a)}}],
\end{equation}
$$

where $$N_t(a)$$ denotes the number of times the action $$a$$ has been selected prior to time $$t$$ and the number $$c > 0$$ controls the degree of exploration. If $$N_t(a) = 0$$, then $$a$$ is considered to be a maximizing action.

This **upper confidence bound (UCB)** action selection is based on the idea that the square-root term is a measure of the uncertainty or variance of action $$a$$ value's estimate. As such, the max'ed over quantity becomes a sort of upper bound on the possible true value of action $$a$$ with $$c$$ determining the confidence level, and thus the uncertainty is reduced each time the action $$a$$ is selected. 

The natural logarithm results in smaller increases over time, meaning that actions with lower value estimates or that have been frequently selected, will be selected with decreasing frequency.

UCB often performs better than $$\epsilon$$-greedy action selection (except in the first $$k$$ steps), but it is harder to extend beyond bandits into the general RL settings. This is due to its difficulties in dealing with more advanced settings, such as non-stationary problems and (function approximation) with large state spaces.

### Section 2.8: Gradient Bandit Algorithms

Beyond using action-value estimates to select actions, it is also possible to learn a numerical *preference* for each action $$a$$, denoted $$H_t(a) \in \mathbb{R}$$, which has no interpretation w.r.t. reward. As such, only the relative preference of 1 action over another is important. The action probabilities are determined according to a softmax distribution as follows:

$$
\begin{equation}
Pr\{A_t = a\} \doteq \frac{\exp(H_t(a))}{\sum_{b = 1}^k \exp(H_t(b))} \doteq \pi_t (a),
\end{equation}
$$

where $$\pi_t(a)$$ is the probability of taking action $$a$$ at time $$t$$. All actions have an equal probability of being selected at first (i.e., $$H_1(a) = 0, \forall a$$).

There exists a natural learning algorithm for softmax action preferences based on the idea of **Stochastic Gradient Ascent (SGA)**, where, at each time step, after selecting action $$A_t$$, and receiving the reward $$R_t$$, the action preferences are updated as follows:

$$
\begin{align}
	H_{t + 1}(A_t) &\doteq H_t(A_t) + \alpha (R_t - \bar{R_t}) (1 - \pi_t(A_t)), &\text{and}\\
	H_{t + 1}(a) &\doteq H_t(a) - \alpha (R_t \bar{R_t}) \pi_t(a), &\forall a \neq A_t,
\end{align}
$$

where $$\alpha > 0$$ is a step-size parameter and $$\bar{R}_t \in \mathbb{R}$$ - which serves as baseline to compare against the reward - is the average of the rewards up to but not including time $$t$$ (with $$\bar{R}_1 \doteq R_1$$). If $$R_t > \bar{R}_t, \  t \neq 1$$, then the probability of taking $$A_t$$ in the future is increased, otherwise, the probability of taking $$A_t$$ is decreased if $$R_t < \bar{R}_t$$. Also, the unselected actions probabilities are updated in the opposite direction.

Since only the relative preferences are taken into account, adding an arbitrary constant value to all the action preferences has no effect on the action probabilities. Also, since the reward baseline term instantaneously adapts to new values of the mean, shifting the mean (e.g., $$\mu_{new} = \mu_{old} + 4$$) of the distribution (while keeping the unit variance) has no effect on the gradient bandit algorithm. However, omitting the baseline term results in a significantly degraded performance.

#### The Bandit Gradient Algorithm as SGA

In exact **Gradient Ascent (GA)**, each action preference $$H_t(a)$$ would be incremented in proportion to the increment's effect on performance, given by:

$$
\begin{equation}
H_{t + 1} (a) \doteq H_t(a) + \alpha \frac{\partial \mathbb{E}[R_t]}{\partial H_t (a)},
\end{equation}
$$

where the measure of performance is the expected reward:

$$
\begin{equation}
\mathbb{E}[R_t] = \sum_x \pi_t (x) \cdot q^{*} (x),
\end{equation}
$$

and the measure of the increment's effect is the *partial derivative* of this performance measure w.r.t. the action preference. Since $$q^{*}(x)$$ is not known, it is impossible to use exact GA. As such, the updates will instead take the form of those used in SGA.

The exact performance gradient can be written as:

$$
\begin{equation}
\frac{\partial \mathbb{E} [R_t]}{\partial H_t(a)} = \frac{\partial [\sum_x \pi_t(x) \cdot q^{*}(x)]}{\partial H_t(a)} = \sum_x q^{*}(x) \frac{\partial \pi_t(x)}{\partial H_t(a)} = \sum_x (q^{*}(x) - B_t) \frac{\partial \pi_t(x)}{\partial H_t(a)},
\end{equation}
$$

where the baseline $$B_t$$ can be any scalar value that doesn't depend on $$x$$. Since the sum of probabilities is always one, the sum of the changes $$\sum_x \frac{\partial \pi_t (x)}{\partial H_t (a)} = 0$$, and the baseline can be added without changing the equality.
We continue by multiplying each term of the sum by $$\pi_t(x)/\pi_t(x)$$, as follows:

$$
\begin{align}
	\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} &= \sum_x \pi_t(x) \cdot (q^{*}(x) - B_t) \cdot \frac{\partial \pi_t (x)}{\partial H_t(x)} / \pi_t(x) \nonumber\\
	&= \mathbb{E} [(q^{*}(A_t) - B_t) \cdot \frac{\partial \pi_t (A_t)}{\partial H_t(a)}/\pi_t(A_t)] \nonumber\\
	&= \mathbb{E}[(R_t - \bar{R_t}) \cdot \frac{\partial \pi_t(A_t)}{\partial H_t(a)}/\pi_t(A_t)] \nonumber\\
	&= \mathbb{E} [(R_t - \bar{R}_t) \cdot \pi_t(A_t) \cdot (\mathbb{1}_{a = A_t} - \pi_t(a))/\pi_t(A_t)] \nonumber\\
	&= \mathbb{E}[(R_t - \bar{R}_t) \cdot (\mathbb{1}_{a = A_t} - \pi_t(a))],
\end{align}
$$

where the chosen baseline is $$B_t = \bar{R_t}$$ and $$R_t$$ is substituted for $$q^{*}(A_t)$$, which is allowed since $$\mathbb{E}[R_t|A_t] = q^{*}(A_t)$$. 
By substituting a sample of the expectation above for the performance gradient, w.h.t.:

$$
\begin{equation}
H_{t + 1}(a) = H_t (a) + \alpha \cdot (R_t - \bar{R}_t) \cdot (\mathbb{1}_{a = A_t} - \pi_t(a)), \quad \forall a,
\end{equation}
$$

which is equivalent to the original algorithm.
By recalling the standard quotient rule for derivatives

$$
\begin{equation}
\frac{\partial}{\partial x}[\frac{f(x)}{g(x)}] = \frac{\frac{\partial f(x)}{\partial x}g(x) - f(x)\frac{\partial g(x)}{\partial x}}{g(x)^2},
\end{equation}
$$

we can then write

$$
\begin{align}
	\frac{\partial \pi_t(x)}{\partial H_t(a)} &= \frac{\partial}{\partial H_t(a)} \pi_t(x) \nonumber\\
	&= \frac{\partial}{\partial H_t(a)} [\frac{\exp(H_t(x))}{\sum_{y=1}^k \exp(H_t(y))}] \nonumber\\
	&= \frac{\frac{\partial \exp(H_t(x))}{\partial H_t(a)} \sum_{y=1}^k \exp(H_t(y)) - \exp(H_t(x)) \cdot \exp (H_t(a))}{(\sum_{y=1}^k \exp(H_t(y)))^2} \nonumber\\
	&= \frac{\mathbb{1}_{a=x \exp(H_t(x))} \sum_{y=1}^k \exp(H_t(y)) - \exp(H_t(x)) \exp(H_t(a))}{(\sum_{y=1}^k \exp(H_t(y)))^2} \nonumber\\
	&= \frac{\mathbb{1}_{a = x \exp(H_t(x))}}{\sum_{y=1}^k \exp(H_t(y))} - \frac{\exp(H_t(y)) \exp(H_t(a))}{(\sum_{y=1}^k \exp(H_t(y)))^2} \nonumber\\
	&= \mathbb{1}_{a = x} \pi_t(x) - \pi_t(x) \pi_t(a) \nonumber\\
	&= \pi_t(x) (\mathbb{1}_{a=x} - \pi_t(a)),
\end{align}
$$

thus showing that the expected updated of the gradient bandit algorithm is equivalent to the gradient of the expected reward, making the the algorithm a instance of SGA.

### Section 2.9: Associative Search (Contextual Bandits)

*Associative search* tasks - which involve learning about which actions are the best through trial-and-error and associating these actions with which situations they work the best in - are often called *contextual bandits*. These tasks serve as an intermediate between the $$k$$-armed bandit problem and the full RL problem, since each action affects only the immediate reward - like the former - and also involves learning a policy, like the latter.

An example of an associative task is a one composed of several $$k$$-armed bandit problems, each identified by a given color, where at each step you are confronted with one of the $$k$$-armed bandit problems at random. If the action values change as the color changes, you can then learn a policy that maps a color to the best associated actions.

### Section 2.10: Summary

W.r.t. performance (average reward) in the $$k$$-bandit problem, with $$k = 10$$ and taking into account the first 1000 steps, w.h.t. UCB $$\geq$$ Greedy with optimistic initialization $$\alpha = 0.1 \ \geq$$ Gradient bandit $$\geq$$ $$\epsilon$$-greedy. 

Another approach to balance exploration and exploitation in $$k$$-armed bandit problems is the Bayesian method known as *Gittins* index. It assumes a known prior distribution over the actions values and then updates the distribution after each step (assuming that the true action values are stationary).

## Chapter 3: Finite Markov Decision Processes

Markov Decision Processes (MDPs) are a formalization of sequential decision making where actions influence not only the immediate reward, but also future rewards. As such, this is an associative problem that takes into account the need to trade-off immediate and delayed reward. While in bandit problems we estimated the value $$q^{*}(a), \ \forall a \in \mathcal{A},$$ in an MDP, we estimate the value $$q^{*}(s, a), \ \forall a \in \mathcal{A}, \forall s \in \mathcal{S},$$ or the value $$v^{*}(s), \forall s \in \mathcal{S}$$ given optimal action selections. Such state-dependent values are important to assign credit for long-term rewards to individual action selections.

### Section 3.1: The Agent-Environment Interface
An MDP involves a learner and decision maker (i.e., the *agent*) that interacts with its surroundings (i.e., the *environment*) by continually selecting actions and having the environment respond by presenting new situations (or states) to the agent and giving rise to rewards, which the agent seeks to maximize over time. This process is illustrated in Figure 1.

<figure align='center'>
    <img alt="The agent-environment interaction in a Markov decision process." src="http://acfpeacekeeper.github.io/github-pages/images/literature/rl_mdp.png" onerror="this.src='http://localhost:4000/images/literature/rl_mdp.png';">
	<figcaption>Figure 1: The agent-environment interaction in a MDP.</figcaption>
</figure>

The agent and environment interact with each other in a sequence of discrete time steps $$t = 0, 1, \dots, n.$$ At each time step $$t$$, the agent receives a representation of the environment's *state* $$S_t \in \mathcal{S},$$ and on that basis selects an *action* $$A_t \in \mathcal{A}(s).$$ At the next time step $$t + 1$$, the agent receives a reward $$R_{t + 1} \in \mathcal{R} \subset \mathbb{R}$$ and finds itself in another state $$s_{t+1}.$$  Then, the MDP and agent give rise to a sequence or *trajectory* like this:

$$
\begin{equation}
S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \dots
\end{equation}
$$

In a *finite* MDP, the random variables $$R_t$$ and $$S_t$$ have well defined discrete probability distributions that depend only on the previous state and action, i.e., for particular values of these random variables $$s' \in \mathcal{A}, \ r \in \mathcal{R},$$ there is a probability of those values occurring at time step $$t,$$ given particular values of the previous state and action:

$$
\begin{equation}
p(s', r \vert s, a) \doteq P(S_t = s', R_t \vert S_{t-1} = s, A_{t-1} = a), \ \forall s', s \in \mathcal{S}, \forall r \in \mathcal{R}, \forall a \in \mathcal{A}(s).
\end{equation}
$$

The function $$p: \mathcal{S} \times \mathcal{R} \times \mathcal{S} \times \mathcal{A} \rightarrow [0,1]$$, which completely characterizes the MDP environment's *dynamics*, is an ordinary deterministic function with four arguments. This function $$p$$ specifies a probability distribution for each choice of $$s$$ and $$a$$, i.e., 

$$
\begin{equation}
\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r \vert s, a) = 1, \ \forall s \in \mathcal{S}, \forall a \in \mathcal{A}(s).
\end{equation}
$$

**Markov property**: the state must include information about all aspects of the past agent-environment interaction that make a difference for the future. In practice, this means that the probability of each possible value for $$S_t$$ and $$R_t$$ depends only on the previous state $$S_{t-1}$$ and action $$A_{t-1}$$.
	- While most methods in this book assume this property to be true, there are methods that don't rely on it. [Chapter 17](#chapter-17-frontiers) considers how to efficiently learn a Markov state from non-Markov observations.

From the dynamics function $$p,$$ we can compute the *state-transition probabilities* $$p: \mathcal{S} \times \mathcal{S} \times \mathcal{A} \rightarrow [0,1],$$

$$
\begin{equation}
p(s' \vert s, a) \doteq P(S_t = s' \vert S_{t-1} = s, A_{t-1} = a) = \sum_{r \in \mathcal{R}} p(s', r \vert s, a).
\end{equation}
$$

We can also compute the expected rewards for state-action pairs as a two-argument function $$r : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$$:

$$
\begin{equation}
r(s, a) \doteq \mathbb{E} [R_t \vert S_{t-1} = s, A_{t-1} = a] = \sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}} p(s', r \vert s, a),
\end{equation}
$$

and the expected rewards for the state-action-new_state triples as a three-argument function $$r: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R},$$ 

$$
\begin{equation}
r(s, a, s') \doteq \mathbb{E} [R_t \vert S_{t-1} = s, A_{t-1} = a, S_t = s'] = \sum_{r \in \mathcal{R}} r \frac{p(s', r \vert s, a)}{p(s' \vert s, a)}.
\end{equation}
$$

The MDP framework is flexible and can be applied to problems in many different ways, e.g., time steps don't need to refer to fixed intervals of real time; they can refer to arbitrary sequential stages of decision making and acting. In general, actions can be any decision we want to learn how to make, and states can be anything we can know that might useful in making them.

The boundary between agent and environment is usually drawn closer to the agent than the physical boundary of a robot or an animal's body, e.g., the motors and mechanical linkages of a robot and its sensing hardware should normally be considered parts of the environment, rather than of the agent. In a similar sense, rewards, which are usually computed inside the physical bodies of natural and artificial learning systems, are considered external to the agent. In general, anything that cannot be changed arbitrarily by the agent is considered to be outside of it and thus part of its environment.

We don't assume that everything in the environment is unknown to the agent, e.g., the agent often knows quite a bit about how its rewards are computed as a function of its actions and the states in which they are taken. However, the reward computation is considered to be external to the agent, because it defines the task facing the agent and must thus be beyond its ability to arbitrarily change.

The MDP framework proposes that any detail of whatever objective (of a problem of learning goal-directed behavior) one is trying to achieve can be reduced to three signal passing back and forth between an agent and its environment:
- Action: signal that represents the choices made by the agent;
- State: signal that represents the basis on which the choices are made;
- Reward: signal that defines the agent's goal.

### Section 3.2: Goals and Rewards

In RL, the purpose of the agent is formalized in terms of a simple number, the *reward* (at each time step $$R_t \in \mathbb{R}$$), which passes from the environment to the agent. The agent's goal is to maximize the total cumulative reward, something stated in the *reward hypothesis*:
```
That all of what we mean by goals and purposes can be well thought of as
the maximization of the expected value of the cumulative sum of a received
scalar signal (called reward).
```

In order for the agent to achieve our goals, it is critical that the reward signals defined truly indicate what we want accomplished. Of particular importance, one must not use the reward signal to impart prior knowledge to the agent. Using chess as an example, we should naturally define the reward as $$+1$$ for winning, $$-1$$ for losing and $$0$$ for draws and all non-terminal positions. We should **NOT** give rewards for sub-goals like taking an opponent's chess piece, otherwise the agent might find a way to maximize its reward, even at the cost of losing the game. The reward signal defines **what** you want the agent to achieve, not *how* you want the agent to achieve it.

### Section 3.3: Returns and Episodes

We seek to maximize the *expected return*, where the return $$G_t$$ is defined as some specific function of the reward sequence. In the simplest case the return is the sum of the rewards:

$$
\begin{equation}
G_t \doteq R_{t+1} + R_{t+2} + R_{t+3} + \dots + R_T,
\end{equation}
$$

where $$T$$ is a final time step. This approach makes sense in applications where the agent-environment interaction can be naturally broken down into sub-sequences, called **episodes,** like the plays of a game or trips to a maze. Each episode ends in a special state, called the *terminal* state, and the resets to a starting state or a sample from standard distribution of starting states. Since the next episode begins independently of how the previous one ended, episodes can all be considered to end in the same terminal state, just with different rewards for different outcomes.

**Episodic task**: tasks that have discrete independent episodes, i.e., where the outcome of the ending of an episode doesn't effect the start state of the next episode.
- We sometimes distinguish the set of all non-terminal states $$\mathcal{S}$$ from the set of all states plus the terminal state $$\mathcal{S}^+$$;
- The time of termination $$T$$ is a random variable that can vary from episode to episode.

**Continuing task**: tasks where the agent-environment interaction cannot be naturally broken down into identifiable episodes, and goes on continually without limit.
- With the current formulation, the final time step for these tasks would be $$T = \infty$$ and thus, the total return we are trying to maximize could easily be infinite;
- To deal with this, we add the concept of *discounting* to the formulation. Now, the agent must try to select actions that maximize the sum of discounted rewards received.

Following this formulation, the agent must then chose $$A_t$$ s.t. the expected *discounted return* is maximal:

$$
\begin{equation}
G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k = 0}^{\infty} \gamma^k \cdot R_{t+k+1},
\end{equation}
$$

where $$\gamma \in [0, 1]$$ is a parameter called the *discount rate*. This parameter determines the current value of future rewards, i.e., a reward received $$k$$ time steps in future is only worth $$\gamma^{k+1}$$ times what it would be worth if it were received now. If $$\gamma < 1$$ and the reward sequence $$\{R_k\}$$ is bounded, then the infinite sum of discounted rewards has a finite value.  If $$\gamma = 0,$$ the agent is only concerned with maximizing its immediate reward, i.e., its objective is to learn how to select $$A_t$$ s.t. it maximizes only $$R_{t+1}.$$ As $$\gamma \rightarrow 1,$$ the return objective takes future rewards more strongly into account, i.e., the agent becomes more farsighted. The returns at successive time steps are related to each other s.t.

$$
\begin{align}
G_t &\doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \dots \nonumber\\
	&= R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \dots) \nonumber\\
	&= R_{t+1} + \gamma G_{t+1}.
\end{align}
$$

This works for all time steps $$t < T,$$ even if the termination occurs at $$t + 1,$$ provided we define $$G_T = 0$$. 
If the reward is a constant $$+1,$$ then the return is

$$
\begin{equation}
G_t = \sum_{k = 0}^{\infty} \gamma^k = \frac{1}{1 - \gamma}.
\end{equation}
$$

### Section 3.4: Unified Notation for Episodic and Continuing Tasks

To be precise about episodic tasks, instead of considering one long sequence of time steps, we need to consider a series of episodes, where each episode consists of a finite sequence of time steps. As such, we refer to $$S_{t, i}$$ as the state representation at time step $$t$$ of episode $$i$$ (the same for $$A_{t, i}, R_{t, i}, \pi_{t, i}, T_i, \dots$$). In practice however, since we are almost always considering a particular episode or stating a fact that is true for all episodes, we can drop the explicit reference to the episode number.

We can unify the finite sum of terms for the total return in the episodic case and the infinite sum for the total reward in the continuing case by considering episode termination to be the entering of a special *absorbing state* that transitions only to itself and generates only rewards of zero, as exemplified in Figure 2.

<figure align='center'>
<img alt="Example of an MDP with a absorbing state." src="http://acfpeacekeeper.github.io/github-pages/images/literature/absorbing_state.png" onerror="this.src='http://localhost:4000/images/literature/absorbing_state.png';">

<figcaption>Figure 2: Example of an MDP with an absorbing state.</figcaption>
</figure>

Including the possibility that $$T = \infty \oplus \gamma = 1$$ (where $$\oplus$$ is the XOR operator), we can write

$$
\begin{equation}
G_t \doteq \sum_{k = t + 1}^T \gamma^{k - t - 1} R_k.
\end{equation}
$$

### Section 3.5: Policies and Value Functions

**Value function**: a function of states (or state-action pairs) that estimates how good it it is for the agent to be in a given state, defined in terms of future rewards that can be expected.
- Since the rewards the agent can expect to receive depend on what actions it takes, value functions are defined w.r.t. a particular way of acting, called a policy.

**Policy**: a mapping from states to probabilities of selecting each possible action, i.e., if the agent is following policy $$\pi$$ at time $$t,$$ then $$\pi(a \vert s)$$ is the probability that $$A_t = a$$ if $$S_t = s.$$ 
- It is a function that defines a probability distribution over $$a \in \mathcal{A}(s)$$ for each $$s \in \mathcal{S}.$$ 

The *value function* of a state $$s$$ under a policy $$\pi,$$ denoted $$^{\pi}(s),$$ is the expected return when starting in $$s$$ and following $$\pi$$ thereafter. For MDPs, $$v^{\pi}$$ can be formally defined as

$$
\begin{equation}
v^{\pi}(s) \doteq \mathbb{E}_{\pi} [G_t \vert S_t = s] = \mathbb{E}_{\pi} \bigg[ \sum_{k=0}^{\infty} \gamma^k R_{t + k + 1} \vert S_t = s \bigg], \ \forall s \in \mathcal{S},
\end{equation}
$$

where $$\mathbb{E}_{\pi}[\cdot]$$ denotes the expected value of random variable given that the agent follows policy $$\pi,$$ and $$t$$ is any time step (the value of the terminal state is always zero). The function $$v^{\pi}$$ is called the *state-value function for policy $$\pi$$.* 

We can also define the value of taking action $$a$$ in state $$s$$ while following a policy $$\pi,$$ denoted $$q^{\pi}(s, a),$$ as the expected return starting from $$s,$$ taking the action $$a,$$ and following policy $$\pi$$ afterwards:

$$
\begin{equation}
q^{\pi}(s, a) \doteq \mathbb{E}_{\pi} [G_t \vert S_t = s, A_t = a] = \mathbb{E}_{\pi} \bigg[\sum_{k=0}^{\infty} \gamma^k R_{t + k + 1} \vert S_t = s, A_t =a \bigg].
\end{equation}
$$

The function $$q^{\pi}$$ is called the *action-value function for policy $$\pi$$*. Both $$q^{\pi}$$ and $$v^{\pi}$$ can be estimated from experience, e.g., using *Monte Carlo methods*.

**Monte Carlo methods**: estimation methods that involve averaging over many random samples of a random variable.
- For examples, if an agent follows policy $$\pi$$ and maintains an average, for each state encountered, of the actual returns that have followed that state, then the average will converge to the state's actual value $$v^{\pi}(s),$$ as the number of times that state is encountered approaches infinity.  If separate averages are kept for each action taken in each state, these averages will also converge to the action values $$q^{\pi}(s, a)$$;
- Since keeping an average for each state and state-action pair is usually not practical, the agent can instead maintain $$v^{\pi}$$ and $$q^{\pi}$$ as parameterized functions.

A fundamental property of commonly used value functions is that they satisfy recursive relationships similar to that of the return. For any policy $$\pi$$ and state $$s,$$ the following consistency condition holds between the value of $$s$$ and the value of its possible successor states:

$$
\begin{align}
v^{\pi}(s) &\doteq \mathbb{E}_{\pi} [G_t \vert S_t = s] \nonumber\\
	&= \mathbb{E}_{\pi} [R_{t+1} + \gamma G_{t+1} \vert S_t = s] \nonumber\\
	&= \sum_a \pi(a \vert s) \sum_{s'} \sum_r p(s', r \vert s, a) \bigg[r + \gamma \mathbb{E}_{\pi} [G_{t+1} \vert S_{t+1} = s']\bigg] \nonumber\\
	&= \sum_a \pi(a \vert s) \sum_{s', r} p(s', r \vert s, a) [r + \gamma v^{\pi}(s')], \quad \forall s \in \mathcal{S},
\end{align}
$$

where it is implicit that the actions $$a$$ are taken from the set $$\mathcal{A},$$ that the next state $$s'$$ are taken from the set $$\mathcal{S}$$ (or from $$\mathcal{S}^+$$ in the case of an episodic problem), and that the rewards $$r$$ are taken from the set $$\mathcal{R}.$$ The final expression, which is a sum over all values of the three variables $$a$$, $$s'$$, and $$r$$, can be read as an expected value. For each triple, we compute its probability $$\pi(a \vert s) p(s', r \vert s, a),$$ weight the quantity in brackets by that probability and then sum over all possibilities to get an expected value.

The last equation, called the *Bellman equation for $$v^{\pi}$$,* expresses a relationship between the value of a state and the values of its successor states (similar to a look-ahead).  The Bellman equation averages over all the possibilities, weighting each by its probability of occurring, and it states that the value of the start state must equal the discounted value of the expected next state, plus the reward expected along the way. The value function $$v^{\pi}$$ is the unique solution to this equation.

### Section 3.6: Optimal Policies and Optimal Value Functions

Solving a RL task roughly means finding a policy that maximizes the total reward over the long run. For finite MDPs, we can precisely define an *optimal policy $$\pi^*$$* as follows:
- Value function define a partial ordering over policies;
- A policy $$\pi$$ is defined to be better than or equal to a policy $$\pi'$$, i.e., $$\pi \geq \pi'$$, iff $$v^{\pi}(s) \geq v_{\pi'}(s), \ \forall s \in \mathcal{S}$$;
- $$\exists \pi^*: \pi^* \geq \pi, \ \forall \pi \in \mathcal{S}$$.

Although there may be more than one optimal policy, we denote them all by $$\pi^*$$. They share the same state-value function, called the *optimal state-value function*, denoted $$v^*$$, and defined as

$$
\begin{equation}
v^*(s) \doteq \max_{\pi} v^{\pi}(s), \quad \forall s \in \mathcal{S}.
\end{equation}
$$

Optimal policies also share the same *optimal action-value function*, denoted $$q^*$$. For the state-action pair $$(s, a)$$, this function gives the expected return for taking action $$a$$ in state $$s$$ and following an optimal policy afterwards. This function can defined as

$$
\begin{align}
q^*(s, a) &\doteq \max_{\pi} q^{\pi}(s, a), \quad \forall s \in \mathcal{S}, \forall a \in \mathcal{A}(s),\\
	&= \mathbb{E}[R_{t+1} + \gamma v^*(S_{t+1}) \vert S_t = s, A_t = a].
\end{align}
$$

### Section 3.7: Optimality and Approximation

Even if we have a complete and accurate model of the environment's dynamics, it is usually not possible to compute an optimal policy by solving the Bellman optimality equation. An important aspect of the problem is the amount of computational power available to the agent, in particular, the amount of computation it can perform in a single time step. Another important constraint is the memory available, specially since a large amount of memory is often required to build up approximations of value functions, policies, and models.

*tabular case*: task with small, finite state sets, which allow one to form approximations using arrays or tables with one entry for each state (or state-action pair), i.e., tabular methods. 
- In tasks that have state sets too large to use tabular methods, we must instead rely on approximation using compact parameterized function representations.

The online nature of RL makes it possible to approximate optimal policies in ways that put more effort into learning to make good decisions for frequently encountered states, at the expense of less effort for infrequently encountered states.

### Section 3.8: Summary

RL is about an *agent* learning how to behave in order to achieve a goal by interacting with its *environment* over a sequence of discrete time steps. The specification of their interface defines a particular task:
- *Actions*: choices made by the agent;
- *States*: the basis for making the choices;
- *Rewards*: the basis for evaluating the choices.

A *policy* is a stochastic rule by which the agent selects the actions as function of states. The *optimal policy* is that which best achieves the agent's objective, maximizing the amount of reward it receives over time.

The *return* is the function of future rewards that the agent seeks to maximize (in expected value). Its definition depends on the nature of the task and whether one wishes to *discount* delayed reward. The undiscounted formulation is appropriate for *episodic tasks*, and the discounted formulation is appropriate for tabular *continuing tasks*.

A policy's *value functions* ($$v^*$$ and $$q^*$$) assign to each state, or state–action pair, the largest expected return achievable by any policy. Whereas the optimal value functions are unique for a given MDP, there can be many optimal policies. In most cases, we seek approximations of the optimal value functions and policies, not their exact values.

## Chapter 4: Dynamic Programming

In this (and future) chapters, we will usually assume that the environment is a finite MDP. This means that we assume $$|\mathcal{S}| < \infty, |\mathcal{A}| < \infty, |\mathcal{R}| < \infty,$$ and that its dynamics are given by a set of probabilities $$p(s', r \vert s, a), \forall s \in \mathcal{S}, \forall a \in \mathcal{A}, \forall r \in \mathcal{R}, s' \in \mathcal{S}^+$$. 

Beyond finite MDPs, Dynamic Programming (DP) ideas can be applied to problems with a continuous state and action spaces, by exact solutions for these types of problems are only possible in special cases.

The key idea of DP (and RL in general) is the use of value functions to both structure and organize the search for good policies. DP algorithms are obtained by turning Bellman equations into assignments, i.e., into update rules for improving approximations of the desired value functions. Remember that we can easily obtain optimal policies if we find the optimal value functions, $$v^* \lor q^*$$, which satisfy the Bellman optimality equations, $$\forall s \in \mathcal{S}, \forall a \in \mathcal{A}(s), s' \in \mathcal{S}^+$$:

$$
\begin{align}
v^*(s) &= \max_a \mathbb{E}[R_{t+1} + \gamma \cdot v^*(S_{t+1} \vert S_t = s, A_t = a)] \nonumber\\
	&= \max_a \sum_{s', r} p(s' r, \vert s, a) [r + \gamma \cdot v^*(s')],\\
	&\lor \nonumber\\
q^*(s, a) &= \mathbb{E}[R_{t+1} + \gamma \max_{'a} q^*(S_{t+1}, a') \vert S_t = s, S_t =a], \nonumber\\
	&= \sum_{s', r} p(s', r \vert s, a) [r + \gamma \max_{a'} q^*(s', a')].
\end{align}
$$

### Section 4.1: Policy Evaluation (Prediction)

**Policy evaluation:** computing the state-value function $$v^{\pi}$$ for an arbitrary policy $$\pi$$. This is also referred to as the *prediction problem*. Recall from [Chapter 3](#chapter-3-finite-markov-decision-processes) that, $$\forall s \in \mathcal{S}$$,

$$
\begin{align}
v^{\pi}(s) &\doteq \mathbb{E}_{\pi}[G_t \vert S_t = s], \nonumber\\
	&= \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} \vert S_t = s], \nonumber\\
	&= \mathbb{E}_{\pi}[R_{t+1} + \gamma v^{\pi} (S_{t+1} \vert S_t = s)], \\
	&= \sum_a \pi(a \vert s) \sum_{s', r} p(s', r \vert s, a)[r + \gamma v^{\pi}(s')],
\end{align}
$$

where $$\pi(a\vert s)$$ is the probability of taking action $$a$$ in state $$s$$ under policy $$\pi$$, and the expectations are subscripted by $$\pi$$ to indicate that they are conditional on following policy $$\pi$$. If $$\gamma < 1$$ or eventual termination is guaranteed from all states under the policy $$\pi$$, then w.h.t. that $$v^{\pi}$$ exists and is unique.

If the environment's dynamics are completely known, then the previous equation is a system of $$|\mathcal{S}|$$ simultaneous linear equations in $$|\mathcal{S}|$$ unknowns (the $$v^{\pi}(s), s \in \mathcal{S}$$). Consider a sequence of approximate value functions $$v_0, v_1, \dots,$$ each mapping $$S^+ \rightarrow \mathbb{R}$$. The initial approximation $$v_0$$ is chosen arbitrarily (except the terminal state, which, if it exists, must have value $0$), and, $$\forall s \in \mathcal{S},$$ each successive approximation is obtained by using the Bellman equation for $$v^{\pi}$$ as an update rule:

$$
\begin{align}
v_{k+1}(s) &\doteq \mathbb{E}_{\pi}[R_{t+1} + \gamma \cdot v_k (S_{t+1}) \vert S_t = s] \nonumber\\
	&= \sum_a \pi(a \vert s)\sum_{s', r} p(s', r \vert s,a )[r + \gamma \cdot v_k(s')].
\end{align}
$$

Since the Bellman equation for $$v^{\pi}$$ assures us of the equality in the case that $$v_k = v^{\pi},$$ this is a fixed point for this update rule. Then, when under the same conditions that guarantee the existence of $$v^{\pi},$$ w.h.t. $$\lim_{k \rightarrow \infty} \{v_k\} = v^{\pi}$$. This algorithm, called **iterative policy evaluation**, produces each successive approximation $$v_{k+1}$$ from $$v_k$$, which we call an *expected update* (since they are based on an expectation over all possible states), by applying the same operation to each state $$s$$: replacing the old value of $$s$$ with a new value obtained from the old values of the successor states of $$s$$, and the expected immediate rewards, along all the one-step transitions possible under the policy being evaluated.

```
def get_new_value(policy, gamma, mdp, cur_state, state_values):
	new_val = 0;
	for a in mdp.actions:
		tmp = 0;
		take_a_in_s_prob = policy[cur_state][a];
		for state in mdp.states:
			tmp += mdp.probabilities[state][state.reward] * (state.reward + gamma * state_values[state.id]);
		
		new_val += take_a_in_s_prob * tmp;

	return new_value;
		

def iterative_policy_evaluation(policy, theta, gamma, mdp):
	assert theta > 0;
	
	state_values[len(mdp.states)];
	state_values[len(mdp.states)] = 0;
	for x in range(len(mdp.states) - 1):
		state_values[x] = random_value();
		
	do {
		gradient = 0;
		for s in mdp.states:
			v = state_values[s.id];
			state_values[s.id] = get_new_value(policy, gamma, s, state_values)
			gradient = max(gradient, absolute_value(v - state_values[s.id]));
	} while(gradient > theta);

	return state_values;
```

### Section 4.2: Policy Improvement

Knowing the value function $$v^{\pi}$$ for an arbitrary deterministic policy $$\pi$$, and given some state $$s$$, how can we determine whether or not we should change the policy to deterministically choose an action $$a \neq \pi(s)$$? One possible way, is to consider selecting $a$ in $s$ and afterwards following the existing policy $$\pi$$. The value of this manner of behaving is given by

$$
\begin{align}
q^{\pi}(s, a) &\doteq \mathbb{E}[R_{t+1} + \gamma v^{\pi}(S_{t+1} \vert S_t = s, A_t = a)] \nonumber\\
	&= \sum_{s', r} p(s', r \vert s, a)[r + \gamma \cdot v^{\pi}(s')].
\end{align}
$$

The key criterion is whether this is greater than or less than $$v^{\pi}$$. If it is greater, i.e., if it is better to select $$a$$ once in $$s$$ and thereafter follow $$\pi$$ than it is to always follow $$\pi$$, then one can expect that it would be better to select $$a$$ ever time $$s$$ is encountered, and that such new policy would be a better one overall. This fact is a special case of a general result called the *policy improvement theorem*. 

Let $$\pi$$ and $$\pi'$$ be any pair of deterministic policies s.t.:

$$
\begin{equation}
q^{\pi}(s, \pi'(s)) \geq v^{\pi}(s), \forall s \in \mathcal{S}.
\end{equation}
$$

Then the policy $$\pi'$$ must be as good as, or better than, $$\pi$$, i.e., it must obtain greater or equal expected return from all states:

$$
\begin{equation}
v^{\pi'}(s) \geq v^{\pi}(s), s \in \mathcal{S}.
\end{equation}
$$

Additionally, if there is strict inequality of $$q^{\pi}(s, \pi'(s))$$ at any state, then there must be strict inequality of $$v^{\pi'}(s)$$ at that state. This theorem applies to the two policies considered: a deterministic policy $$\pi$$, and a changed policy $$\pi$$, identical to $$\pi$$ in everything except $$\pi'(s) = a \neq \pi(s)$$. Since $$v^{\pi'} = v^{\pi}, \forall s' \in \mathcal{S} \setminus \{s\},$$ if $$q^{\pi}(s, a) > v^{\pi}(s)$$, then the changed policy $$\pi'$$ must be better than $$\pi$$. 

In order to understand the idea of the proof behind the policy improvement theorem, we can start from $$q^{\pi}(s, \pi'(s)) \geq v^{\pi}(s)$$, then keep expanding the l.h.s. with the equation for $$q^{\pi}(s, a)$$ and reapplying $$q^{\pi}(s, \pi'(s)) \geq v^{\pi}(s)$$:

$$
\begin{align}
v^{\pi}(s) &\leq q^{\pi}(s, \pi'(s))\\
	&= \mathbb{E}[R_{t+1} + \gamma \cdot v^{\pi}(S_{t=1}) \vert S_t = s, A_t = \pi'(s)] \nonumber\\
	&= \mathbb{E}_{\pi'}[R_{t+1} + \gamma \cdot v^{\pi}(S_{t+1}) \vert S_t = s] \nonumber\\
	&\leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma \cdot q^{\pi}(S_{t+1}, \pi'(S_{t+1})) \vert S_t = s] \nonumber\\
	&= \mathbb{E}_{\pi'}\bigg[R_{t+1} + \gamma \cdot \mathbb{E}[R_{t+2} + \gamma \cdot v^{\pi}(S_{t+2}) \vert S_{t+1}, A_{t+1} = \pi'(S_{t+1})] \vert S_t = s \bigg] \nonumber\\
	&\leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} \gamma^3 R_{t+4} + \dots \vert S_t = s] \nonumber\\
	&\dots \nonumber\\
	&= v^{\pi'}(s).
\end{align}
$$

Now knowing how, given a policy policy and its value function, to evaluate a change in the policy at a single, we to extend this result to consider changes at all states - selecting at each state the action that appears best according to $$q^{\pi}(s, a)$$, i.e., to consider the new *greedy* policy $$\pi'$$, given by

$$
\begin{align}
\pi'(s) &\doteq \argmax_a \  q^{\pi}(s, a) \nonumber\\
	&= \argmax_a \ \mathbb{E}[R_{t+1} + \gamma \cdot v^{\pi}(S_{t+1}) \vert S_t = s, A_t = a] \nonumber\\
	&= \argmax_a \sum_{s', r} p(s', r \vert s, a)[r + \gamma \cdot v^{\pi}(s')].
\end{align}
$$

The greedy policy takes the action that looks best according to $$v^{\pi}$$ after a single step of lookahead. Since, by construction, the greedy policy meets the conditions of the policy improvement theorem, we know that it is as good as - or better than - the original policy.

**Policy improvement**:  process of making a new policy that improves on an original policy by making it greedy w.r.t. the value function of the original policy.

Supposing that the new greedy policy $$\pi'$$ is as good as, but not better than, the old policy $$\pi$$, then $$v^{\pi}(s) = v^{\pi'}$$, and, $$\forall s \in \mathcal{S}$$, it follows from the previous equation that

$$
\begin{align}
v^{\pi'}(s) &= \max_a \mathbb{E}[R_{t+1} + \gamma \cdot v^{\pi'}(S_{t+1} \vert S_t = s, A_t = a)] \nonumber\\
	&= \max_a \sum_{s', r} p(s', r \vert s, a)[r + \gamma \cdot v^{\pi'}(s')].
\end{align}
$$

As this is the same as the Bellman optimality equation, $$v^{\pi'}$$ must be $$v^*$$, and both $$\pi$$ and $$\pi'$$ must be optimal policies. Thus, policy improvement must give us a strictly better policy except when the original policy is already optimal.

The policy improvement theorem carries through as stated for the stochastic case. Also, if there are any ties in policy improvement steps - i.e., there are several actions at which the maximum is achieved - then we don't need to select a single action from among them, as each maximizing action can be given a portion of the probability of being selected in the new greedy policy (as long as we give zero probability to all submaximal actions).

### Section 4.3: Policy Iteration

## Chapter 5: Monte Carlo Methods

## Chapter 6: Temporal-Difference Learning

## Chapter 7: $$n$$-step Bootstrapping

## Chapter 8: Planning and Learning with Tabular Methods

# Part II: Approximate Solution Methods

## Chapter 9: On-policy Prediction with Approximation

## Chapter 10: On-policy Control with Approximation

## Chapter 11: \*Off-policy Methods with Approximation

## Chapter 12: Eligibility Traces

## Chapter 13: Policy Gradient Methods

**Notation** relevant for this chapter:
- Policy's parameter vector: $$\theta \in \mathbb{R}^{d'}$$;
- Learned value function's weight vector: $$w \in \mathbb{R}^d$$;
- Scalar performance measure w.r.t. the policy parameter: $$J(\theta)$$;
- Probability that action $$a$$ is taken at time $$t$$, given that the environment is in state $$s$$ at time $$t$$: $$
\pi (a \vert s, \theta) = P (A_t = a|S_t = s,\theta_t = \theta)
$$.

**Policy gradient** methods seek to learn an approximation to the policy by maximizing performance. Their updates approximate gradient ascent such as:

$$
\begin{equation}
\theta_{t+1} = \theta_t + \alpha \cdot \hat{\nabla J(\theta_t)}, \ \hat{\nabla J(\theta_t)} \in \mathbb{R}^{d'}.
\end{equation}
$$

Methods that learn an approximation of both policy and value functions are called **actor-critic** methods. The *actor* is a reference to the learned policy and *critic* a reference to the learned (state-)value function. 

### Section 13.1: Policy Approximation and its Advantages

The policy can be parameterized in any way, as long as 2 conditions are met

- As long as $$\pi(a\vert s, \theta)$$ is differentiable w.r.t. its parameters:

$$
\forall s \in S \ \forall a \in A(s) \ \exists \ \nabla \pi (a|s, \theta) : |\nabla \pi (a|s, \theta)| < \infty \ \wedge \theta \in \mathbb{R}^{d'};
$$

- As long as it continues to perform exploration (to avoid a deterministic policy): 

$$
\pi (a|s, \theta) \ \in \  ]0, 1[.
$$

For small-to-medium discrete action spaces, it is common to form parameterized numerical preferences $$h(s, a, \theta) \in \mathbb{R}$$ for each $$(s, a)$$ pair. The probabilities of each action being selected can be calculated with, e.g., an exponential softmax distribution:

$$
\begin{equation}
\pi (a|s, \theta) \doteq \frac{\exp(h(s,a,\theta))}{\sum_b \exp(h(s,b,\theta))}.
\end{equation}
$$

This kind of policy parameterization is called *softmax in action preferences*.

The action preferences can be parameterized arbitrarily, e.g., as the output of a Deep Neural Network (DNN) with parameters $$\theta$$. They can also be linear in features:

$$
\begin{equation}
h(s,a,\theta) = \theta^T \cdot x(s, a), \ x(s, a) \in \mathbb{R}^{d'}.
\end{equation}
$$

The *choice of policy parameterization* can be a good way of *injecting prior knowledge* about the desired form of the policy into the RL system. Beyond this, parameterizing policies according to the softmax in action preferences has several advantages:
- Allows the approximate policy to approach a deterministic policy, unlike with $$\epsilon$$-greedy action selection (due to the $$\epsilon$$ probability of selecting a random action);
- Enables the selection of actions with arbitrary probabilities (useful if the optimal policy is a stochastic policy, e.g., in a game of chance such as poker);
- If the policy is easier to approximate then the action-value function, then policy-based methods learn faster and yield a superior asymptotic policy.

### Section 13.2: The Policy Gradient Theorem

Policy parameterization has an important theoretical advantage over $$\epsilon$$-greedy action selection: the continuity of the policy dependence on the parameters that enables policy gradient methods to approximate gradient ascent. 

Due to the continuous policy parameterization the action probabilities change smoothly as a function of the learned parameters, unlike with $$\epsilon$$-greedy selection, where the action probabilities may change dramatically if an update changes which action has the maximal value.

In the episodic case, the performance measure is defined as the value of the start state of the episode. Taking a (non-random) state $$s_0$$ as the start state of the episode and $$v_{\pi_{\theta}}$$ as the true value function for the policy $$\pi_{\theta}$$, the performance can be defined as:

$$
\begin{equation}
J(\theta) \doteq v_{\pi_{\theta}} (s_0).
\end{equation}
$$

Since the policy parameters affect both the action selections and the distribution of states in which those selections are made, and performance also depends on both, it may be challenging to to change the policy parameters such that improvement is ensured, particularly since the effect of the policy on the state distribution is a function of the environment, and thus is typically unknown.

The **policy gradient theorem** provides a solution towards this challenge in the manner of an analytic expression for the gradient of performance w.r.t. the policy parameters, without the need for the derivative of the state distribution. The equation for episodic case of the policy gradient theorem is given by:

$$
\begin{equation}
\nabla J(\theta) \propto \sum_s \mu (s) \sum_a q^{\pi}(s, a) \cdot \nabla \pi(a|s, \theta),
\end{equation}
$$

where $$\mu$$ is the on-policy distribution under $$\pi$$. For the episodic case, the constant of proportionality is the average length of an episode, and for the continuing case it is 1.

### Section 13.3: REINFORCE: Monte Carlo Policy Gradient

Since any constant of proportionality can be absorbed into the step size $$\alpha$$, all that is required is a way of sampling that approximates the policy gradient theorem. As the r.h.s. of the theorem is a sum over states weighted by their probability of occurring under the target policy $$\pi$$, w.h.t.:
$$
\begin{align}
	\nabla J(\theta) &\propto \sum_s \sum_a q^{\pi}(s, a) \cdot \nabla \pi (a|s, \theta) \nonumber\\
	&= \mathbb{E}_{\pi} [\sum_a q^{\pi}(S_t, a) \cdot \nabla \pi(a|S_t, \theta)].
\end{align}
$$
Thus, we can instantiate the stochastic gradient ascent algorithm (known as the *all-actions* method, due to its update involving all of the actions) as:

$$
\begin{equation}
\theta_{t + 1} \doteq \theta_t + \alpha \sum_a \hat{q}(S_t, a, w) \cdot \nabla \pi (a|S_t, \theta),
\end{equation}
$$

where $$\hat{q}$$ is a learned approximation of $$q^{\pi}$$.
Unlike the previous method, the update step at time $$t$$ of the **REINFORCE** algorithm involves only $$A_t$$ (the action taken at time $$t$$). By multiplying and then dividing the summed terms by $$\pi (a\vert S_t, \theta)$$, we can introduce the weighting needed for an expectation under $$\pi$$., then, given that $$G_t$$ is the return, w.h.t.:

$$
\begin{align}
	\nabla J(\theta) &\propto \mathbb{E}_{\pi} [\sum_a \pi(a|S_t, \theta) \cdot q^{\pi} (S_t,a) \cdot \frac{\nabla \pi(s|S_t, \theta)}{\pi(a|S_t, \theta)}] \nonumber\\
	&= \mathbb{E}_{\pi} [q^{\pi}(S_t, A_t) \cdot \frac{\nabla \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)}] \nonumber\\
	&= \mathbb{E}_{\pi} [G_t \cdot \frac{\nabla \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)}].
\end{align}
$$

Since the final (in brackets) expression is a quantity that can be sampled on each time step with expectation proportional to the gradient, such a sample can be used to instantiate the generic stochastic gradient ascent algorithm, yielding the REINFORCE update:

$$
\begin{align}
	\theta_{t + 1} &\doteq \theta_t + \alpha \cdot \gamma^t \cdot G_t \cdot \frac{\nabla \pi(A_t|S_t, \theta_t)}{\pi(A_t|S_t, \theta_t)} \nonumber\\
	&= \theta_t + \alpha \cdot \gamma^t \cdot G_t \cdot \nabla \ln \pi(A_t| S_t, \theta_t), 
\end{align}
$$\\
where $$\gamma$$ is the discount factor and $$\ln \pi(A_t|S_t, \theta_t)$$ is the *eligibility* vector. In this update, each increment is proportional to the return - causing the parameters to move most in the directions of the actions that yield the highest return - and is inversely proportional to the action probability - since the most frequent selected actions would have an advantage otherwise. 

Since REINFORCE uses the complete return from time $$t$$ (including all future rewards until the end of an episode), it is considered a Monte Carlo algorithm and is only well defined in the episodic case with all updates made in retrospect after the episode's completion.

As a stochastic gradient method, REINFORCE assures an improvement in the expected performance (given a small enough $$\alpha$$) and convergence to a local optimum (under standard stochastic approximation conditions for decreasing $$\alpha$$). However, as a Monte Carlo method, REINFORCE may have high variance and subsequently produce slow learning.

For some examples of Python implementations of the REINFORCE algorithm, <a href="https://acfpeacekeeper.github.io/github-pages/ml/rl/dl/2024/03/28/REINFORCE.html" onerror="this.href='http://localhost:4000/ml/rl/dl/2024/03/28/REINFORCE.html'">see here</a>.

### Section 13.4: REINFORCE with Baseline

Generalizing the policy gradient theorem to include a comparison of the action value to an arbitrary baseline $$b(s)$$ gives the following expression:

$$
\begin{equation}
\nabla J(\theta) = \sum_s \mu (s) \sum_a (q^{\pi}(s, a) - b(s)) \cdot \nabla \pi (a|s, \theta).
\end{equation}
$$

The baseline can be any arbitrary function (as long as it doesn't vary with $$a$$), since the equation will remain valid as the subtracted quantity is zero:

$$
\begin{equation}
\sum_a b(s)\cdot \nabla \pi(a|s, \theta) = b(s) \cdot \nabla \sum_a \pi(a|s, \theta) = b(s) \cdot \nabla 1 = 0.
\end{equation}
$$

The theorem can then be used to derive an update rule similar to the previous version of REINFORCE, but which includes a general baseline:

$$
\begin{equation}
\theta_{t + 1} \doteq \theta_t + \alpha \cdot \gamma^t \cdot (G_t - b(S_t)) \cdot \frac{\nabla \pi(A_t|S_t, \theta_t)}{\pi(A_t|S_t, \theta_t)}.
\end{equation}
$$

While the baseline generally leaves the expected value of the update unchanged, it can have a significant effect on its variance and thus the learning speed. 

The value of the baseline should follow that of the actions, i.e., if all actions have high/low values, then the baseline must also have high/low value, so as to differentiate the higher valued actions from the lower valued ones. A common choice for the baseline is an estimate of state value $$\hat{v}(S_t, w)$$, with $$w \in \mathbb{R}^d$$ being a learned weight vector.

The algorithm has two step sizes, $$\alpha_{\theta}$$ (which is the same as the step size $$\alpha$$ in previous equations) and $$\alpha_{w}$$. A good rule of thumb for setting the step size for values in the linear case is 

$$
\begin{equation}
\alpha_w = 0.1/\mathbb{E}[||\nabla \hat{v}(S_t, w)]||_{\mu}^2.
\end{equation}
$$

The step size for the policy parameters $$\alpha_{\theta}$$ will depend on the range of variation of the rewards and on the policy parameterization.

You can find an example of a Python implementation of REINFORCE with baseline <a href="https://acfpeacekeeper.github.io/github-pages/ml/rl/dl/2024/03/28/REINFORCE.html#reinforce-with-baseline" onerror="this.href='http://localhost:4000/ml/rl/dl/2024/03/28/REINFORCE.html#reinforce-with-baseline'">here</a>.

### Section 13.5: Actor-Critic Methods

In actor-critic methods, the state-value function is applied to the second state of the transition, unlike in REINFORCE, where the learned state-value function only estimates the value of the first state of each state transition and thus can´t be used to assess that action. After discount and adding the estimated value of the second state to the reward, it constitutes the 1-step return $$G_{t:t+1}$$, which can be used to assess the action.

Even though the 1-step return introduces bias, it is often superior to the actual return in terms of its variance and computational congeniality. The bias can also be flexibly modulated through $$n$$-step returns and eligibility traces.

1-step actor-critic methods are analogs of the TD methods such as TD(0), Sarsa(0) and Q-learning. Such methods are appealing since they function in fully online and incremental manner, while avoiding the complexities of eligibility traces. In these methods, the full return of REINFORCE is replaced with the 1-step return (with a state-value function as the baseline) as follows:

$$
\begin{align}
	\theta_{t+1} &\doteq \theta_t + \alpha (G_{t:t+1} - \hat{v}(S_t, w)) \frac{\nabla \pi (A_t|S_t, \theta_t)}{\pi(A_t|S_t, \theta_t)} \nonumber\\
	&= \theta_t + \alpha (R_{t+1} + \gamma \cdot \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)) \frac{\nabla \pi (A_t|S_t, \theta_t)}{\pi(A_t|S_t, \theta_t)} \nonumber\\
	&= \theta_t + \alpha \cdot \delta_t \cdot \frac{\nabla \pi (A_t|S_t, \theta_t)}{\pi(A_t|S_t, \theta_t)}.
\end{align}
$$

A usual state value function learning method to pair with this is semi-gradient TD(0). 
To generalize to the forward view of $$n$$-steps methods and to a $$\lambda$$-return, one only needs to replace the on-step return in the previous equation by $$G_{t:t+1}$$ or $$G_t^{\lambda}$$, respectively. The backward view of the $$\lambda$$-return algorithm is also simple, only requiring using separate eligibility traces for the actor and critic.

For some examples of Python implementations of actor-critic methods, <a href="https://acfpeacekeeper.github.io/github-pages/ml/rl/dl/2024/03/29/Actor-Critic-Methods.html" onerror="this.href='http://localhost:4000/ml/rl/dl/2024/03/29/Actor-Critic-Methods.html'">see here</a>.

### Section 13.6: Policy Gradient for Continuing Problems

For continuing problems without episode boundaries, the performance must be defined in terms of the average rate of reward per time step:

$$
\begin{align}
J(\theta) \doteq r(\pi) &\doteq \lim_{h \rightarrow \infty} \frac{1}{h} \sum_{t = 1}^h \mathbb{E}[R_t \vert S_0, A_{0:t - 1} \sim \pi] \nonumber\\
	&= \lim_{t \rightarrow \infty} \mathbb{E}[R_t \vert S_0, A_{0:t - 1} \sim \pi] \nonumber\\
	&= \sum_s \mu(s) \sum_a \pi (a \vert s) \sum_{s', r} p(s', r \vert s, a) \cdot r, 
\end{align}
$$

where $$\mu$$ is the steady-state distribution under $$\pi$$, $$\mu(s) \doteq \lim_{t \rightarrow \infty} P(S_t = s \vert A_{0:t} \sim \pi)$$,
which is assumed to exist and - due to the ergodicity assumption - to be independent of $$S_0.$$ This is a special distribution under where, if you select actions according to $$\pi$$, you remain in the same distribution, as follows:

$$
\begin{equation}
\sum_s \mu(s) \sum_a \pi(a \vert s, \theta) \cdot p(s' \vert s, a) = \mu(s'), \ \forall s' \in S.
\end{equation}
$$

In the continuing case, we define values: $$v^{\pi} (s) \doteq \mathbb{E}_{\pi} (G_t \vert S_t = s)$$ and $$q^{\pi}(s, a) \doteq \mathbb{E} [G_t \vert S_t = s, A_t = a]$$, w.r.t. the differential return (s.t. the policy gradient theorem holds true for the continuing case):

$$
\begin{equation}
G_t \doteq R_{t+1} - r(\pi) + R_{t+2} - r(\pi) + R_{t+3} - r(\pi) + \dots \ .
\end{equation}
$$

You can find an example implementation of a policy gradient method for continuing problems <a href="https://acfpeacekeeper.github.io/github-pages/ml/rl/dl/2024/03/29/Actor-Critic-Methods.html#continuing-actor-critic-with-eligibility-traces" onerror="this.href='http://localhost:4000/ml/rl/dl/2024/03/29/Actor-Critic-Methods.html#continuing-actor-critic-with-eligibility-traces'">here</a>.

### Section 13.7: Policy Parameterization for Continuous Actions

In policy-based methods, instead of computing learned probabilities for each and every action, instead we learn statistics of the probability distribution, e.g., the action set might be $$\mathbb{R}$$, with actions chosen from a normal (Gaussian) distribution.
The *Probability Density Function* (PDF) for this (normal) distribution can be written as

$$
\begin{equation}
p(x) \doteq \frac{1}{\sigma \sqrt{2 \pi}} \exp(-\frac{(x - \mu)^2}{2 \sigma^2}),
\end{equation}
$$

where $$\mu$$ is the mean and $$\sigma$$ the standard deviation of the normal distribution, and $$\pi \approx 3.14159$$. 

In order to make a policy parameterization, the policy may be defined as the normal probability density over a real-valued scalar action, with $$\mu$$ and $$\sigma$$ given by the parametric function approximators that depend on the state, i.e.,

$$
\begin{equation}
\pi (a \vert s, \theta) \doteq \frac{1}{\sigma(s, \theta)\sqrt{2 \pi}} \exp \bigg(-\frac{(a - \mu(s, \theta))^2}{2 \sigma(s, \theta)^2} \bigg),
\end{equation}
$$

where $$\mu: \mathcal{S} \times \mathbb{R}^{d'} \rightarrow \mathbb{R}$$ and $$\sigma : \mathcal{S} \times \mathbb{R}^{d'} \rightarrow \mathbb{R}^+$$ are two parameterized function approximators. Now, we just need to give a form for these approximators. To do this, we divide the policy's parameters into two parts, i.e., $$\theta = [\theta_{\mu}, \theta_{\sigma}]$$, one part will be used to approximate $$\mu$$ and the other to approximate $$\sigma > 0$$, the latter of which is better approximated as the exponential of a linear function. Thus

$$
\begin{equation}
\mu (s, \theta) \doteq \theta_{\mu}^T x_{\mu} \wedge \sigma(s, \theta) \doteq \exp \bigg(\theta_{\sigma}^T x_{\sigma}(s) \bigg),
\end{equation}
$$

where $$x_{\mu}(s)$$ and $$x_{\sigma}(s)$$ are state feature vectors. With these additional definitions, all previously described algorithms can be applied to learn to select real-valued actions.

### Section 13.8: Summary

Prior to this chapter: focus on *action-value methods* - which are methods that learn action values and then use them to select actions.

During this chapter: describes methods that learn parameterized policies that enable actions to be taken without consulting action-value estimates, with a focus on *policy-gradient methods* - which are methods that, on each step, update the policy parameter in the direction of an estimate of the gradient of the performance w.r.t. the policy parameter.

Advantages of methods that learn and store a policy parameter:
- They can learn specific probabilities for taking actions;
- They can learn appropriate levels of exploration and approach deterministic policies asymptotically;
- They can inherently handle continuous action spaces;
- The policy may be simpler to represent parametrically than the value function on some problems;
- The *policy gradient theorem* provides an exact formula (that doesn't involve derivatives of the state distribution) for how performance is affected by the policy parameter.

A state-value function baseline reduces the variance of the REINFORCE method without introducing bias. If the state-value function is (also) used to assess the policy's action selections, then it is called a *critic*, the policy is called an *actor*, and the overall algorithm is called an *actor-critic method*. The critic introduces bias into the actor’s gradient estimates, but this is often desirable since it substantially reduces variance (similar to the advantage bootstrapping TD methods have over Monte Carlo methods).

# Part III: Looking Deeper

## Chapter 14: Psychology

## Chapter 15: Neuroscience

## Chapter 16: Applications and Case Studies

## Chapter 17: Frontiers
