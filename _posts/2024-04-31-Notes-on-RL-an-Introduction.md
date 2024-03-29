---
layout: post
title: "Notes on the 2nd edition of Reinforcement Learning: An Introduction"
date: 2024-03-28
categories: RL ML DL
usemathjax: true
---
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