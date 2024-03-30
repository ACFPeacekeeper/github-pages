---
layout: post
title: "Python Examples: REINFORCE Algorithm"
date: 2024-03-28
categories: RL ML DL
usemathjax: true
---
Here are some example implementations of the REINFORCE algorithm, to go along with my review of <a href="https://acfpeacekeeper.github.io/github-pages/rl/ml/dl/2024/03/28/Notes-on-RL-an-Introduction.html#ch-133-reinforce-monte-carlo-policy-gradient" onerror="this.href='http://localhost:4000/rl/ml/dl/2024/03/28/Notes-on-RL-an-Introduction.html#ch-133-reinforce-monte-carlo-policy-gradient'">Section 13.3</a> and <a href="https://acfpeacekeeper.github.io/github-pages/rl/ml/dl/2024/03/28/Notes-on-RL-an-Introduction.html#ch-134-reinforce-with-baseline" onerror="this.href='http://localhost:4000/rl/ml/dl/2024/03/28/Notes-on-RL-an-Introduction.html#ch-134-reinforce-with-baseline'">Section 13.4</a> of the <a href="http://acfpeacekeeper.github.io/github-pages/docs/literature/books/RLbook2020.pdf" onerror="this.href='http://localhost:4000/docs/literature/books/RLbook2020.pdf'">Reinforcement Learning: An Introduction</a> book.

# Post Index
1. [Auxiliar Classes](#auxiliar-classes)
    1. [Policy Parameterization](#policy-parameterization)
    2. [Baseline State-Value Function](#baseline-state-value-function)
2. [REINFORCE Algorithm](#reinforce-algorithm)
    1. [REINFORCE: (Episodic) Monte-Carlo Policy-Gradient Control](#reinforce-episodic-monte-carlo-policy-gradient-control)
    2. [REINFORCE with Baseline](#reinforce-with-baseline)

# Auxiliar Classes
## Policy Parameterization
{% highlight python %}
import torch.nn as nn

class PolicyParameterization(nn.Module):
    def __init__(self, dim_observations, dim_actions, hidden_dim=128):
        super(PolicyParameterization, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(dim_observations, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_actions),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.policy(x)
{%endhighlight %}

## Baseline State-Value Function
{% highlight python %}
import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self, dim_observations, hidden_dim=128):
        super(Baseline, self).__init__()
        self.baseline = nn.Sequential(
            nn.Linear(dim_observations, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.baseline(x)
{% endhighlight %}

# REINFORCE Algorithm
## REINFORCE: (Episodic) Monte-Carlo Policy-Gradient Control
{% highlight python %}
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

def reinforce(
    alpha: float,
    gamma: float
    NUM_EPISODES: int,
    MAX_STEPS: int,
    gym_environment: str
):
    assert alpha > 0, "alpha must be a float value greater than 0"
    assert gamma > 0 and gamma < 1, "gamma must be a float value greater than 0 and smaller than 1"
    assert NUM_EPISODES > 0, "NUM_EPISODES must be an int value greater than 0"
    assert MAX_STEPS > 0, "MAX_STEPS must be an int value greater than 0"

    # Initialize policy parameters
    policy = PolicyParameterization(env.observation_space.shape[0], env.action_space.n)
    for module in policy.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform(module.weight)
            module.bias.data.fill_(0.01)

    # Initialize optimizer
    policy_optimizer = optim.SGD(policy.parameters(), lr=alpha)

    state_transition = collections.namedtuple("state_transition", ["state", "action", "reward", "next_state", "done"])
    env = gym.make(gym_environment)
    scores = []
    for episode_id in range(NUM_EPISODES):
        state = env.reset()
        episode = []
        for _ in range(MAX_STEPS):
            # Get action and log probabilities
            a_probs = policy(state)
            prob_dist = Categorical(a_probs)
            a = prob_dist.sample()
            action = a.item()
            log_probs = prob_dist.log_prob(action)

            # Step with action
            new_state, R, done, _ = env.step(action)

            # Update episode score
            score += R

            # Keep track of state transitions
            episode.append(state_transition(state=state, action=action, reward=R, next_state=new_state, done=done))

            if done:
                break

            # Move into new state
            state = new_state
        
        # Append episode score
        scores.append(score)

        # Update policy
        for id, transition in enumerate(episode):
            total_reward = sum(gamma**episode_id * t.reward for i, t in enumerate(episode[t:]))
            policy_optimizer.zero_grad()
            total_reward.backward()
            policy_optimizer.step()

    return scores, policy
{% endhighlight %}

## REINFORCE with Baseline
{% highlight python %}
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

def reinforce(
    alpha_theta: float,
    alpha_w: float,
    gamma: float
    NUM_EPISODES: int,
    MAX_STEPS: int,
    gym_environment: str
):
    assert alpha_theta > 0, "alpha_theta must be a float value greater than 0"
    assert alpha_w > 0, "alpha_w must be a float value greater than 0"
    assert gamma > 0 and gamma < 1, "gamma must be a float value greater than 0 and smaller than 1"
    assert NUM_EPISODES > 0, "NUM_EPISODES must be an int value greater than 0"
    assert MAX_STEPS > 0, "MAX_STEPS must be an int value greater than 0"

    # Initialize policy parameters
    policy = PolicyParameterization(env.observation_space.shape[0], env.action_space.n)
    for module in policy.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform(module.weight)
            module.bias.data.fill_(0.01)

    # Initialize baseline parameters
    baseline = Baseline(env.observation_space.shape[0])
    for module in baseline.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.fill_(0.0)
            module.bias.data.fill_(0.01)

    # Initialize optimizers
    policy_optimizer = optim.SGD(policy.parameters(), lr=alpha_theta)
    baseline_optimizer = optim.SGD(baseline.parameters(), lr=alpha_w)

    state_transition = collections.namedtuple("state_transition", ["state", "action", "reward", "next_state", "done"])

    env = gym.make(gym_environment)
    scores = []
    for episode_id in range(NUM_EPISODES):
        state = env.reset()
        episode = []
        for _ in range(MAX_STEPS):
            # Get action and log probabilities
            a_probs = policy(state)
            prob_dist = Categorical(a_probs)
            a = prob_dist.sample()
            action = a.item()
            log_probs = prob_dist.log_prob(action)

            # Step with action
            new_state, R, done, _ = env.step(action)

            # Update episode score
            score += R

            # Keep track of state transitions
            episode.append(state_transition(state=state, action=action, reward=R, next_state=new_state, done=done))

            if done:
                break

            # Move into new state
            state = new_state
        
        # Append episode score
        scores.append(score)

        # Update policy and baseline
        for id, transition in enumerate(episode):
            total_reward = sum(gamma**episode_id * t.reward for i, t in enumerate(episode[t:]))
            baseline_value = baseline(transition.state)
            delta = total_reward - baseline_value

            baseline_optimizer.zero_grad()
            total_reward.backward()
            baseline_optimizer.step()

            policy_optimizer.zero_grad()
            delta.backward()
            policy_optimizer.step()

    return scores, policy, baseline
{% endhighlight %}