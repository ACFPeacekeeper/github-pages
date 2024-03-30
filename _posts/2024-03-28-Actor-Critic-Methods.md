---
layout: post
title: "Actor-Critic Methods: PyTorch Examples"
date: 2024-03-28
categories: RL ML DL
usemathjax: true
---
Here are some example implementations of actor-critic methods, to go along with <a href="https://acfpeacekeeper.github.io/github-pages/rl/ml/dl/2024/03/28/Notes-on-RL-an-Introduction.html#chapter-13-policy-gradient-methods" onerror="this.href='http://localhost:4000/rl/ml/dl/2024/03/28/Notes-on-RL-an-Introduction.html#chapter-13-policy-gradient-methods'">my review of Chapter 13</a> of the <a href="http://acfpeacekeeper.github.io/github-pages/docs/literature/books/RLbook2020.pdf" onerror="this.href='http://localhost:4000/docs/literature/books/RLbook2020.pdf'">Reinforcement Learning: An Introduction</a> book.

## Auxiliar Classes
### Policy
{% highlight python %}
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, dim_observations, dim_actions):
        super(Policy, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(dim_observations, 128),
            nn.ReLU(),
            nn.Linear(128, dim_actions),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.policy(x)

{%endhighlight %}

### State-Value Function
{% highlight python %}
import torch.nn as nn

class StateValueFunction(nn.Module):
    def __init__(self, dim_observations):
        super(StateValueFunction, self).__init__()
        self.stateval_func = nn.Sequential(
            nn.Linear(dim_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.stateval_func(x)
{% endhighlight %}

## Actor-Critic Methods
### Episodic One-step Actor-Critic
{% highlight python %}
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from torch.distributions import Categorical

def episodic_one_step_actor_critic(
    alpha_theta: float,
    alpha_w: float,
    gamma: float,
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
    policy = Policy(env.observation_space.shape[0], env.action_space.n)
    for module in policy.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform(module.weight)
            module.bias.data.fill_(0.01)

    # Initialize state-value function parameters
    state_value_func = StateValueFunction(env.observation_space.shape[0])
    for module in state_value_func.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.fill_(0.0)
            module.bias.data.fill_(0.01)

    # Initialize optimizers
    policy_optimizer = optim.SGD(policy.parameters(), lr=alpha_theta)
    stateval_optimizer = optim.SGD(state_value_func.parameters(), lr=alpha_w)

    env = gym.make(gym_environment)
    scores = []
    for _ in range(NUM_EPISODES):
        score = 0.
        I = 1.
        state = env.reset()
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

            # Get state value of current state
            state_value = state_value_func(state)

            # Get state value of next state
            # If terminal state, next state value is 0
            new_state_value = [0.] if done else state_value_func(new_state)

            # Calculate value function loss with MSE
            delta = R + gamma * new_state_value.item() - state_value.item()
            val_loss = delta * F.mse_loss(R + gamma * new_state_value, state_value)

            # Calculate policy loss
            policy_loss = I * delta * log_probs

            # Backpropagate value
            stateval_optimizer.zero_grad()
            val_loss.backward()
            stateval_optimizer.step()

            # Backpropagate policy
            policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            policy_optimizer.step()

            if done:
                break

            # Move into new state, discount I
            I *= gamma
            state = new_state

        # Append episode score
        scores.append(score)

    return scores, policy, state_value_func
{% endhighlight %}

### Episodic Actor-Critic with Eligibility Traces
{% highlight python %}
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from torch.distributions import Categorical

def episodic_actor_critic_with_eligibility_traces(
	lambda_theta: float,
	lambda_w: float,
	alpha_theta: float,
	alpha_w: float,
	gamma: float,
	NUM_EPISODES: int,
	MAX_STEPS: int,
	gym_environment: str
):
    assert lambda_theta >= 0 and lambda_theta =< 1, "lambda_theta must be a float value between 0 and 1"
    assert lambda_w >= 0 and lambda_w =< 1, "lambda_w must be a float value between 0 and 1"
    assert alpha_theta > 0, "alpha_theta must be a float value greater than 0"
    assert alpha_w > 0, "alpha_w must be a float value greater than 0"
    assert gamma > 0 and gamma < 1, "gamma must be a float value greater than 0 and smaller than 1"
    assert NUM_EPISODES > 0, "NUM_EPISODES must be an int value greater than 0"
    assert MAX_STEPS > 0, "MAX_STEPS must be an int value greater than 0"

    # Initialize policy parameters
    policy = Policy(env.observation_space.shape[0], env.action_space.n)
    for module in policy.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform(module.weight)
            module.bias.data.fill_(0.01)

    # Initialize state-value function parameters
    state_value_func = StateValueFunction(env.observation_space.shape[0])
    for module in state_value_func.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.fill_(0.0)
            module.bias.data.fill_(0.01)

    # Initialize optimizers
    policy_optimizer = optim.SGD(policy.parameters(), lr=alpha_theta)
    stateval_optimizer = optim.SGD(state_value_func.parameters(), lr=alpha_w)

    env = gym.make(gym_environment)
    scores = []
    for _ in range(NUM_EPISODES):
        z_theta = 0.
        z_w = 0.
        score = 0.
        I = 1.
        state = env.reset()
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

            # Get state value of current state
            state_value = state_value_func(state)

            # Get state value of next state
            # If terminal state, next state value is 0
            new_state_value = [0.] if done else state_value_func(new_state)

            # Calculate value function loss with MSE
            delta = R + gamma * new_state_value.item() - state_value.item()
            z_w = delta * (gamma * lambda_w * z_w + F.mse_loss(R + gamma * new_state_value, state_value))

            # Calculate policy loss
            z_theta = delta * (gamma * lambda_theta * z_theta + I * log_probs)

            # Backpropagate value
            stateval_optimizer.zero_grad()
            z_w.backward()
            stateval_optimizer.step()

            # Backpropagate policy
            policy_optimizer.zero_grad()
            z_theta.backward(retain_graph=True)
            policy_optimizer.step()

            if done:
                break

            # Move into new state, discount I
            I *= gamma
            state = new_state

        # Append episode score
        scores.append(score)

    return scores, policy, state_value_func
{% endhighlight %}

### Continuing Actor-Critic with Eligibility Traces
{% highlight python %}
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from torch.distributions import Categorical

def episodic_actor_critic_with_eligibility_traces(
    lambda_theta: float,
    lambda_w: float,
    alpha_theta: float,
    alpha_w: float,
    alpha_R: float,
    gamma: float,
    MAX_STEPS: int,
    gym_environment: str
):
    assert lambda_theta >= 0 and lambda_theta =< 1, "lambda_theta must be a float value between 0 and 1"
    assert lambda_w >= 0 and lambda_w =< 1, "lambda_w must be a float value between 0 and 1"
    assert alpha_theta > 0, "alpha_theta must be a float value greater than 0"
    assert alpha_w > 0, "alpha_w must be a float value greater than 0"
    assert alpha_R > 0, "alpha_R must be a float value greater than 0"
    assert gamma > 0 and gamma < 1, "gamma must be a float value greater than 0 and smaller than 1"
    assert MAX_STEPS > 0, "MAX_STEPS must be an int value greater than 0"

    # Initialize policy parameters
    policy = Policy(env.observation_space.shape[0], env.action_space.n)
    for module in policy.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform(module.weight)
            module.bias.data.fill_(0.01)

    # Initialize state-value function parameters
    state_value_func = StateValueFunction(env.observation_space.shape[0])
    for module in state_value_func.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.fill_(0.0)
            module.bias.data.fill_(0.01)

    # Initialize optimizers
    policy_optimizer = optim.SGD(policy.parameters(), lr=alpha_theta)
    stateval_optimizer = optim.SGD(state_value_func.parameters(), lr=alpha_w)

    env = gym.make(gym_environment)
    avg_R = 0.
    score = 0.
    state = env.reset()
    z_theta = 0.
    z_w = 0.
    for _ in range(MAX_STEPS):
        # Get action and log probabilities
        a_probs = policy(state)
        prob_dist = Categorical(a_probs)
        a = prob_dist.sample()
        action = a.item()
        log_probs = prob_dist.log_prob(action)

        # Step with action
        new_state, R, _, _ = env.step(action)

        # Update episode score
        score += R

        # Get state value of current state
        state_value = state_value_func(state)

        # Get state value of next state
        # If terminal state, next state value is 0
        new_state_value = state_value_func(new_state)

        # Update reward values
        delta = R - avg_R + new_state_value - state_value
        avg_R += alpha_R * delta

        # Calculate value function loss with MSE
        z_w = lambda_w * z_w + F.mse_loss(R + gamma * new_state_value, state_value)

        # Calculate policy loss
        z_theta = lambda_theta * z_theta + log_probs

        # Backpropagate value
        stateval_optimizer.zero_grad()
        z_w.backward()
        stateval_optimizer.step()

        # Backpropagate policy
        policy_optimizer.zero_grad()
        z_theta.backward(retain_graph=True)
        policy_optimizer.step()

        # Move into new state
        state = new_state

    return score, policy, state_value_func
{% endhighlight %}