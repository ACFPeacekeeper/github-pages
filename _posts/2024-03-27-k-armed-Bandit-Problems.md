---
layout: post
title: "Python Examples: k-armed Bandit Problems"
date: 2024-03-27
categories: RL
usemathjax: true
---
Here are some example implementations of bandit problems, to go along with my review of <a href="https://acfpeacekeeper.github.io/github-pages/rl/ml/dl/2024/03/30/Notes-on-RL-an-Introduction.html#section-24-incremental-implementation" onerror="this.href='http://localhost:4000/rl/ml/dl/2024/03/30/Notes-on-RL-an-Introduction.html#section-24-incremental-implementation'">Section 4.2</a>  of the <a href="http://acfpeacekeeper.github.io/github-pages/docs/literature/books/RLbook2020.pdf" onerror="this.href='http://localhost:4000/docs/literature/books/RLbook2020.pdf'">Reinforcement Learning: An Introduction</a> book.

# Post Index
1. [Auxiliar Classes](#auxiliar-classes)
    1. [One-armed Bandit](#one-armed-bandit)
2. [k-armed Bandit Problems](#armed-bandit-problems)
    1. [Simple Bandit Algorithm](#simple-bandit-algorithm)

# Auxiliar Classes
## One-armed Bandit
{% highlight python %}
import torch
import random

class Bandit(object):
    def __init__(self, min_mean, max_mean, std=1):
        self.mean = random.randint(min_mean, max_mean)
        self.std = std
        self.dist = torch.distributions.Normal(self.mean, self.std)

    def sample(self, size=[1]):
        return self.dist.sample(size)
{%endhighlight %}

# $$k$$-armed Bandit Problems
## Simple Bandit Algorithm
{% highlight python %}
import torch
import random

def simple_bandit(
    bandits: tuple,
    epsilon: List[Bandit],
    MAX_STEPS:int
):
    assert len(bandits) > 0, "bandits must be a tuple with length greater than 0"
    assert epsilon > 0 and epsilon < 1, "epsilon must be a float value greater than 0 and smaller than 1"
    assert MAX_STEPS > 0, "MAX_STEPS must be an int value greater than 0"
    Q_a = torch.zeros((k))
    N_a = torch.zeros((k))

    actions = []
    rewards = []
    for _ in range(MAX_STEPS):
        action = -1
        if random.uniform(0, 1) <= epsilon:
            action = random.randint(0, k)
        else:
            action = torch.argmax(Q_a)

        R = bandits[action].sample()

        N_a[action] += 1
        Q_a[action] += (1/N_a[action])*(R - Q_a[action]) 

        actions.append(action)
        rewards.append(R)

    return actions, rewards, Q_a
        
        

        

{% endhighlight %}