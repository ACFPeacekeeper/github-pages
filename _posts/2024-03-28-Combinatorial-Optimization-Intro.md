---
layout: post
title: "Combinatorial Optimization: An Introduction"
date: 2024-04-28
categories: ML RL CO
usemathjax: true
---
## Introduction
To understand what exactly Combinatorial Optimization means, we can start by understanding each term of the expression: Combinatorial and Optimization.  

### Combinatorial Problems
CO deals with a class of problems called combinatorial problems. These are a particular case of discrete problems - which are problems where the variables are assumed to take discrete values - that involve finding a ordered or unordered grouping which, given a finite (and usually large) set of objects, satisfy a given set of conditions.

Combinations of elements from the set that may be encountered when trying to formulate a solution to a instance of this class of problems are called *candidate* or *feasible* solutions. Then, the *solutions* to this kind of problem are the feasible solutions that satisfy all required conditions.

#### The Traveling Salesman Problem
One of the most well known combinatorial problems is the *Traveling Salesman Problem* (TSP), where, given a graph $$G(N, E): N = \{v_1, ..., v_n\} \wedge E \subseteq \{(u, v): u, v \in N\}$$, the objective is to find shortest path possible $$s^* = \{\bar{s} \subseteq E: \forall s \subseteq E, \ \text{dist}(\bar{s}) \leq \text{dist}(s)\}$$ that visits each of the nodes $$v \in N$$ once and only once.


### Optimization
Colloquially, optimization refers to the process of optimizing something, i.e., making it the best. What that means can vary, depending on the task at hand. For example, if we want to optimize the path to a location, then we may want to minimize the distance travelled or the travel time. However, if we want to optimize our wealth, than that may involve both maximizing our income and minimizing our expenses.
Formally, an optimization problem consists of:
- Objective function $$f(x): X \rightarrow Y$$, which gives us the output we are trying to optimize;
- Variable set $$X = \{x_1, ..., x_n\}$$, which are the inputs to $$f(x)$$;
- A set of constraints $$C = \{h_1(x), ..., h_n(x), g_1(x), ..., g_n(x)\}$$, which are (equality $$h_k(x)$$ and inequality $$g_k(x)$$) equations that place limits on the values that some variables may take.

#### Mathematical Optimization
Mathematical Optimization is the branch of applied mathematics and numerical analysis that aims to develop and analyse algorithms that, given a objective function and a set of variables, seek to find the optimal solution(s) from the set containing all feasible solutions.
Mathematical Optimization deals with problems that can be categorized according to:
- Whether the set of constraints is empty or not: Unlimited vs. Limited;
- What values the variables can take: Discrete vs. Continuous;
- Whether the problem state changes or not over time: Dynamic vs. Static;
- Whether randomness is involved or not: Stochastic vs. Deterministic;
- Whether equations map graphs to lines or to curves: Linear vs. Non-Linear.

### Combinatorial Optimization (CO)
Now knowing the meaning of both Combinatorial (problems) and (mathematical) Optimization, we can define CO as a subfield of Discrete Optimization with the aim of developing algorithms that, given a finite set of discrete variables $$X$$ and constraints $$C$$, searches for the maxima or minima of an objective function $$f(x)$$.


## Bibliography
- [Introduction: Combinatorial Problems & Search by UBC](https://www.cs.ubc.ca/labs/algorithms/Courses/CPSC532D-05/Slides/ch1-slides.pdf)
- [Introduction to Mathematical Optimization by Stanford](https://web.stanford.edu/group/sisl/k12/optimization/MO-unit1-pdfs/1.1optimization.pdf)
- [Mathematical Optimization by DeepAI](https://deepai.org/machine-learning-glossary-and-terms/mathematical-optimization)
- [Combinatorial Optimization by CMU](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/learn-43/lib/photoz/.g/web/glossary/comb.html)
- [Combinatorial Optimization by UT Dallas](https://personal.utdallas.edu/~dxd056000/cs6363/LectureNotes.pdf)
- [Introduction to Mathematical Optimization with Python by indrag49](https://indrag49.github.io/Numerical-Optimization/)
- Bernardo, F. P., & Oliveira, N. M. C. (2017). <a href="https://www.sciencedirect.com/science/article/abs/pii/B9780444639653504931">Discrete optimization in the chemical engineering curriculum</a>. In A. Espuña, M. Graells, & L. Puigjaner (Eds.), 27th European Symposium on Computer Aided Process Engineering (pp. 2947–2952). doi:10.1016/B978-0-444-63965-3.50493-1
- Hoos, H. H., & Stützle, T. (2005). 1 - INTRODUCTION. In H. H. Hoos & T. Stützle (Eds.), <a href="https://www.sciencedirect.com/science/article/abs/pii/B9781558608726500184">Stochastic Local Search</a> (pp. 13–59). doi:10.1016/B978-155860872-6/50018-4
- Baty, L., Jungel, K., Klein, P. S., Parmentier, A., & Schiffer, M. (2023). <a href="http://acfpeacekeeper.github.io/github-pages/docs/literature/papers/co_ml_dynamic_vrp_timewindows.pdf" onerror="this.href='http://localhost:4000/docs/literature/papers/co_ml_dynamic_vrp_timewindows.pdf'">Combinatorial Optimization enriched Machine Learning to solve the Dynamic Vehicle Routing Problem with Time Windows</a>. arXiv [Math.OC]. Retrieved from http://arxiv.org/abs/2304.00789
- Greif, T., Bouvier, L., Flath, C. M., Parmentier, A., Rohmer, S. U. K., & Vidal, T. (2024). <a href="http://acfpeacekeeper.github.io/github-pages/docs/literature/papers/co_ml_dynamic_inv_routing.pdf" onerror="this.href='http://localhost:4000/docs/literature/papers/co_ml_dynamic_inv_routing.pdf'">Combinatorial Optimization and Machine Learning for Dynamic Inventory Routing</a>. arXiv [Math.OC]. Retrieved from http://arxiv.org/abs/2402.04463
- Hopfield, J., & Tank, D. (02 1985). <a href="http://acfpeacekeeper.github.io/github-pages/docs/literature/papers/1985tsp.pdf" onerror="this.href='http://localhost:4000/docs/literature/papers/1985tsp.pdf'">Neural Computation of Decisions in Optimization Problems</a>. Biological Cybernetics, 52, 141–152. doi:10.1007/BF00339943
- Smith‐Miles, K. (1999). <a href="http://acfpeacekeeper.github.io/github-pages/docs/literature/papers/Neural_Networks_for_Combinatorial_Optimization_Review.pdf" onerror="this.href='http://localhost:4000/docs/literature/papers/Neural_Networks_for_Combinatorial_Optimization_Review.pdf'">Neural Networks for Combinatorial Optimization: A Review of More Than a Decade of Research</a>. INFORMS J. Comput., 11, 15–34. Retrieved from https://api.semanticscholar.org/CorpusID:5634955
- Li, Y., Archetti, C., & Ljubic, I. (2022). <a href="http://acfpeacekeeper.github.io/github-pages/docs/literature/papers/rl_for_op_stochastic_dynamic_release_dates.pdf" onerror="this.href='http://localhost:4000/docs/literature/papers/rl_for_op_stochastic_dynamic_release_dates.pdf'">Reinforcement Learning Approaches for the Orienteering Problem with Stochastic and Dynamic Release Dates</a>. arXiv [Math.OC]. Retrieved from http://arxiv.org/abs/2207.00885
