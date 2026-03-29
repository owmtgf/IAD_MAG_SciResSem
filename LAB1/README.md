# LAB 1: Task Description

## Team:
- [Аверьянова Мария](https://github.com/maria-aver)
- [Калягин Дмитрий](https://github.com/owmtgf)
- [Кашникова Анна](https://github.com/melpomene310)
- [Лобанов Илья](https://lh3.google.com/u/0/d/1Nl1sIu4EQ-1S9WXjwD43XeXN6kYh-VNd=w1080-h1792-iv2?auditContext=prefetch)
- [Посаженников Максим](https://github.com/trixTRr)
- [Железнова Александра](https://github.com/aleksa2001)
- [Кочнев Максим](https://lh3.google.com/u/0/d/1Nl1sIu4EQ-1S9WXjwD43XeXN6kYh-VNd=w1080-h1792-iv2?auditContext=prefetch)

## Task Assignment

> **Group 5**

### Problem
Consider an **assignment problem** where the goal is to assign $n$ workers to $n$ tasks such that the
total cost is minimized. The problem can be formulated as:

$$
\min_x \sum_{i=1}^{n} \sum_{j=1}^{n} c_{ij} x_{ij}
$$

$$
\sum_{j=1}^{n} x_{ij} = 1, \qquad \forall i \in \{ 1, \dots , n \} \\
\sum_{i=1}^{n} x_{ij} = 1, \qquad \forall j \in \{ 1, \dots , n \} \\
x_{ij} \in \{ 0, 1 \}, \qquad \forall i,j \in \{ 1, \dots , n \}
$$

where $c_{ij}$ is the cost of assigning worker $i$ to task $j$. Furthermore, $x_{ij} = 1$ if worker $i$ is assigned to task $j$, and $0$ otherwise.

### Robust optimization approach

1. What are the uncertain parameters in your problem? Can we assume that these parameters
are bounded or lie within an uncertainty set?
2. Formulate a robust version of the problem with the budget-constrained uncertainty set
similar to [[2]](https://www.researchgate.net/publication/225134339_Sim_M_Robust_discrete_optimization_and_network_flows_Math_Prog_98_49-71).
3. Reformulate the problem as a single-level problem and use `Gurobi`/`CPLEX` to solve it.
4. For each $n \in \{5, 10, \dots , 50\}$, generate `100` test instances of the problem by changing the
nominal values of the uncertain problem parameters.
5. For fixed `n`, compute the average in-sample and out-of-sample performance of robust solutions (with standard deviations), depending on the budget (`Γ`).
6. For fixed `Γ`, compute the average solution times as a function of `n`.
7. Provide the code with detailed comments

### Stochastic programming approach

1. Assume a true distribution of the uncertain problem parameters, which is typical in the
context of your optimization problem.
2. Formulate risk-neutral and risk-averse versions of the stochastic programming problem and
their sample average approximations (`SAA`).
3. For each $n \in \{5, 10, \dots , 50\}$, generate `100` test instances of the problem by using `k = 30`
samples from the true distribution.
4. For fixed `n`, compute the average quality of `SAA` solutions (with standard deviations),
for each objective function. What is the out-of-sample error that the decision-maker may
encounter when using the risk-neutral `SAA` solution for the risk-averse objective function
and vice versa?
5.  Compute the average solution times as a function of n.
6.  Provide the code with detailed comments.

---

# LAB 1: Contents Description

## Robust optimization approach

###  Uncertain parameters
> What are the uncertain parameters in your problem? \
Can we assume that these parameters are bounded or lie within an uncertainty set?

Given the task
$$
\min_x \sum_{i=1}^{n} \sum_{j=1}^{n} c_{ij} x_{ij}
$$
Where we are minimizing a sum of all cost values of $C^\top X$ matrix.

$x_{ij}$ is an indicator if the worker $i$ is assigned to task $j$ or not. We can only assign a single worker on a single task.

$c_{ij}$ is a cost of assigning the worker $i$ to $j$ task. So The **uncertain parameters for our task** are the assignment costs $c_{ij}$.

The parameter $\Gamma$ (budget) does not represent a monetary budget, but rather the number of coefficients that are allowed to deviate from their nominal values in the worst-case scenario.

In the assignment problem, since exactly $n$ variables take value 1, the parameter $\Gamma$ controls how many of these selected assignments can experience the worst-case increase in cost.

When $\Gamma = 0$, the problem reduces to the nominal (deterministic) formulation. \
When $\Gamma = n$, all selected costs may deviate, resulting in the fully worst-case (most conservative) solution.

We assume that each cost lies within a bounded interval around its nominal value:
$$
c_{ij} = \bar{c}_{ij} + d_{ij} z_{ij}, \qquad z_{ij} \in [0, 1]
$$
and the total deviation is limited by a budget $\Gamma$:
$$
\sum_{i=1}^{n} \sum_{j=1}^{n} z_{ij} \le \Gamma
$$


Here, $\bar{c}_{ij}$ denotes the nominal (deterministic) value of the assignment cost, which represents the expected or baseline cost (already known or the most expected).

$d_{ij}$ represents the maximum possible deviation of the cost coefficient $c_{ij}$ from its nominal value $\bar{c}_{ij}$.  

The variable $z_{ij} \in [0,1]$ determines the level of deviation, where:
- $z_{ij} = 0$ corresponds to no deviation,
- $z_{ij} = 1$ corresponds to the worst-case deviation.

Thus, the parameter $c_{ij}$ can increase up to $\bar{c}_{ij} + d_{ij}$, depending on the realization of uncertainty.



### Robust version of the problem with the budget-constrained uncertainty set
> Formulate a robust version of the problem with the budget-constrained uncertainty set

> Reformulate the problem as a single-level problem.

similar to [[2]](https://www.researchgate.net/publication/225134339_Sim_M_Robust_discrete_optimization_and_network_flows_Math_Prog_98_49-71).

Substituting the uncertain parameters into the objective function, we obtain the following min–max problem:

$$
\min_x \max_{z} \sum_{i,j} (\bar{c}_{ij} + d_{ij} z_{ij}) x_{ij}
$$
subject to:
$$
0 \le z_{ij} \le 1, \quad \sum_{i,j} z_{ij} \le \Gamma.
$$

Next we expand the sum:
$$
\min_x \max_{z} \Big(\sum_{i,j} \bar{c}_{ij} x_{ij} + \sum_{i,j} d_{ij} x_{ij} z_{ij} \Big)
$$
The first term does not depend on $z$, thus:
$$
\min_x \Big(\sum_{i,j} \bar{c}_{ij} x_{ij} + \max_{z} \sum_{i,j} d_{ij} x_{ij} z_{ij} \Big)
$$
which can be interpreted as
$$
\min_x (\text{nominal cost + worst-case additional cost})
$$

Continue with inner maximization task:
$$
\max_{z} \sum_{i,j} d_{ij} x_{ij} z_{ij}
$$
Assign a vector $a_{ij} = d_{ij} x_{ij}$:
$$
\max_{z} \sum_{i,j} a_{ij} z_{ij}
$$

Rewriting the maximization problem as a minimization problem, we obtain:
$$
\min_{z} - \sum_{i,j} a_{ij} z_{ij}
$$
subject to:
$$
z_{ij} - 1 \le 0, \quad -z_{ij} \le 0, \quad \sum_{i,j} z_{ij} - \Gamma \le 0
$$

Introducing Lagrange multipliers:
- $\nu_{ij} \ge 0$ for $z_{ij} - 1 \le 0$,
- $\lambda_{ij} \ge 0$ for $-z_{ij} \le 0$,
- $\mu \ge 0$ for $\sum z_{ij} - \Gamma \le 0$,

the Lagrangian is:
$$
L(z, \lambda, \nu, \mu) = -\sum_{i,j} a_{ij} z_{ij} - \sum_{i,j} \lambda_{ij} z_{ij} + \sum_{i,j} \nu_{ij} (z_{ij} - 1) + \mu \left(\sum_{i,j} z_{ij} - \Gamma \right)
$$

Then group by $z_{ij}$:
$$
L = -\mu \Gamma - \sum_{i,j} \nu_{ij} + \sum_{i,j} z_{ij} (-a_{ij} -\lambda_{ij} +\nu_{ij} +\mu)
$$
subject to:
$$
\nu_{ij} \ge 0 \\
\lambda_{ij} \ge 0 \\
\mu \ge 0 \\
-a_{ij} -\lambda_{ij} +\nu_{ij} +\mu = 0
$$
Due $\lambda_{ij} = -a_{ij} +\nu_{ij} +\mu = 0$ and $\lambda_{ij} \ge 0$ we can eliminate $\lambda$ completely:
$$
\nu_{ij} \ge 0 \\
\mu \ge 0 \\
-a_{ij} +\nu_{ij} +\mu \ge 0
$$

Thus, the dual problem becomes:
$$
\min_{\mu, \nu} \; \Gamma \mu + \sum_{i,j} \nu_{ij}
$$
subject to:
$$
-a_{ij} +\nu_{ij} +\mu \ge 0,
$$
$$
\mu \ge 0, \quad \nu_{ij} \ge 0
$$

Substituting the dual formulation back into the original problem, we obtain the following robust counterpart:

$$
\min_{x, \mu, \nu} \sum_{i,j} \bar{c}_{ij} x_{ij} + \Gamma \mu + \sum_{i,j} \nu_{ij}
$$

subject to:
$$
\nu_{ij} \ge d_{ij} x_{ij} - \mu, \quad \forall i,j,
$$
$$
\mu \ge 0, \quad \nu_{ij} \ge 0,
$$

and the assignment constraints:
$$
\sum_j x_{ij} = 1, \quad \forall i,
$$
$$
\sum_i x_{ij} = 1, \quad \forall j,
$$
$$
x_{ij} \in \{0,1\}
$$

### Implementation

You can find `RobustAssignment` class in `./solver.py` and use it to instantiate a specified task solver with input parameters:
- `c_bar`: nominal cost matrix `(n x n)`
- `d`: deviation matrix `(n x n)`
- `Gamma`: uncertainty budget

Then you can use `solve()` method to get an optimized solution consisting of a tuple of found $X$ matrix and minimum $\min_x \sum_{i=1}^{n} \sum_{j=1}^{n} c_{ij} x_{ij}$ function value.

**Example usage:**
```python
from pprint import pprint
import numpy as np

from solver import RobustAssignment

n = 5
c_bar = np.random.randint(1, 20, (n, n))
d = np.random.randint(1, 5, (n, n))
Gamma = 2

model = RobustAssignment(
    c_bar=c_bar, 
    d=d, 
    Gamma=Gamma
)

sol, obj = model.solve()
pprint(sol)
print(f"{obj=}")
```
```
>>> [[-0.0, 1.0, 0.0, -0.0, -0.0],
    [0.0, -0.0, 1.0, -0.0, 0.0],
    [1.0, -0.0, -0.0, 0.0, -0.0],
    [-0.0, -0.0, -0.0, 1.0, -0.0],
    [-0.0, -0.0, -0.0, -0.0, 1.0]]

>>> obj=39.0
```

## Stochastic programming approach

### Distributional Assumptions
> Assume a true distribution of the uncertain problem parameters, which is typical in the context of your optimization problem.

We assume that the $c_{ij}$ are independent random variables and satisfy:

$$
c_{ij} \sim \mathcal{N}(\mu_{ij}, \sigma_{ij}^2) = \mathbb{P}_{ij}, \quad \forall i,j.
$$

The parameters are defined as:

$$
\mu_{ij} = 5 + i + 2j, \qquad
\sigma_{ij} = 1 + 0.1(i + j).
$$

The expected cost $\mu_{ij}$ represents the typical cost of assigning worker $i$ to task $j$. It depends on both the worker and the task:

- Larger $j$ corresponds to a more difficult task, so the expected cost is higher.
- Larger $i$ corresponds to a less efficient worker, so the expected cost is also higher.

The standard deviation $\sigma_{ij}$ represents the uncertainty of this assignment cost. Assignments with larger $i+j$ are more variable, reflecting that complex or difficult worker–task combinations are less predictable.

### Stochastic problem formulation

> Formulate risk-neutral and risk-averse versions of the stochastic programming problem and their sample average approximations (SAA).

Suppose:
$$
f(x, c) = \sum_{i=1}^{n} \sum_{j=1}^{n} c_{ij} x_{ij}
$$

Then the risk-neutral stochastic optimization problem can be formulate as:
$$
\min_{x \in X} \mathbb{E}_\mathbb{P}[f(x, c)] = \min_{x \in X} \sum_{i=1}^{n} \sum_{j=1}^{n} \mathbb{E}_{\mathbb{P}_{ij}}[c_{ij}]x_{ij}
$$

To incorporate risk averse, we consider the conditional value-at-risk (CVaR) at level  $\alpha \in (0,1)$. The problem is formulated as:
$$
\min_{x \in X,\; t \in \mathbb{R}} 
\left(
t + \frac{1}{1-\alpha} \mathbb{E}_{\mathbb{P}}\left[\max \left \{f(x, c) - t, 0 \right\}\right]
\right) = \\ = \min_{x \in X,\; t \in \mathbb{R}} 
\left(
t + \frac{1}{1-\alpha} \left[\max \left \{\sum_{i=1}^{n} \sum_{j=1}^{n} \mathbb{E}_{\mathbb{P_{ij}}}[c_{ij}] x_{ij} - t, 0 \right \}\right]
\right)
$$

All this general formulations with discrete distribution assumption can be simplified into:
- Risk-Neutral:
$$
\min_{x \in X} \sum_{i=1}^{n} \sum_{j=1}^{n} \left( \sum_{k=1}^{K}{p_{ij}^{k}}c_{ij}^{k} \right) x_{ij}
$$
- Risk-Averse:
$$
\min_{x \in X,\; t\in \mathbb{R}}
\left(
t + \frac{1}{1-\alpha} \left[\max \left \{\sum_{i=1}^{n} \sum_{j=1}^{n} \left( \sum_{k=1}^{K} p_{ij}^k c_{ij}^k \right) x_{ij} - t, 0 \right \}\right]
\right) \Longrightarrow \\
\Longrightarrow
\begin{cases}
\min_{x \in X,\; t\in \mathbb{R}}
\left(
t + \frac{1}{1-\alpha} \sum_{k=1}^{K} p^k z^k
\right) \\
z^k \geq \sum_{i=1}^{n} \sum_{j=1}^{n} c_{ij}^k x_{ij} - t \\
z^k \geq 0
\end{cases}
$$
where each $c_{ij} \sim p_{ij}$ with $K$ unique values.

Their sample average approximation (SAA) will be:
- Risk-Neutral:
$$
\min_{x \in X} \frac{1}{K} \sum_{k=1}^{K} f(x, c^k)
= \min_{x \in X} \left( \sum_{i=1}^{n} \sum_{j=1}^{n} \left(\frac{1}{K} \sum_{k=1}^{K} c_{ij}^k \right) x_{ij}\right)
$$
- Risk-Averse:
$$
\min_{x \in X,\; t\in \mathbb{R}} 
\left[
t + \frac{1}{(1-\alpha)}  \frac{1}{K}
\sum_{k=1}^{K} 
\max\left\{ f(x,c^k) - t, 0 \right\}
\right] = \\ = 
\min_{x \in X,\; t\in \mathbb{R}} 
\left[
t + \frac{1}{(1-\alpha)}  
\max\left \{ \sum_{i=1}^{n} \sum_{j=1}^{n} \left( \frac{1}{K}
\sum_{k=1}^{K} c_{ij}^k \right) x_{ij} - t, 0 \right \}
\right]
\Longrightarrow \\
\Longrightarrow
\begin{cases}
\min_{x \in X,\; t\in \mathbb{R}}
\left(
t + \frac{1}{1-\alpha} \sum_{k=1}^{K} z^k
\right) \\
z^k \geq \sum_{i=1}^{n} \sum_{j=1}^{n} c_{ij}^k x_{ij} - t \\
z^k \geq 0
\end{cases}
$$

### Implementation
You can find `StochasticInstanceGenerator` class in `./instance_generator.py` for sampling instances from distribution of cost matrix as was defined above. 
For solving find `StochasticAssignment` class in `./solver.py` and use it to instantiate a specified task solver with input parameters:
- `cost_matrix`: generated cpst matrix `(n x n)`
- `alpha`: alpha param (need for RA SAA solver)

Then you can use 2 types of solvers: 
- Risk-Neutral Sample Average Approximation: `solve_risk_neutral()`;
- Risk-Averse Sample Average Approximation: `solve_risk_averse()`;
output format aligned with `RobustAssignment` solver.