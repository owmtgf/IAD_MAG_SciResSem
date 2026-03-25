# LAB 1

## Team:
- [Аверьянова Мария](https://github.com/maria-aver)
- [Калягин Дмитрий](https://github.com/owmtgf)
- [Кашникова Анна](https://github.com/melpomene310)
- [Лобанов Илья](https://lh3.google.com/u/0/d/1Nl1sIu4EQ-1S9WXjwD43XeXN6kYh-VNd=w1080-h1792-iv2?auditContext=prefetch)
- [Посаженников Максим](https://lh3.google.com/u/0/d/1Nl1sIu4EQ-1S9WXjwD43XeXN6kYh-VNd=w1080-h1792-iv2?auditContext=prefetch)
- [Железнова Александра](https://lh3.google.com/u/0/d/1Nl1sIu4EQ-1S9WXjwD43XeXN6kYh-VNd=w1080-h1792-iv2?auditContext=prefetch)
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

## Interpretation & TODO


| Role | People | Task |
|-|-|-|
| A. Base model + solver | 1 | assignment model, solver setup |
| B. Robust optimization | 2 | robust formulation + Γ budget |
| C. Stochastic programming | 2 | SAA + risk-neutral + risk-averse |
| D. Instance generator + experiments | 1 | generate 100 tests, loops |
| E. Analysis + plots + report | 1 | stats, tables, writing |

---
**A.** Core Model
- implement assignment `MILP`
- make function `solve_assignment(c)`
- connect to `Gurobi`/`CPLEX`
- test for `n=5,10`

---
**B.** Robust Optimization 

Tasks:
- define uncertain parameters -> costs $c_{ij}$ uncertain
- define uncertainty set
- implement `Γ`-budget
- reformulate to MILP
- solve for different `Γ`

Compute:
- in-sample performance
- out-of-sample performance

---
**C.** Stochastic Programming

Tasks:
- define distribution of costs
- generate samples
- implement `SAA`
- risk-neutral
- risk-averse

Compute:
- solution quality
- out-of-sample error
- `time` vs `n`

---
**D.** Experiment automation
- loop over `n = 5, ... ,50`
- generate `100` instances
- call solvers
- save results
- measure time

---
**E.** Analysis + report
- averages
- std deviation
- plots
- tables
- write explanations
- clean comments
