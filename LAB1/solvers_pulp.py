"""
Alternative solver using PuLP (free)
"""

import pulp
import numpy as np


class RobustAssignment:
    def __init__(self, c_bar, d, Gamma):
        """
        c_bar : nominal cost matrix (n x n)
        d     : deviation matrix (n x n)
        Gamma : uncertainty budget
        """
        self.c_bar = c_bar
        self.d = d
        self.Gamma = Gamma
        self.n = c_bar.shape[0]
        self.model = None
        self.x = None
        self.mu = None
        self.nu = None

    def _build_model(self):
        n = self.n
        
        # Create model
        self.model = pulp.LpProblem("Robust_Assignment", pulp.LpMinimize)
        
        # Variables
        self.x = pulp.LpVariable.dicts("x", (range(n), range(n)), cat='Binary')
        self.mu = pulp.LpVariable("mu", lowBound=0)
        self.nu = pulp.LpVariable.dicts("nu", (range(n), range(n)), lowBound=0)
        
        # Objective
        obj = pulp.lpSum(self.c_bar[i, j] * self.x[i][j] for i in range(n) for j in range(n))
        obj += self.Gamma * self.mu
        obj += pulp.lpSum(self.nu[i][j] for i in range(n) for j in range(n))
        self.model += obj
        
        # Assignment constraints
        for i in range(n):
            self.model += pulp.lpSum(self.x[i][j] for j in range(n)) == 1
        
        for j in range(n):
            self.model += pulp.lpSum(self.x[i][j] for i in range(n)) == 1
        
        # Robust constraints
        for i in range(n):
            for j in range(n):
                self.model += self.nu[i][j] >= self.d[i, j] * self.x[i][j] - self.mu

    def solve(self):
        self._build_model()
        
        # Solve
        self.model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if pulp.LpStatus[self.model.status] == 'Optimal':
            # Extract solution
            x_sol = [[int(self.x[i][j].varValue) for j in range(self.n)] for i in range(self.n)]
            obj = pulp.value(self.model.objective)
            return x_sol, obj
        else:
            return None, None


class StochasticAssignment:
    def __init__(self, cost_samples, alpha=0.9, time_limit=120):
        """
        cost_samples : list of k matrices (n x n)
        alpha : CVaR confidence level
        """
        self.cost_samples = cost_samples
        self.n = cost_samples[0].shape[0]
        self.k = len(cost_samples)
        self.alpha = alpha
        self.time_limit = time_limit

    def solve_risk_neutral(self):
        avg_cost = np.mean(self.cost_samples, axis=0)
        
        model = pulp.LpProblem("RN_SAA", pulp.LpMinimize)
        
        x = pulp.LpVariable.dicts("x", (range(self.n), range(self.n)), cat='Binary')
        
        obj = pulp.lpSum(avg_cost[i, j] * x[i][j] for i in range(self.n) for j in range(self.n))
        model += obj
        
        for i in range(self.n):
            model += pulp.lpSum(x[i][j] for j in range(self.n)) == 1
        for j in range(self.n):
            model += pulp.lpSum(x[i][j] for i in range(self.n)) == 1
        
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if pulp.LpStatus[model.status] == 'Optimal':
            x_sol = [[int(x[i][j].varValue) for j in range(self.n)] for i in range(self.n)]
            return x_sol, pulp.value(model.objective)
        else:
            return None, None

    def solve_risk_averse(self):
        model = pulp.LpProblem("CVaR_SAA", pulp.LpMinimize)
        
        x = pulp.LpVariable.dicts("x", (range(self.n), range(self.n)), cat='Binary')
        t = pulp.LpVariable("t")
        z = pulp.LpVariable.dicts("z", range(self.k), lowBound=0)
        
        coeff = 1.0 / ((1.0 - self.alpha) * self.k)
        obj = t + coeff * pulp.lpSum(z[k] for k in range(self.k))
        model += obj
        
        for i in range(self.n):
            model += pulp.lpSum(x[i][j] for j in range(self.n)) == 1
        for j in range(self.n):
            model += pulp.lpSum(x[i][j] for i in range(self.n)) == 1
        
        for k in range(self.k):
            cost_s = pulp.lpSum(self.cost_samples[k][i, j] * x[i][j]
                                for i in range(self.n) for j in range(self.n))
            model += z[k] >= cost_s - t
        
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if pulp.LpStatus[model.status] == 'Optimal':
            x_sol = [[int(x[i][j].varValue) for j in range(self.n)] for i in range(self.n)]
            return x_sol, pulp.value(model.objective)
        else:
            return None, None