from pprint import pprint
import numpy as np
import gurobipy as gp
from gurobipy import GRB


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

        self.model = gp.Model("robust_assignment")
        self.model.setParam("OutputFlag", 0)

        self._build_model()

    def _build_model(self):
        n = self.n

        # VARIABLES
        self.x = self.model.addVars(n, n, vtype=GRB.BINARY, name="x")
        self.mu = self.model.addVar(lb=0, name="mu")
        self.nu = self.model.addVars(n, n, lb=0, name="nu")

        # OBJECTIVE
        self.model.setObjective(
            gp.quicksum(self.c_bar[i, j] * self.x[i, j] for i in range(n) for j in range(n))
            + self.Gamma * self.mu
            + gp.quicksum(self.nu[i, j] for i in range(n) for j in range(n)),
            GRB.MINIMIZE
        )

        # ASSIGNMENT CONSTRAINTS
        for i in range(n):
            self.model.addConstr(
                gp.quicksum(self.x[i, j] for j in range(n)) == 1
            )

        for j in range(n):
            self.model.addConstr(
                gp.quicksum(self.x[i, j] for i in range(n)) == 1
            )

        # ROBUST CONSTRAINTS
        for i in range(n):
            for j in range(n):
                self.model.addConstr(
                    self.nu[i, j] >= self.d[i, j] * self.x[i, j] - self.mu
                )

    def solve(self):
        self.model.optimize()

        x_sol = [[self.x[i, j].X for j in range(self.n)] for i in range(self.n)]
        obj = self.model.ObjVal

        return x_sol, obj
    

class StochasticAssignment:
    def __init__(self, cost_samples, alpha=0.9, time_limit=120):
        """
        cost_samples : list of k matrices (n x n)
        alpha : CVaR confidence level
        time_limit : seconds for Gurobi
        """
        self.cost_samples = cost_samples
        self.n = cost_samples[0].shape[0]
        self.k = len(cost_samples)
        self.alpha = alpha
        self.time_limit = time_limit

    def solve_risk_neutral(self):
        avg_cost = np.mean(self.cost_samples, axis=0)
        model = gp.Model("RN_SAA")
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", self.time_limit)

        x = model.addVars(self.n, self.n, vtype=GRB.BINARY, name="x")
        obj = gp.quicksum(avg_cost[i, j] * x[i, j] for i in range(self.n) for j in range(self.n))
        model.setObjective(obj, GRB.MINIMIZE)

        for i in range(self.n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(self.n)) == 1)
        for j in range(self.n):
            model.addConstr(gp.quicksum(x[i, j] for i in range(self.n)) == 1)

        model.optimize()
        x_sol = [[int(round(x[i, j].X)) for j in range(self.n)] for i in range(self.n)]
        return x_sol, model.ObjVal

    def solve_risk_averse(self):
        model = gp.Model("CVaR_SAA")
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", self.time_limit)

        x = model.addVars(self.n, self.n, vtype=GRB.BINARY, name="x")
        t = model.addVar(lb=-GRB.INFINITY, name="t")
        z = model.addVars(self.k, lb=0, name="u")

        coeff = 1.0 / ((1.0 - self.alpha) * self.k)
        obj = t + coeff * gp.quicksum(z[k] for k in range(self.k))
        model.setObjective(obj, GRB.MINIMIZE)

        for i in range(self.n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(self.n)) == 1)
        for j in range(self.n):
            model.addConstr(gp.quicksum(x[i, j] for i in range(self.n)) == 1)

        for k in range(self.k):
            cost_s = gp.quicksum(self.cost_samples[k][i, j] * x[i, j]
                                 for i in range(self.n) for j in range(self.n))
            model.addConstr(z[k] >= cost_s - t)

        model.optimize()
        x_sol = [[int(round(x[i, j].X)) for j in range(self.n)] for i in range(self.n)]
        return x_sol, model.ObjVal


# example
if __name__ == "__main__":
    n = 5
    c_bar = np.random.randint(1, 20, (n, n))
    d = np.random.randint(1, 5, (n, n))
    Gamma = 2

    model = RobustAssignment(c_bar, d, Gamma)
    sol, obj = model.solve()

    pprint(sol)
    print(f"{obj=}")
