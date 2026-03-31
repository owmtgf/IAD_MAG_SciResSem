"""
metrics.py
----------
Evaluation metrics for the robust assignment problem.

After solving the robust problem we need to judge whether the obtained solution
is actually *good* — not just in the worst case, but also under nominal
conditions and under random uncertainty.  Three complementary cost metrics
answer this question, plus a summary statistic called the Price of Robustness.

────────────────────────────────────────────────────────────────────
METRICS OVERVIEW
────────────────────────────────────────────────────────────────────

Let x*(Γ) be the solution found by the robust solver with budget Γ.
For a given cost matrix c_bar and deviation matrix d, the costs are:

    c_ij  =  c̄_ij  +  d_ij · z_ij,    z_ij ∈ [0, 1],  Σ z_ij ≤ Γ

1. NOMINAL COST
   Cost when no uncertainty occurs at all (z = 0 everywhere):

       nominal(x*) = Σ_{i,j} c̄_ij · x*_ij

   This is the "best case" — what we pay if the world behaves exactly as
   expected.  A robust solution may sacrifice some nominal performance in
   exchange for protection against bad scenarios.

2. IN-SAMPLE COST  (= Robust objective value)
   Worst-case cost within the uncertainty budget Γ.  This is the value the
   solver minimises and returns directly.  Analytically it equals:

       in_sample(x*, Γ) = nominal(x*) + Σ_{top-Γ} d_ij · x*_ij

   where the "top-Γ" sum picks the Γ largest deviation contributions
   d_ij among the selected assignments (x*_ij = 1).  The adversary sets
   z_ij = 1 for exactly those assignments — the most damaging ones.

   A higher Γ → larger in-sample cost (we guard against more deviations).

3. OUT-OF-SAMPLE COST
   Expected cost when uncertainty is realised *randomly* rather than
   adversarially.  Each z_ij is drawn i.i.d. from Uniform[0, 1] in each
   scenario, giving a Monte-Carlo estimate:

       oos(x*) = (1/S) · Σ_s  Σ_{i,j} (c̄_ij + d_ij · z^s_ij) · x*_ij

   This models the real world: deviations happen unpredictably, not at the
   worst possible moment.  The out-of-sample cost typically lies between the
   nominal cost (z = 0) and the in-sample cost (z = worst case).

   KEY INSIGHT: if the in-sample cost (worst case) is much larger than the
   out-of-sample cost (average case), the solution is being overly cautious
   — the robust model is "paying" for scenarios that rarely occur.

4. PRICE OF ROBUSTNESS (PoR)
   How much *more* do we pay under nominal conditions compared with the
   purely deterministic solution x*(0)?

       PoR(Γ) = [ nominal(x*(Γ)) − nominal(x*(0)) ] / nominal(x*(0)) × 100 %

   PoR = 0 % means the robust solution is as cheap as the deterministic one
   under nominal conditions — protection was "free".
   PoR > 0 % reveals the cost of conservatism: the robust solver chose a
   somewhat more expensive nominal assignment because it is better protected.

   As Γ increases, PoR generally increases: we pay more nominally in exchange
   for a lower worst-case cost.

────────────────────────────────────────────────────────────────────
FUNCTIONS
────────────────────────────────────────────────────────────────────

  nominal_cost(x_sol, c_bar)
      → cost under nominal parameters

  worst_case_cost(x_sol, c_bar, d, Gamma)
      → analytical worst-case cost (should match solver objective)

  out_of_sample_cost(x_sol, c_bar, d, n_scenarios, rng)
      → Monte-Carlo estimate of expected cost under random uncertainty

  compute_robust_metrics(instances, gammas, n_oos_scenarios, seed)
      → task point 5: aggregate metrics for each Γ over all instances

  compute_solve_times(gamma, n_values, num_instances, seed)
      → task point 6: average solve time as a function of n
"""

import time
import numpy as np
from solvers import RobustAssignment


# ============================================================================
# Low-level helpers
# ============================================================================

def nominal_cost(x_sol, c_bar):
    """
    Cost of solution x under the nominal (expected) costs c_bar,
    i.e. when no uncertainty occurs (z_ij = 0 for all i, j).

    Parameters
    ----------
    x_sol : list[list[float]] or np.ndarray  — binary assignment matrix (n × n)
    c_bar : np.ndarray (n × n)               — nominal cost matrix

    Returns
    -------
    float
    """
    return float(np.sum(np.array(x_sol) * c_bar))


def worst_case_cost(x_sol, c_bar, d, Gamma):
    """
    Analytical worst-case cost of solution x under uncertainty budget Γ.

    For a fixed binary solution x*, the adversary solves:
        max_{z: z_ij ∈ [0,1], Σ z_ij ≤ Γ}  Σ_{i,j} d_ij · x*_ij · z_ij

    This LP has a simple greedy optimum: set z_ij = 1 for the Γ entries
    with the largest d_ij · x*_ij (i.e. the Γ selected assignments with
    the highest deviation), and z_ij = 0 everywhere else.

    Result:
        worst_case(x*, Γ) = nominal(x*) + sum of top-Γ values of {d_ij : x*_ij = 1}

    This value should coincide with the objective returned by the solver
    (they are equal by LP strong duality).

    Parameters
    ----------
    x_sol : list[list[float]] or np.ndarray  — binary assignment matrix (n × n)
    c_bar : np.ndarray (n × n)               — nominal cost matrix
    d     : np.ndarray (n × n)               — maximum deviation matrix
    Gamma : int or float                      — uncertainty budget

    Returns
    -------
    float
    """
    x = np.array(x_sol)

    # d_ij * x_ij: nonzero only for selected assignments
    contributions = (d * x).flatten()

    # Sort descending; adversary picks the top-Gamma contributions
    contributions_sorted = np.sort(contributions)[::-1]

    gamma_floor = int(Gamma)
    gamma_frac  = Gamma - gamma_floor

    worst_addition = float(np.sum(contributions_sorted[:gamma_floor]))
    if gamma_frac > 0 and gamma_floor < len(contributions_sorted):
        worst_addition += gamma_frac * float(contributions_sorted[gamma_floor])

    return float(np.sum(c_bar * x)) + worst_addition


def out_of_sample_cost(x_sol, c_bar, d, n_scenarios=1000, rng=None):
    """
    Monte-Carlo estimate of the expected cost of solution x when uncertainty
    is realised *randomly* (not adversarially).

    Each z_ij is sampled i.i.d. from Uniform[0, 1] independently across all
    entries and all scenarios.  This replaces the adversary with "nature":
    deviations occur without any coordination or worst-case intent.

    Parameters
    ----------
    x_sol       : list[list[float]] or np.ndarray  — binary assignment matrix
    c_bar       : np.ndarray (n × n)               — nominal cost matrix
    d           : np.ndarray (n × n)               — maximum deviation matrix
    n_scenarios : int   — number of Monte-Carlo samples (default: 1000)
    rng         : np.random.Generator or None

    Returns
    -------
    mean : float  — average realised cost over all scenarios
    std  : float  — standard deviation of realised costs
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.array(x_sol)
    n = c_bar.shape[0]

    # z[s, i, j] ~ Uniform[0, 1]  for s = 1..n_scenarios
    z = rng.uniform(0.0, 1.0, size=(n_scenarios, n, n))

    # Realised cost matrix for each scenario
    c_realised = c_bar[np.newaxis, :, :] + d[np.newaxis, :, :] * z  # (S, n, n)

    # Scalar cost per scenario: Σ_{i,j} c^s_ij · x_ij
    costs = np.sum(c_realised * x[np.newaxis, :, :], axis=(1, 2))    # (S,)

    return float(np.mean(costs)), float(np.std(costs))


# ============================================================================
# Main metrics computation  —  task point 5
# ============================================================================

def compute_robust_metrics(instances, gammas, n_oos_scenarios=1000, seed=42):
    """
    For each value of Γ, solve all instances and compute aggregate metrics.

    Implements task point 5:
        "For fixed n, compute the average in-sample and out-of-sample
         performance of robust solutions (with standard deviations),
         depending on the budget Γ."

    Workflow per instance per Γ:
      1. Solve the robust problem  →  x*(Γ), in-sample objective
      2. Compute nominal cost of x*(Γ)
      3. Estimate out-of-sample cost via Monte-Carlo
      4. Compute Price of Robustness vs. the deterministic baseline x*(0)

    Parameters
    ----------
    instances       : list of (c_bar, d) pairs  —  each is an n × n np.ndarray
    gammas          : list[int | float]           —  Γ values to evaluate
    n_oos_scenarios : int    — number of Monte-Carlo scenarios per instance
    seed            : int    — random seed for reproducibility

    Returns
    -------
    dict  mapping each Γ  →  dict with keys:
        "in_sample_mean"   / "in_sample_std"   — robust (worst-case) objective
        "nominal_mean"     / "nominal_std"      — cost under nominal parameters
        "oos_mean"         / "oos_std"          — expected cost under random unc.
        "por_mean"         / "por_std"          — Price of Robustness (%)
        "solve_time_mean"  / "solve_time_std"   — wall-clock solve time (s)
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Deterministic baseline (Γ = 0) — solved once, reused for all Γ.
    # We need nominal_cost(x*(0)) to compute the Price of Robustness.
    # ------------------------------------------------------------------
    baseline_nominal = np.empty(len(instances))
    for idx, (c_bar, d) in enumerate(instances):
        model = RobustAssignment(c_bar, d, Gamma=0)
        x_det, _ = model.solve()
        baseline_nominal[idx] = nominal_cost(x_det, c_bar)

    results = {}

    for gamma in gammas:
        in_sample_list = []
        nominal_list   = []
        oos_list       = []
        por_list       = []
        time_list      = []

        for idx, (c_bar, d) in enumerate(instances):

            # 1. Solve the robust problem
            t0 = time.perf_counter()
            model = RobustAssignment(c_bar, d, Gamma=gamma)
            x_sol, in_sample_obj = model.solve()
            time_list.append(time.perf_counter() - t0)

            # 2. In-sample cost (worst-case objective from the solver)
            in_sample_list.append(in_sample_obj)

            # 3. Nominal cost (z = 0, no uncertainty)
            nom = nominal_cost(x_sol, c_bar)
            nominal_list.append(nom)

            # 4. Out-of-sample cost (random uncertainty realisation)
            oos_mean, _ = out_of_sample_cost(
                x_sol, c_bar, d, n_oos_scenarios, rng
            )
            oos_list.append(oos_mean)

            # 5. Price of Robustness vs. deterministic baseline
            por = (nom - baseline_nominal[idx]) / baseline_nominal[idx] * 100.0
            por_list.append(por)

        results[gamma] = {
            "in_sample_mean":  float(np.mean(in_sample_list)),
            "in_sample_std":   float(np.std(in_sample_list)),
            "nominal_mean":    float(np.mean(nominal_list)),
            "nominal_std":     float(np.std(nominal_list)),
            "oos_mean":        float(np.mean(oos_list)),
            "oos_std":         float(np.std(oos_list)),
            "por_mean":        float(np.mean(por_list)),
            "por_std":         float(np.std(por_list)),
            "solve_time_mean": float(np.mean(time_list)),
            "solve_time_std":  float(np.std(time_list)),
        }

    return results


# ============================================================================
# Solve time as a function of n  —  task point 6
# ============================================================================

def compute_solve_times(gamma, n_values, num_instances=100, seed=42):
    """
    For a fixed Γ, measure average solve time as a function of problem size n.

    Implements task point 6:
        "For fixed Γ, compute the average solution times as a function of n."

    Parameters
    ----------
    gamma         : int | float   — fixed uncertainty budget
    n_values      : list[int]     — problem sizes (e.g. [5, 10, 15, ..., 50])
    num_instances : int           — number of random instances per n
    seed          : int           — random seed for instance generation

    Returns
    -------
    dict  mapping each n  →  {"mean": float, "std": float}  (times in seconds)
    """
    from instance_generator import RobustInstanceGenerator

    gen = RobustInstanceGenerator(seed=seed)
    results = {}

    for n in n_values:
        instances = gen.generate_batch(n, num_instances)
        times = []
        for c_bar, d in instances:
            t0 = time.perf_counter()
            model = RobustAssignment(c_bar, d, Gamma=gamma)
            model.solve()
            times.append(time.perf_counter() - t0)

        results[n] = {
            "mean": float(np.mean(times)),
            "std":  float(np.std(times)),
        }

    return results


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    from instance_generator import RobustInstanceGenerator

    n = 10
    gammas = [0, 1, 2, 3, 5, 10]
    gen = RobustInstanceGenerator(seed=42)
    instances = gen.generate_batch(n=n, num_instances=20)  # 20 for quick demo

    print(f"Computing robust metrics for n={n}, 20 instances ...\n")
    metrics = compute_robust_metrics(instances, gammas, n_oos_scenarios=500, seed=0)

    header = f"{'Gamma':>6}  {'In-sample':>20}  {'Nominal':>20}  {'OOS':>20}  {'PoR %':>12}  {'Time (s)':>12}"
    print(header)
    print("-" * len(header))
    for gamma, m in metrics.items():
        print(
            f"{gamma:>6}  "
            f"{m['in_sample_mean']:>9.2f} ± {m['in_sample_std']:<8.2f}  "
            f"{m['nominal_mean']:>9.2f} ± {m['nominal_std']:<8.2f}  "
            f"{m['oos_mean']:>9.2f} ± {m['oos_std']:<8.2f}  "
            f"{m['por_mean']:>6.2f} ± {m['por_std']:<4.2f}  "
            f"{m['solve_time_mean']:>6.4f} ± {m['solve_time_std']:.4f}"
        )

    print("\nComputing solve times for Gamma=3, n in [5,10,...,30] ...\n")
    times = compute_solve_times(gamma=3, n_values=list(range(5, 35, 5)), num_instances=20)
    print(f"{'n':>4}  {'Mean time (s)':>14}  {'Std (s)':>10}")
    print("-" * 32)
    for n_val, t in times.items():
        print(f"{n_val:>4}  {t['mean']:>14.4f}  {t['std']:>10.4f}")
