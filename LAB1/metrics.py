import time
import numpy as np
from solvers import RobustAssignment

def nominal_cost(x_sol, c_bar):
    return float(np.sum(np.array(x_sol) * c_bar))


def worst_case_cost(x_sol, c_bar, d, Gamma):
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

def compute_robust_metrics(instances, gammas, n_oos_scenarios=1000, seed=42):
    rng = np.random.default_rng(seed)

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

def compute_solve_times(gamma, n_values, num_instances=100, seed=42):
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
