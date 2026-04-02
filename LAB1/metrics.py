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


def stochastic_metrics(
    scenario_costs,
    alpha=0.95,
    threshold=None,
    ddof=1,
):
    """
    Вычисляет основные стохастические метрики для случайной стоимости решения.

    Возвращает
    ----------
    dict
        Словарь с метриками:
        - mean
        - variance
        - std
        - min
        - max
        - var_alpha
        - cvar_alpha
        - prob_exceed_threshold (если threshold задан)
        - rn_gap_vs_baseline (если baseline_mean задан)
        - ra_gap_vs_baseline (если baseline_cvar задан)
    """
    costs = np.asarray(scenario_costs, dtype=float)

    if costs.ndim != 1:
        raise ValueError("scenario_costs должен быть одномерным массивом.")
    if len(costs) == 0:
        raise ValueError("scenario_costs не должен быть пустым.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha должен лежать в интервале (0, 1).")

    # Базовые статистики
    mean_cost = float(np.mean(costs))
    var_cost = float(np.var(costs, ddof=ddof)) if len(costs) > ddof else 0.0
    std_cost = float(np.std(costs, ddof=ddof)) if len(costs) > ddof else 0.0
    min_cost = float(np.min(costs))
    max_cost = float(np.max(costs))

    # VaR_alpha для задачи минимизации стоимости:
    # это alpha-квантиль затрат
    var_alpha = float(np.quantile(costs, alpha, method="linear"))

    # CVaR_alpha = среднее по худшему хвосту: Z >= VaR_alpha
    tail = costs[costs >= var_alpha]
    cvar_alpha = float(np.mean(tail)) if len(tail) > 0 else var_alpha

    result = {
        "mean": mean_cost,
        "variance": var_cost,
        "std": std_cost,
        "min": min_cost,
        "max": max_cost,
        f"var_{alpha:.2f}": var_alpha,
        f"cvar_{alpha:.2f}": cvar_alpha,
    }

    # Вероятность превышения заданного порога
    if threshold is not None:
        prob_exceed = float(np.mean(costs > threshold))
        result["threshold"] = float(threshold)
        result["prob_exceed_threshold"] = prob_exceed

    return result