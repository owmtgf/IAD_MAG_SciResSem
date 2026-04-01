import numpy as np


def stochastic_metrics(
    scenario_costs,
    alpha=0.95,
    threshold=None,
    baseline_mean=None,
    baseline_cvar=None,
    ddof=1
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

    # Сравнение с baseline по risk-neutral критерию
    if baseline_mean is not None:
        result["rn_gap_vs_baseline"] = mean_cost - float(baseline_mean)

    # Сравнение с baseline по risk-averse критерию
    if baseline_cvar is not None:
        result["ra_gap_vs_baseline"] = cvar_alpha - float(baseline_cvar)

    return result