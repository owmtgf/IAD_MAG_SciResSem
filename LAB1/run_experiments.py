import time
import json
from collections import defaultdict

import numpy as np
import gurobipy as gp
from gurobipy import GurobiError
from tqdm import tqdm

from instance_generator import RobustInstanceGenerator, StochasticInstanceGenerator
from solvers import RobustAssignment, StochasticAssignment
from visualization import ExperimentVisualizer
from metrics import nominal_cost, out_of_sample_cost, stochastic_metrics

GUROBI_CONF_PATH = "./gurobi_conf.json"


def set_environ(environ: gp.Env, conf_path: str = GUROBI_CONF_PATH):
    with open(conf_path, "r") as cfp:
        configuration = json.load(cfp)
    
    access_id = configuration["environ"].get("WLSAccessID")
    secret = configuration["environ"].get("WLSSecret")
    licence_id = configuration["environ"].get("LicenseID")

    print(f"NOTE: Got environment configuration parameters:\nWLSAccessID: {access_id},\nWLSSecret: {secret},\nLicenseID: {licence_id}\n")
    if all([access_id, secret, licence_id]):
        print(f"NOTE: found existing Gurobi licence")
        environ.setParam('WLSAccessID', access_id)
        environ.setParam('WLSSecret', secret)
        environ.setParam('LicenseID', licence_id)
    else:
        print(f"WARNING: some of the environment configuration parameters are None - using free Gurobi licence: you will not be able to solve large problems!")


def run_robust_experiments(n=10, num_instances=100, gamma_values=[1, 2, 3, 4, 5, 6], n_oos_scenarios=1000, seed: int = 42):
    """Запуск экспериментов для робастной оптимизации"""
    rng = np.random.default_rng(seed)
    
    gen = RobustInstanceGenerator(seed=42)
    instances = gen.generate_batch(n=n, num_instances=num_instances)

    env = gp.Env(empty=True)  # Initialize an empty environment
    # Set license parameters using the environment
    set_environ(env)
    # Now, initialize the environment
    env.start()
    
    # Сбор результатов для разных Gamma
    results_by_gamma = {}
    for Gamma in gamma_values:
        solve_times = []
        in_sample_list = []
        nominal_list = []
        oos_list = []

        for (c_bar, d) in instances:
            start_time = time.perf_counter()
            model = RobustAssignment(c_bar, d, Gamma, env)
            sol, obj = model.solve()
            solve_time = time.perf_counter() - start_time

            solve_times.append(solve_time)

            # In-sample cost (worst-case objective from the solver)
            in_sample_list.append(obj)

            # Nominal cost (z = 0, no uncertainty)
            nom = nominal_cost(sol, c_bar)
            nominal_list.append(nom)

            # Out-of-sample cost (random uncertainty realisation)
            oos_mean, _ = out_of_sample_cost(
                sol, c_bar, d, n_oos_scenarios, rng,
            )
            oos_list.append(oos_mean)

        results_by_gamma[Gamma] = {
            "in_sample": in_sample_list,
            "nominal": nominal_list,
            "oos": oos_list, 
            "in_sample_mean": float(np.mean(in_sample_list)),
            "in_sample_std": float(np.std(in_sample_list)),
            "nominal_mean": float(np.mean(nominal_list)),
            "nominal_std": float(np.std(nominal_list)),
            "oos_mean": float(np.mean(oos_list)),
            "oos_std": float(np.std(oos_list)),
            "solve_time_mean": float(np.mean(solve_times)),
            "solve_time_std": float(np.std(solve_times)),
        }

        print(f"  Gamma={Gamma}: Mean={results_by_gamma[Gamma]['in_sample_mean']:.2f}, "
              f"Std={results_by_gamma[Gamma]['in_sample_std']:.2f}, "
              f"Avg Time={np.mean(solve_times):.3f}s")

    env.close()
    return results_by_gamma

def run_stochastic_experiments(n=10, num_instances=100, k=30, 
                               alpha_values=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95], n_oos_scenarios=1000):
    """Запуск экспериментов для стохастической оптимизации"""
    gen = StochasticInstanceGenerator(seed=42)
    instances = gen.generate_batch(n=n, num_instances=num_instances, k=k)

    env = gp.Env(empty=True)  # Initialize an empty environment
    # Set license parameters using the environment
    set_environ(env)
    # Now, initialize the environment
    env.start()
    
    # Сбор результатов для Risk-Neutral
    print("\nSolving Risk-Neutral problems...")
    rn_results = {
        "mean_list": [],
        "cvar_list": [],
        "solve_times": [],
    }
    for scenarios in instances:
        start_time = time.time()
        model = StochasticAssignment(scenarios)
        sol, obj = model.solve_risk_neutral(env)
        solve_time = time.time() - start_time
        rn_results['solve_times'].append(solve_time)
        start_rn = time.time()
        # generate OUT-OF-SAMPLE scenarios
        test_scenarios = gen.generate_instance(n, k=n_oos_scenarios)
        # compute scenario costs
        sol_array = np.array(sol)
        scenario_costs = np.tensordot(test_scenarios, sol_array, axes=([1,2],[0,1]))
        # evaluate
        metrics = stochastic_metrics(
            scenario_costs,
            alpha=alpha_values[-1]  # use highest alpha for risk eval
        )

        rn_results["mean_list"].append(metrics["mean"])
        rn_results["cvar_list"].append(metrics[f"cvar_{alpha_values[-1]:.2f}"])
        rn_results.setdefault("all_costs", []).extend(scenario_costs)

        end_rn = time.time()
        # print(f"end_rn - start_rn ={end_rn - start_rn}")

    rn_summary = {
        "mean": np.mean(rn_results["mean_list"]),
        "std": np.std(rn_results["mean_list"]),
        "cvar": np.mean(rn_results["cvar_list"]),
        "solve_time_mean": np.mean(rn_results["solve_times"]),
    }

    print(f"RN: mean={rn_summary['mean']:.2f}, "
          f"std={rn_summary['std']:.2f}, "
          f"CVaR={rn_summary['cvar']:.2f}")
    
    # Сбор результатов для Risk-Averse с разными alpha
    ra_results_by_alpha = {}
    for alpha in alpha_values:
        print(f"\nSolving Risk-Averse problems with α={alpha}...")
        mean_list = []
        cvar_list = []
        solve_times = []
        all_costs = []
        
        for i, scenarios in enumerate(instances):
            while True:
                try:
                    start_time = time.time()
                    model = StochasticAssignment(scenarios, alpha=alpha)
                    sol, obj = model.solve_risk_averse(env)
                    solve_time = time.time() - start_time
                    err = None
                except GurobiError as e:
                    err = e
                    print(e)
                    print(f"WARNING: lost connection with host! Retry in 20 seconds.")
                    for t in tqdm(range(20), desc="Waiting for retry"):
                        time.sleep(1)                
                if not err:
                    break
  
            solve_times.append(solve_time)
            start_ra = time.time()
            # OUT-OF-SAMPLE
            test_scenarios = gen.generate_instance(n, k=n_oos_scenarios)
            scenario_costs = [
                np.sum(sol * s) for s in test_scenarios
            ]
            metrics = stochastic_metrics(
                scenario_costs,
                alpha=alpha
            )

            mean_list.append(metrics["mean"])
            cvar_list.append(metrics[f"cvar_{alpha:.2f}"])
            all_costs.extend(scenario_costs)

            end_ra = time.time()
            # print(f"end_ra - start_ra ={end_ra - start_ra}")

        ra_results_by_alpha[alpha] = {
            "mean_list": mean_list,
            "cvar_list": cvar_list,
            "solve_times": solve_times,
            "mean": np.mean(mean_list),
            "std": np.std(mean_list),
            "cvar": np.mean(cvar_list),
            "solve_time_mean": np.mean(solve_times),
            "all_costs": all_costs,
        }

        print(f"  α={alpha}: mean={ra_results_by_alpha[alpha]['mean']:.2f}, "
              f"std={ra_results_by_alpha[alpha]['std']:.2f}, "
              f"CVaR={ra_results_by_alpha[alpha]['cvar']:.2f}")

    env.close()

    return rn_results, ra_results_by_alpha


def run_scaling_experiments(sizes=[5, 10, 15, 20, 25], 
                            num_instances_per_size=20):
    """Анализ масштабирования для разных размеров задач"""
    # Для робастной оптимизации
    robust_scaling_results = []
    env = gp.Env(empty=True)  # Initialize an empty environment
    # Set license parameters using the environment
    set_environ(env)
    # Now, initialize the environment
    env.start()

    for n in sizes:
        print(f"\nTesting size n={n}...")
        gen = RobustInstanceGenerator(seed=42)
        instances = gen.generate_batch(n=n, num_instances=num_instances_per_size)
        
        Gamma = 3  # фиксированное значение Gamma
        
        for idx, (c_bar, d) in enumerate(instances):
            start_time = time.time()
            model = RobustAssignment(c_bar, d, Gamma, env)
            sol, obj = model.solve()
            solve_time = time.time() - start_time
            
            robust_scaling_results.append({
                'n': n,
                'obj': obj,
                'time': solve_time,
                'type': 'robust'
            })
            
            if idx % 5 == 0:
                print(f"  Processed {idx+1}/{num_instances_per_size}")
        
        avg_time = np.mean([r['time'] for r in robust_scaling_results if r['n'] == n])
        avg_obj = np.mean([r['obj'] for r in robust_scaling_results if r['n'] == n])
        print(f"  n={n}: Avg Time={avg_time:.3f}s, Avg Obj={avg_obj:.2f}")
    
    # Для стохастической оптимизации
    stochastic_scaling_results = []
    
    for n in sizes:
        print(f"\nTesting size n={n}...")
        gen = StochasticInstanceGenerator(seed=42)
        instances = gen.generate_batch(n=n, num_instances=num_instances_per_size, k=30)
        
        for idx, scenarios in enumerate(instances):
            # Risk-Neutral
            start_time = time.time()
            model = StochasticAssignment(scenarios)
            sol, obj = model.solve_risk_neutral(env)
            solve_time = time.time() - start_time
            
            stochastic_scaling_results.append({
                'n': n,
                'obj': obj,
                'time': solve_time,
                'type': 'risk_neutral'
            })
            
            # Risk-Averse
            start_time = time.time()
            model = StochasticAssignment(scenarios, alpha=0.9)
            sol, obj = model.solve_risk_averse(env)
            solve_time = time.time() - start_time
            
            stochastic_scaling_results.append({
                'n': n,
                'obj': obj,
                'time': solve_time,
                'type': 'risk_averse'
            })
        
        rn_avg_time = np.mean([r['time'] for r in stochastic_scaling_results 
                              if r['n'] == n and r['type'] == 'risk_neutral'])
        ra_avg_time = np.mean([r['time'] for r in stochastic_scaling_results 
                              if r['n'] == n and r['type'] == 'risk_averse'])
        print(f"  n={n}: RN Time={rn_avg_time:.3f}s, RA Time={ra_avg_time:.3f}s")
    
    return robust_scaling_results, stochastic_scaling_results


def run_out_of_sample_validation(n=10, num_train=50, num_test=1000, k=30):
    """Out-of-sample валидация"""
    print("\n" + "="*80)
    print("OUT-OF-SAMPLE VALIDATION")
    print("="*80)
    
    env = gp.Env(empty=True)  # Initialize an empty environment
    # Set license parameters using the environment
    set_environ(env)
    # Now, initialize the environment
    env.start()

    gen = StochasticInstanceGenerator(seed=42)
    
    # Генерируем тренировочные и тестовые сценарии
    train_instances = gen.generate_batch(n=n, num_instances=num_train, k=k)
    test_scenarios = []
    for _ in range(num_test):
        test_scenarios.append(gen.generate_instance(n=n, k=1)[0])
    
    rn_costs = []
    ra_costs = []
    
    print(f"Training on {num_train} instances...")
    for idx, train_scenarios in enumerate(train_instances):
        if idx % 10 == 0:
            print(f"  Training instance {idx+1}/{num_train}")
        
        # Обучаем модели
        model_rn = StochasticAssignment(train_scenarios)
        sol_rn, _ = model_rn.solve_risk_neutral(env)
        
        model_ra = StochasticAssignment(train_scenarios, alpha=0.9)
        sol_ra, _ = model_ra.solve_risk_averse(env)
        
        # Оцениваем на тестовых сценариях
        rn_costs_instance = []
        ra_costs_instance = []
        
        for test_scenario in test_scenarios:
            # Вычисляем стоимость для risk-neutral решения
            rn_cost = np.sum(sol_rn * test_scenario)
            rn_costs_instance.append(rn_cost)
            
            # Вычисляем стоимость для risk-averse решения
            ra_cost = np.sum(sol_ra * test_scenario)
            ra_costs_instance.append(ra_cost)
        
        rn_costs.extend(rn_costs_instance)
        ra_costs.extend(ra_costs_instance)
    
    print(f"Out-of-sample evaluation on {num_test} scenarios per instance")
    print(f"Risk-Neutral: Mean={np.mean(rn_costs):.2f}, Std={np.std(rn_costs):.2f}, "
          f"90th percentile={np.percentile(rn_costs, 90):.2f}")
    print(f"Risk-Averse: Mean={np.mean(ra_costs):.2f}, Std={np.std(ra_costs):.2f}, "
          f"90th percentile={np.percentile(ra_costs, 90):.2f}")
    
    return rn_costs, ra_costs

if __name__ == "__main__":
    # Инициализируем визуализатор
    visualizer = ExperimentVisualizer()
    
    # 1. Робастная оптимизация: анализ влияния Gamma
    robust_results = run_robust_experiments(
        n=10, 
        num_instances=50,  # уменьшаем для быстродействия
        gamma_values=[1, 2, 3, 4, 5, 6]
    )
    
    # Визуализация робастных результатов
    gamma_values = list(robust_results.keys())
    objectives_list = [robust_results[g]['objectives'] for g in gamma_values]
    visualizer.plot_robust_gamma_analysis(gamma_values, objectives_list)
    
    # 2. Стохастическая оптимизация: сравнение RN vs RA
    rn_results, ra_results_by_alpha = run_stochastic_experiments(
        n=10,
        num_instances=50,
        k=30,
        alpha_values=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    )
    
    # Визуализация сравнения RN и RA
    alpha_90 = 0.9
    visualizer.plot_stochastic_comparison(
        rn_results['objectives'],
        ra_results_by_alpha[alpha_90]['objectives'],
        alpha=alpha_90
    )
    
    # Визуализация чувствительности к alpha
    alpha_values = list(ra_results_by_alpha.keys())
    objectives_by_alpha = [ra_results_by_alpha[a]['objectives'] for a in alpha_values]
    visualizer.plot_alpha_sensitivity(alpha_values, objectives_by_alpha)
    
    # 3. Анализ масштабирования
    robust_scaling, stochastic_scaling = run_scaling_experiments(
        sizes=[5, 8, 10, 12, 15],
        num_instances_per_size=10
    )
    
    # Визуализация масштабирования
    # Группируем результаты по размеру
    robust_by_size = defaultdict(list)
    for r in robust_scaling:
        robust_by_size[r['n']].append(r)
    
    stochastic_by_size = defaultdict(list)
    for s in stochastic_scaling:
        stochastic_by_size[s['n']].append(s)
    
    # Создаем данные для визуализации
    scaling_data = []
    for n, results in robust_by_size.items():
        for r in results:
            scaling_data.append(r)
    for n, results in stochastic_by_size.items():
        for s in results:
            scaling_data.append(s)
    
    # Преобразуем в формат для plot_scaling_analysis
    experiments_2d = []
    sizes = sorted(set([r['n'] for r in robust_scaling] + [s['n'] for s in stochastic_scaling]))
    
    for size in sizes:
        size_experiments = []
        size_experiments.extend([r for r in robust_scaling if r['n'] == size])
        size_experiments.extend([s for s in stochastic_scaling if s['n'] == size])
        experiments_2d.append(size_experiments)
    
    visualizer.plot_scaling_analysis(experiments_2d, size_metric='n', 
                                    target_metrics=['time', 'obj'])
    
    # 4. Out-of-sample валидация
    rn_costs, ra_costs = run_out_of_sample_validation(
        n=10,
        num_train=30,
        num_test=500,
        k=20
    )
    
    visualizer.plot_out_of_sample_performance(rn_costs, ra_costs, n_scenarios=500)
    
    # 5. Создание сводной таблицы
    summary_data = {
        'Robust (Gamma=3)': {
            'Mean': robust_results[3]['mean_obj'],
            'Std': robust_results[3]['std_obj'],
            'Time': np.mean(robust_results[3]['solve_times'])
        },
        'Risk-Neutral': {
            'Mean': rn_results['mean_obj'],
            'Std': rn_results['std_obj'],
            'Time': np.mean(rn_results['solve_times'])
        },
        f'Risk-Averse (α=0.9)': {
            'Mean': ra_results_by_alpha[0.9]['mean_obj'],
            'Std': ra_results_by_alpha[0.9]['std_obj'],
            'Time': np.mean(ra_results_by_alpha[0.9]['solve_times'])
        }
    }
    
    visualizer.create_summary_table(summary_data)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)