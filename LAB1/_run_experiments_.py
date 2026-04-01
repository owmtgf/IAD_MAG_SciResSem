from instance_generator import RobustInstanceGenerator, StochasticInstanceGenerator
from solvers import RobustAssignment, StochasticAssignment
from visualization_ import ExperimentVisualizer
import numpy as np
import time
from collections import defaultdict

def run_robust_experiments(n=10, num_instances=100, gamma_values=[1, 2, 3, 4, 5, 6]):
    """Запуск экспериментов для робастной оптимизации"""
    print("\n" + "="*80)
    print("ROBUST OPTIMIZATION EXPERIMENTS")
    print("="*80)
    
    gen = RobustInstanceGenerator(seed=42)
    instances = gen.generate_batch(n=n, num_instances=num_instances)
    
    # Сбор результатов для разных Gamma
    results_by_gamma = {}
    
    for Gamma in gamma_values:
        print(f"\nTesting Gamma = {Gamma}...")
        objectives = []
        solve_times = []
        
        for idx, (c_bar, d) in enumerate(instances):
            if idx % 20 == 0:
                print(f"  Processing instance {idx+1}/{num_instances}")
            
            start_time = time.time()
            model = RobustAssignment(c_bar, d, Gamma)
            sol, obj = model.solve()
            solve_time = time.time() - start_time
            
            objectives.append(obj)
            solve_times.append(solve_time)
        
        results_by_gamma[Gamma] = {
            'objectives': objectives,
            'solve_times': solve_times,
            'mean_obj': np.mean(objectives),
            'std_obj': np.std(objectives)
        }
        
        print(f"  Gamma={Gamma}: Mean={results_by_gamma[Gamma]['mean_obj']:.2f}, "
              f"Std={results_by_gamma[Gamma]['std_obj']:.2f}, "
              f"Avg Time={np.mean(solve_times):.3f}s")
    
    return results_by_gamma

def run_stochastic_experiments(n=10, num_instances=100, k=30, 
                               alpha_values=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95]):
    """Запуск экспериментов для стохастической оптимизации"""
    print("\n" + "="*80)
    print("STOCHASTIC OPTIMIZATION EXPERIMENTS")
    print("="*80)
    
    gen = StochasticInstanceGenerator(seed=42)
    instances = gen.generate_batch(n=n, num_instances=num_instances, k=k)
    
    # Сбор результатов для Risk-Neutral
    print("\nSolving Risk-Neutral problems...")
    rn_results = {'objectives': [], 'solve_times': []}
    
    for idx, scenarios in enumerate(instances):
        if idx % 20 == 0:
            print(f"  Processing instance {idx+1}/{num_instances}")
        
        start_time = time.time()
        model = StochasticAssignment(scenarios)
        sol, obj = model.solve_risk_neutral()
        solve_time = time.time() - start_time
        
        rn_results['objectives'].append(obj)
        rn_results['solve_times'].append(solve_time)
    
    rn_results['mean_obj'] = np.mean(rn_results['objectives'])
    rn_results['std_obj'] = np.std(rn_results['objectives'])
    
    print(f"Risk-Neutral: Mean={rn_results['mean_obj']:.2f}, "
          f"Std={rn_results['std_obj']:.2f}, "
          f"Avg Time={np.mean(rn_results['solve_times']):.3f}s")
    
    # Сбор результатов для Risk-Averse с разными alpha
    results_by_alpha = {}
    
    for alpha in alpha_values:
        print(f"\nSolving Risk-Averse problems with α={alpha}...")
        objectives = []
        solve_times = []
        
        for idx, scenarios in enumerate(instances):
            if idx % 20 == 0:
                print(f"  Processing instance {idx+1}/{num_instances}")
            
            start_time = time.time()
            model = StochasticAssignment(scenarios, alpha=alpha)
            sol, obj = model.solve_risk_averse()
            solve_time = time.time() - start_time
            
            objectives.append(obj)
            solve_times.append(solve_time)
        
        results_by_alpha[alpha] = {
            'objectives': objectives,
            'solve_times': solve_times,
            'mean_obj': np.mean(objectives),
            'std_obj': np.std(objectives)
        }
        
        print(f"  α={alpha}: Mean={results_by_alpha[alpha]['mean_obj']:.2f}, "
              f"Std={results_by_alpha[alpha]['std_obj']:.2f}, "
              f"Avg Time={np.mean(solve_times):.3f}s")
    
    return rn_results, results_by_alpha

def run_scaling_experiments(sizes=[5, 10, 15, 20, 25], 
                            num_instances_per_size=20):
    """Анализ масштабирования для разных размеров задач"""
    print("\n" + "="*80)
    print("SCALING ANALYSIS EXPERIMENTS")
    print("="*80)
    
    # Для робастной оптимизации
    robust_scaling_results = []
    
    for n in sizes:
        print(f"\nTesting size n={n}...")
        gen = RobustInstanceGenerator(seed=42)
        instances = gen.generate_batch(n=n, num_instances=num_instances_per_size)
        
        Gamma = 3  # фиксированное значение Gamma
        
        for idx, (c_bar, d) in enumerate(instances):
            start_time = time.time()
            model = RobustAssignment(c_bar, d, Gamma)
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
            sol, obj = model.solve_risk_neutral()
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
            sol, obj = model.solve_risk_averse()
            solve_time = time.time() - start_time
            
            stochastic_scaling_results.append({
                'n': n,
                'obj': obj,
                'time': solve_time,
                'type': 'risk_averse'
            })
            
            if idx % 5 == 0:
                print(f"  Processed {idx+1}/{num_instances_per_size}")
        
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
        sol_rn, _ = model_rn.solve_risk_neutral()
        
        model_ra = StochasticAssignment(train_scenarios, alpha=0.9)
        sol_ra, _ = model_ra.solve_risk_averse()
        
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