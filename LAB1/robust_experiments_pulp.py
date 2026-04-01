"""
Robust Optimization Experiments with PuLP - FULL VERSION
"""

import numpy as np
import time
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from instance_generator import RobustInstanceGenerator
from solvers_pulp import RobustAssignment


def main():
    # ПОЛНЫЙ ЭКСПЕРИМЕНТ
    n_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    num_instances = 100
    
    # Адаптивные Gamma значения
    Gamma_dict = {
        5: [0, 2, 4, 6, 8, 10],
        10: [0, 3, 6, 9, 12, 15],
        15: [0, 4, 8, 12, 16, 20],
        20: [0, 5, 10, 15, 20, 25],
        25: [0, 6, 12, 18, 24, 30],
        30: [0, 7, 14, 21, 28, 35],
        35: [0, 8, 16, 24, 32, 40],
        40: [0, 9, 18, 27, 36, 45],
        45: [0, 10, 20, 30, 40, 50],
        50: [0, 10, 20, 30, 40, 50]
    }
    
    print("="*60)
    print("ROBUST OPTIMIZATION WITH PULP - FULL EXPERIMENT")
    print("="*60)
    print(f"n values: {n_values}")
    print(f"Instances per n: {num_instances}")
    print(f"Total problems to solve: {sum(len(Gamma_dict[n]) * num_instances for n in n_values)}")
    print("="*60)
    
    generator = RobustInstanceGenerator(seed=42)
    all_results = []
    
    for n_idx, n in enumerate(n_values):
        print(f"\n{'='*50}")
        print(f"Processing n = {n} ({n_idx+1}/{len(n_values)})")
        print(f"Gamma values: {Gamma_dict[n]}")
        print(f"{'='*50}")
        
        instances = generator.generate_batch(n, num_instances)
        
        for instance_id, (c_bar, d) in enumerate(tqdm(instances, desc=f"n={n}")):
            for Gamma in Gamma_dict[n]:
                start_time = time.time()
                model = RobustAssignment(c_bar, d, Gamma)
                x_sol, in_sample_cost = model.solve()
                solve_time = time.time() - start_time
                
                if x_sol is not None:
                    # Вычисляем out-of-sample
                    assignment = []
                    for i in range(n):
                        for j in range(n):
                            if x_sol[i][j] == 1:
                                assignment.append((i, j))
                    
                    out_total = 0
                    for _ in range(100):  # 100 семплов для out-of-sample
                        u = np.random.uniform(0, 1, size=(n, n))
                        c_real = c_bar + d * u
                        out_total += sum(c_real[i, j] for i, j in assignment)
                    out_sample_cost = out_total / 100
                    
                    all_results.append({
                        'n': n,
                        'instance_id': instance_id,
                        'Gamma': Gamma,
                        'in_sample_cost': in_sample_cost,
                        'out_of_sample_cost': out_sample_cost,
                        'solve_time': solve_time
                    })
        
        # Сохраняем промежуточные результаты после каждого n
        df_checkpoint = pd.DataFrame(all_results)
        df_checkpoint.to_csv(f'robust_results_checkpoint_n_{n}.csv', index=False)
        print(f"  ✓ Checkpoint saved: {len(all_results)} total records")
    
    # Сохраняем полные результаты
    df_full = pd.DataFrame(all_results)
    df_full.to_csv('robust_results_full.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ EXPERIMENT COMPLETED!")
    print(f"Total records: {len(all_results)}")
    print(f"Results saved to robust_results_full.csv")
    print(f"{'='*60}")
    
    # Агрегируем результаты
    print("\nAggregating results...")
    aggregated = df_full.groupby(['n', 'Gamma']).agg({
        'in_sample_cost': ['mean', 'std'],
        'out_of_sample_cost': ['mean', 'std'],
        'solve_time': ['mean', 'std']
    }).reset_index()
    
    aggregated.columns = ['n', 'Gamma', 
                          'in_sample_mean', 'in_sample_std',
                          'out_sample_mean', 'out_sample_std',
                          'time_mean', 'time_std']
    
    aggregated.to_csv('robust_results_aggregated.csv', index=False)
    print("✓ Aggregated results saved to robust_results_aggregated.csv")
    
    # Выводим сводку
    print("\n" + "="*80)
    print("SUMMARY (first few rows):")
    print("="*80)
    print(aggregated.head(20).to_string())
    
    return df_full, aggregated


if __name__ == "__main__":
    df_full, aggregated = main()