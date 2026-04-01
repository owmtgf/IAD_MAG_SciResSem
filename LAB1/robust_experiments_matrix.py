"""
Robust Optimization Experiments with Matrix Output Format
Returns: results[n][instance_id] = {
    'solve_time': float,
    'in_sample_cost': float,
    'out_of_sample_cost': float,
    'Gamma_values': dict  # для каждого Gamma
}
"""

import numpy as np
import time
import pandas as pd
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from instance_generator import RobustInstanceGenerator
from solvers_pulp import RobustAssignment


class RobustExperimentMatrix:
    """Запускает эксперименты и возвращает результаты в виде матрицы n × instances"""
    
    def __init__(self, seed=42):
        self.generator = RobustInstanceGenerator(seed=seed)
        self.results = {}  # results[n][instance_id] = {...}
        
    def compute_out_of_sample(self, x_sol, c_bar, d, num_samples=100):
        """Вычисляет out-of-sample cost для найденного решения"""
        n = len(c_bar)
        assignment = []
        for i in range(n):
            for j in range(n):
                if x_sol[i][j] == 1:
                    assignment.append((i, j))
        
        total = 0.0
        for _ in range(num_samples):
            u = np.random.uniform(0, 1, size=(n, n))
            c_real = c_bar + d * u
            total += sum(c_real[i, j] for i, j in assignment)
        
        return total / num_samples
    
    def run_single_instance(self, c_bar, d, Gamma_values, instance_id, n):
        """Решает один инстанс для всех Gamma"""
        instance_data = {}
        
        for Gamma in Gamma_values:
            start_time = time.time()
            model = RobustAssignment(c_bar, d, Gamma)
            x_sol, in_sample = model.solve()
            solve_time = time.time() - start_time
            
            if x_sol is not None:
                out_sample = self.compute_out_of_sample(x_sol, c_bar, d)
                
                instance_data[Gamma] = {
                    'in_sample_cost': float(in_sample),
                    'out_of_sample_cost': float(out_sample),
                    'solve_time': solve_time
                }
        
        return instance_data
    
    def run_experiments(self, n_values, Gamma_dict, num_instances=100):
        """
        Запускает эксперименты.
        
        Возвращает:
        results[n][instance_id] = {
            'Gamma_values': {
                Gamma: {
                    'in_sample_cost': float,
                    'out_of_sample_cost': float,
                    'solve_time': float
                }
            }
        }
        """
        print("="*70)
        print("ROBUST OPTIMIZATION EXPERIMENTS")
        print(f"Format: results[n][instance_id] -> metrics for each Gamma")
        print("="*70)
        
        for n in n_values:
            print(f"\n{'='*50}")
            print(f"Processing n = {n}")
            print(f"Gamma values: {Gamma_dict[n]}")
            print(f"Instances: {num_instances}")
            print(f"{'='*50}")
            
            # Инициализируем для этого n
            self.results[n] = {}
            
            # Генерируем инстансы
            instances = self.generator.generate_batch(n, num_instances)
            
            for instance_id, (c_bar, d) in enumerate(tqdm(instances, desc=f"n={n}")):
                # Решаем инстанс для всех Gamma
                instance_data = self.run_single_instance(
                    c_bar, d, Gamma_dict[n], instance_id, n
                )
                self.results[n][instance_id] = {
                    'Gamma_values': instance_data
                }
            
            # Сохраняем промежуточный результат
            self._save_checkpoint(n)
        
        return self.results
    
    def _save_checkpoint(self, n):
        """Сохраняет результаты в JSON"""
        # Конвертируем для JSON
        serializable = {}
        for n_val, instances in self.results.items():
            serializable[str(n_val)] = {}
            for inst_id, data in instances.items():
                serializable[str(n_val)][str(inst_id)] = data
        
        with open(f'robust_results_matrix_n_{n}.json', 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"  ✓ Checkpoint saved: robust_results_matrix_n_{n}.json")
    
    def save_results(self, filename='robust_results_matrix.json'):
        """Сохраняет все результаты в JSON"""
        # Конвертируем для JSON
        serializable = {}
        for n_val, instances in self.results.items():
            serializable[str(n_val)] = {}
            for inst_id, data in instances.items():
                serializable[str(n_val)][str(inst_id)] = data
        
        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\n✓ Results saved to {filename}")
    
    def to_dataframe(self):
        """Конвертирует результаты в плоский DataFrame"""
        rows = []
        for n_val, instances in self.results.items():
            for inst_id, data in instances.items():
                for Gamma, metrics in data['Gamma_values'].items():
                    rows.append({
                        'n': int(n_val),
                        'instance_id': int(inst_id),
                        'Gamma': Gamma,
                        'in_sample_cost': metrics['in_sample_cost'],
                        'out_of_sample_cost': metrics['out_of_sample_cost'],
                        'solve_time': metrics['solve_time']
                    })
        return pd.DataFrame(rows)
    
    def get_aggregated(self):
        """Возвращает агрегированные данные (средние, std)"""
        df = self.to_dataframe()
        
        aggregated = df.groupby(['n', 'Gamma']).agg({
            'in_sample_cost': ['mean', 'std'],
            'out_of_sample_cost': ['mean', 'std'],
            'solve_time': ['mean', 'std']
        }).reset_index()
        
        aggregated.columns = ['n', 'Gamma', 
                              'in_sample_mean', 'in_sample_std',
                              'out_sample_mean', 'out_sample_std',
                              'time_mean', 'time_std']
        
        return aggregated


def define_gamma_values(n_values):
    """Определяет Gamma значения для каждого n"""
    Gamma_dict = {}
    for n in n_values:
        if n <= 15:
            Gamma_dict[n] = [0, 2, 4, 6, 8, 10, 12, n]
        elif n <= 30:
            Gamma_dict[n] = [0, 5, 10, 15, 20, 25, n//2, n]
        else:
            Gamma_dict[n] = [0, 10, 20, 30, 40, n//3, n//2, n]
    return Gamma_dict


def main():
    """Основная функция"""
    
    # Параметры эксперимента
    n_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    num_instances = 100
    
    
    
    Gamma_dict = define_gamma_values(n_values)
    
    print("="*70)
    print("ROBUST OPTIMIZATION EXPERIMENTS")
    print("="*70)
    print(f"n values: {n_values}")
    print(f"Instances per n: {num_instances}")
    print(f"Output format: results[n][instance_id] -> metrics per Gamma")
    print("="*70)
    
    # Запускаем эксперименты
    runner = RobustExperimentMatrix(seed=42)
    results = runner.run_experiments(n_values, Gamma_dict, num_instances)
    
    # Сохраняем в разных форматах
    
    # 1. JSON с матричной структурой
    runner.save_results('robust_results_matrix.json')
    
    # 2. Плоский CSV для анализа
    df = runner.to_dataframe()
    df.to_csv('robust_results_flat.csv', index=False)
    print(f"✓ Flat CSV saved: {len(df)} records")
    
    # 3. Агрегированные данные
    aggregated = runner.get_aggregated()
    aggregated.to_csv('robust_results_aggregated.csv', index=False)
    print(f"✓ Aggregated results saved")
    
    # Выводим структуру результатов
    print("\n" + "="*70)
    print("RESULTS STRUCTURE:")
    print("="*70)
    print("results[n][instance_id] = {")
    print("    'Gamma_values': {")
    print("        Gamma_value: {")
    print("            'in_sample_cost': float,")
    print("            'out_of_sample_cost': float,")
    print("            'solve_time': float")
    print("        }")
    print("    }")
    print("}")
    
    # Пример вывода для первого n
    first_n = n_values[0]
    print(f"\nПример для n={first_n}:")
    print(f"  {len(results[first_n])} инстансов")
    first_instance = list(results[first_n].keys())[0]
    print(f"  Инстанс {first_instance}:")
    for Gamma, metrics in results[first_n][first_instance]['Gamma_values'].items():
        print(f"    Γ={Gamma}: in={metrics['in_sample_cost']:.2f}, "
              f"out={metrics['out_of_sample_cost']:.2f}, "
              f"time={metrics['solve_time']:.3f}s")
    
    return results, df, aggregated


if __name__ == "__main__":
    results, df, aggregated = main()