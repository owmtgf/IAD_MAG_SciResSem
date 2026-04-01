import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import pandas as pd

class ExperimentVisualizer:
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        plt.style.use(style)
        sns.set_palette("husl")
        
    def plot_robust_gamma_analysis(self, gamma_values: List[float], 
                                   objectives: List[List[float]]):
        """
        Анализ влияния Gamma на робастные решения
        
        Args:
            gamma_values: список значений Gamma
            objectives: список списков objective values для каждого Gamma
                       [ [obj_for_gamma1], [obj_for_gamma2], ... ]
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # График 1: Средние значения с доверительными интервалами
        means = [np.mean(objs) for objs in objectives]
        stds = [np.std(objs) for objs in objectives]
        
        ax1 = axes[0]
        ax1.errorbar(gamma_values, means, yerr=stds, 
                    marker='o', capsize=5, capthick=2, 
                    linewidth=2, markersize=8, 
                    label='Mean objective')
        ax1.fill_between(gamma_values, 
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2)
        ax1.set_xlabel('Gamma (Uncertainty Budget)', fontsize=12)
        ax1.set_ylabel('Objective Value', fontsize=12)
        ax1.set_title('Impact of Gamma on Robust Solution Cost', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # График 2: Box plots для каждого Gamma
        ax2 = axes[1]
        data_to_plot = [objs for objs in objectives]
        bp = ax2.boxplot(data_to_plot, positions=gamma_values, 
                        widths=0.6, patch_artist=True,
                        showmeans=True, meanline=True)
        
        # Раскрашиваем box plots
        for box in bp['boxes']:
            box.set_facecolor('lightblue')
            box.set_alpha(0.7)
            
        ax2.set_xlabel('Gamma', fontsize=12)
        ax2.set_ylabel('Objective Value', fontsize=12)
        ax2.set_title('Distribution of Solutions for Different Gamma', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_stochastic_comparison(self, risk_neutral_objs: List[float],
                                  risk_averse_objs: List[float],
                                  alpha: float = 0.9):
        """
        Сравнение risk-neutral и risk-averse подходов
        
        Args:
            risk_neutral_objs: список objective values для risk-neutral
            risk_averse_objs: список objective values для risk-averse
            alpha: уровень CVaR
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # График 1: Box plot сравнения
        ax1 = axes[0]
        data = [risk_neutral_objs, risk_averse_objs]
        labels = ['Risk-Neutral', f'Risk-Averse (α={alpha})']
        
        bp = ax1.boxplot(data, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True)
        
        # Раскрашиваем
        colors = ['lightgreen', 'lightcoral']
        for box, color in zip(bp['boxes'], colors):
            box.set_facecolor(color)
            box.set_alpha(0.7)
            
        ax1.set_ylabel('Objective Value', fontsize=12)
        ax1.set_title('Comparison: Risk-Neutral vs Risk-Averse', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Добавляем статистику
        stats_text = f"Risk-Neutral:\nMean: {np.mean(risk_neutral_objs):.2f}\nStd: {np.std(risk_neutral_objs):.2f}\n\n"
        stats_text += f"Risk-Averse (α={alpha}):\nMean: {np.mean(risk_averse_objs):.2f}\nStd: {np.std(risk_averse_objs):.2f}\n"
        stats_text += f"Difference: {np.mean(risk_averse_objs) - np.mean(risk_neutral_objs):.2f}"
        
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # График 2: Гистограммы распределения
        ax2 = axes[1]
        ax2.hist(risk_neutral_objs, bins=20, alpha=0.5, 
                label='Risk-Neutral', color='green', density=True)
        ax2.hist(risk_averse_objs, bins=20, alpha=0.5, 
                label=f'Risk-Averse (α={alpha})', color='red', density=True)
        
        ax2.axvline(np.mean(risk_neutral_objs), color='green', 
                   linestyle='--', linewidth=2, label='Mean RN')
        ax2.axvline(np.mean(risk_averse_objs), color='red', 
                   linestyle='--', linewidth=2, label='Mean RA')
        
        ax2.set_xlabel('Objective Value', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Distribution of Solution Costs', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_alpha_sensitivity(self, alpha_values: List[float],
                              objectives: List[List[float]]):
        """
        Анализ чувствительности к alpha в CVaR
        
        Args:
            alpha_values: список значений alpha
            objectives: список списков objective values для каждого alpha
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        means = [np.mean(objs) for objs in objectives]
        stds = [np.std(objs) for objs in objectives]
        
        # Основная линия с доверительным интервалом
        ax.plot(alpha_values, means, 'b-o', linewidth=2, markersize=8, 
               label='Mean objective')
        ax.fill_between(alpha_values, 
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2, color='blue')
        
        # Добавляем box plots для каждого alpha
        for i, (alpha, objs) in enumerate(zip(alpha_values, objectives)):
            # Добавляем jitter для точек
            x_jitter = np.random.normal(alpha, 0.01, len(objs))
            ax.scatter(x_jitter, objs, alpha=0.3, s=30, color='gray')
            
        ax.set_xlabel('α (CVaR Confidence Level)', fontsize=12)
        ax.set_ylabel('Objective Value', fontsize=12)
        ax.set_title('Sensitivity Analysis: Impact of α on Risk-Averse Solutions', 
                    fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Добавляем аннотацию
        ax.text(0.05, 0.95, f'Cost increase: {means[-1] - means[0]:.2f}',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
    def plot_out_of_sample_performance(self, 
                                      risk_neutral_costs: List[float],
                                      risk_averse_costs: List[float],
                                      n_scenarios: int = 1000):
        """
        Оценка out-of-sample производительности
        
        Args:
            risk_neutral_costs: costs for risk-neutral solution on test scenarios
            risk_averse_costs: costs for risk-averse solution on test scenarios
            n_scenarios: количество out-of-sample сценариев
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # График 1: ECDF (Empirical Cumulative Distribution Function)
        ax1 = axes[0]
        
        # Sort data for ECDF
        rn_sorted = np.sort(risk_neutral_costs)
        ra_sorted = np.sort(risk_averse_costs)
        
        # Calculate ECDF
        rn_ecdf = np.arange(1, len(rn_sorted) + 1) / len(rn_sorted)
        ra_ecdf = np.arange(1, len(ra_sorted) + 1) / len(ra_sorted)
        
        ax1.plot(rn_sorted, rn_ecdf, 'g-', linewidth=2, 
                label='Risk-Neutral')
        ax1.plot(ra_sorted, ra_ecdf, 'r-', linewidth=2, 
                label=f'Risk-Averse (α=0.9)')
        
        # Highlight CVaR region
        ax1.axvline(np.percentile(risk_neutral_costs, 90), 
                   color='green', linestyle='--', alpha=0.5)
        ax1.axvline(np.percentile(risk_averse_costs, 90), 
                   color='red', linestyle='--', alpha=0.5)
        
        ax1.set_xlabel('Cost', fontsize=12)
        ax1.set_ylabel('Cumulative Probability', fontsize=12)
        ax1.set_title('Out-of-Sample Performance: ECDF', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: Tail comparison (90th percentile and above)
        ax2 = axes[1]
        
        # Calculate tail metrics
        rn_tail = [c for c in risk_neutral_costs if c > np.percentile(risk_neutral_costs, 90)]
        ra_tail = [c for c in risk_averse_costs if c > np.percentile(risk_averse_costs, 90)]
        
        tail_data = [rn_tail, ra_tail]
        tail_labels = ['Risk-Neutral\n(90th percentile tail)', 
                      'Risk-Averse\n(90th percentile tail)']
        
        bp = ax2.boxplot(tail_data, labels=tail_labels, patch_artist=True)
        
        for box, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
            box.set_facecolor(color)
            box.set_alpha(0.7)
            
        ax2.set_ylabel('Cost', fontsize=12)
        ax2.set_title('Tail Comparison (90th Percentile and Above)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"Tail Mean:\nRN: {np.mean(rn_tail):.2f}\nRA: {np.mean(ra_tail):.2f}\n\n"
        stats_text += f"Tail Std:\nRN: {np.std(rn_tail):.2f}\nRA: {np.std(ra_tail):.2f}\n\n"
        stats_text += f"Tail 95th:\nRN: {np.percentile(rn_tail, 95):.2f}\nRA: {np.percentile(ra_tail, 95):.2f}"
        
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()

    def create_summary_table(self, results_dict: Dict[str, Dict[str, float]]):
        """
        Создание сводной таблицы результатов
        """
        print("\n" + "="*80)
        print("SUMMARY OF EXPERIMENT RESULTS")
        print("="*80)
        
        # Определяем все возможные метрики
        all_metrics = set()
        for stats in results_dict.values():
            all_metrics.update(stats.keys())
        
        # Выводим заголовок
        header = f"{'Method':<25}"
        for metric in sorted(all_metrics):
            header += f" {metric:<12}"
        print(header)
        print("-"*80)
        
        # Выводим данные
        for method, stats in results_dict.items():
            row = f"{method:<25}"
            for metric in sorted(all_metrics):
                value = stats.get(metric, 'N/A')
                if isinstance(value, (int, float)):
                    if metric.lower() in ['time', 'solve_time', 'solvetime']:
                        row += f" {value:.3f}{' ':<8}"
                    else:
                        row += f" {value:.2f}{' ':<9}"
                else:
                    row += f" {str(value):<12}"
            print(row)
        
        print("="*80)

    def plot_scaling_analysis(self, experiments_2d: List[List[Dict]], 
                          size_metric: str = 'n',
                          target_metrics: List[str] = None):
        '''
        Анализ масштабирования: зависимость метрик от размера задачи
        
        Args:
            experiments_2d: двумерный список экспериментов 
                        [ [exp1, exp2, ...], [exp1, exp2, ...], ... ]
                        где каждый exp - словарь с метриками
            size_metric: название метрики размера (например, 'n')
            target_metrics: список метрик для анализа (по умолчанию ['time', 'obj'])
        '''
        if target_metrics is None:
            target_metrics = ['time', 'obj']
        
        n_sizes = len(experiments_2d)
        sizes = []
        metrics_data = {metric: [] for metric in target_metrics}
        
        # Собираем данные
        for size_idx, experiments in enumerate(experiments_2d):
            if not experiments:
                continue
                
            # Определяем размер (берем из первого эксперимента)
            size_val = experiments[0].get(size_metric, size_idx)
            sizes.append(size_val)
            
            for metric in target_metrics:
                values = [exp.get(metric, np.nan) for exp in experiments 
                        if exp.get(metric) is not None]
                metrics_data[metric].append(values)
        
        if not sizes:
            print(f"Metric '{size_metric}' not found in experiments")
            return
        
        n_metrics = len(target_metrics)
        fig, axes = plt.subplots(2, n_metrics, figsize=(5*n_metrics, 10))
        
        if n_metrics == 1:
            axes = axes.reshape(-1, 1)
        
        for i, metric in enumerate(target_metrics):
            # График 1: Средние значения с доверительными интервалами
            ax1 = axes[0, i]
            means = [np.mean(vals) for vals in metrics_data[metric]]
            stds = [np.std(vals) for vals in metrics_data[metric]]
            
            ax1.errorbar(sizes, means, yerr=stds, 
                        marker='o', capsize=5, capthick=2,
                        linewidth=2, markersize=8)
            ax1.fill_between(sizes, 
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            alpha=0.2)
            
            # Добавляем линию тренда
            z = np.polyfit(sizes, means, 2)
            p = np.poly1d(z)
            x_trend = np.linspace(min(sizes), max(sizes), 100)
            ax1.plot(x_trend, p(x_trend), 'r--', alpha=0.8, 
                    label=f'Trend: {z[0]:.2e}x² + {z[1]:.2e}x + {z[2]:.2e}')
            
            ax1.set_xlabel(f'{size_metric.capitalize()} (Problem Size)', fontsize=12)
            ax1.set_ylabel(metric.capitalize(), fontsize=12)
            ax1.set_title(f'{metric.capitalize()} Scaling Analysis', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=8)
            
            # График 2: Box plots для каждого размера
            ax2 = axes[1, i]
            bp = ax2.boxplot(metrics_data[metric], positions=sizes, 
                            widths=0.6, patch_artist=True,
                            showmeans=True, meanline=True)
            
            # Раскрашиваем
            for box in bp['boxes']:
                box.set_facecolor('lightblue')
                box.set_alpha(0.7)
            
            ax2.set_xlabel(f'{size_metric.capitalize()}', fontsize=12)
            ax2.set_ylabel(metric.capitalize(), fontsize=12)
            ax2.set_title(f'{metric.capitalize()} Distribution by Size', fontsize=14)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    def plot_parameter_sensitivity_grid(self, experiments_dict: Dict[str, List[Dict]],
                                    param_name: str,
                                    target_metrics: List[str] = None):
        '''
        Сетка графиков чувствительности к параметру
        
        Args:
            experiments_dict: словарь {param_value: [list of experiment dicts]}
            param_name: название параметра
            target_metrics: список метрик для анализа
        '''
        if target_metrics is None:
            target_metrics = ['time', 'obj']
        
        param_values = sorted(experiments_dict.keys())
        n_metrics = len(target_metrics)
        
        fig, axes = plt.subplots(n_metrics, 2, figsize=(14, 4*n_metrics))
        
        if n_metrics == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric in enumerate(target_metrics):
            # Собираем данные
            data_by_param = []
            for param in param_values:
                values = [exp.get(metric, np.nan) for exp in experiments_dict[param] 
                        if exp.get(metric) is not None]
                if values:
                    data_by_param.append(values)
                else:
                    data_by_param.append([])
            
            # График 1: Box plots
            ax1 = axes[i, 0]
            bp = ax1.boxplot(data_by_param, labels=[str(p) for p in param_values],
                            patch_artist=True, showmeans=True, meanline=True)
            
            # Раскрашиваем
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(param_values)))
            for box, color in zip(bp['boxes'], colors):
                box.set_facecolor(color)
                box.set_alpha(0.7)
            
            ax1.set_xlabel(param_name.capitalize(), fontsize=12)
            ax1.set_ylabel(metric.capitalize(), fontsize=12)
            ax1.set_title(f'{metric.capitalize()} vs {param_name.capitalize()}', fontsize=14)
            ax1.grid(True, alpha=0.3, axis='y')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Добавляем статистику
            stats_text = ""
            for j, (param, values) in enumerate(zip(param_values, data_by_param)):
                if values:
                    stats_text += f"{param}:\n  Mean: {np.mean(values):.2f}\n"
                    stats_text += f"  Std: {np.std(values):.2f}\n"
                    if j < len(param_values)-1:
                        stats_text += "\n"
            
            ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # График 2: Линия тренда
            ax2 = axes[i, 1]
            means = [np.mean(vals) if vals else np.nan for vals in data_by_param]
            stds = [np.std(vals) if vals else np.nan for vals in data_by_param]
            
            # Убираем NaN
            valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
            valid_params = [param_values[i] for i in valid_indices]
            valid_means = [means[i] for i in valid_indices]
            valid_stds = [stds[i] for i in valid_indices]
            
            if len(valid_params) > 1:
                ax2.errorbar(valid_params, valid_means, yerr=valid_stds,
                            marker='o', capsize=5, capthick=2,
                            linewidth=2, markersize=8)
                ax2.fill_between(valid_params,
                                np.array(valid_means) - np.array(valid_stds),
                                np.array(valid_means) + np.array(valid_stds),
                                alpha=0.2)
                
                # Добавляем регрессию
                z = np.polyfit(valid_params, valid_means, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(valid_params), max(valid_params), 100)
                ax2.plot(x_trend, p(x_trend), 'r--', alpha=0.8,
                        label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
                ax2.legend()
            
            ax2.set_xlabel(param_name.capitalize(), fontsize=12)
            ax2.set_ylabel(metric.capitalize(), fontsize=12)
            ax2.set_title(f'{metric.capitalize()} Trend Analysis', fontsize=14)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()    

# Пример использования:
if __name__ == "__main__":
    # Генерация демо-данных
    np.random.seed(42)
    
    visualizer = ExperimentVisualizer()
    
    # Демо для робастного анализа
    gamma_vals = [1, 2, 3, 4, 5, 6]
    objectives_gamma = [np.random.normal(100 + g*5, 10, 100) for g in gamma_vals]
    visualizer.plot_robust_gamma_analysis(gamma_vals, objectives_gamma)
    
    # Демо для стохастического сравнения
    rn_objs = np.random.normal(100, 15, 100)
    ra_objs = np.random.normal(115, 8, 100)
    visualizer.plot_stochastic_comparison(rn_objs, ra_objs, alpha=0.9)
    
    # Демо для alpha sensitivity
    alpha_vals = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    objectives_alpha = [np.random.normal(100 + a*30, 10, 100) for a in alpha_vals]
    visualizer.plot_alpha_sensitivity(alpha_vals, objectives_alpha)
    
    # Демо для out-of-sample
    rn_test = np.random.normal(100, 20, 1000)
    ra_test = np.random.normal(112, 10, 1000)
    visualizer.plot_out_of_sample_performance(rn_test, ra_test)