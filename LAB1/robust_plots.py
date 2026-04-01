"""
Simple visualization for test results
"""

import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('robust_test_results.csv')
    print(f"Loaded {len(df)} records")
    print(df)
    
    if len(df) > 0:
        plt.figure(figsize=(10, 6))
        
        # Plot in-sample vs out-of-sample
        gammas = df['Gamma'].values
        in_sample = df['in_sample_cost'].values
        out_sample = df['out_of_sample_cost'].values
        
        plt.plot(gammas, in_sample, 'o-', label='In-sample', linewidth=2, markersize=8)
        plt.plot(gammas, out_sample, 's-', label='Out-of-sample', linewidth=2, markersize=8)
        
        plt.xlabel('Uncertainty Budget Γ')
        plt.ylabel('Cost')
        plt.title('Robust Optimization Results (Test)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('robust_test_plot.png', dpi=300)
        plt.show()
        
        print("\n✓ Plot saved to robust_test_plot.png")
        
except FileNotFoundError:
    print("No results file found. Run robust_experiments_complete.py first.")
except Exception as e:
    print(f"Error: {e}")