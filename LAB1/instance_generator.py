import numpy as np


class RobustInstanceGenerator:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    # Robust: generate single instance
    def generate_instance(self, n):
        """
        Returns:
            c_bar : nominal costs (n x n)
            d     : deviations (n x n)
        """
        # nominal costs
        c_bar = self.rng.integers(10, 50, size=(n, n))

        # deviations (10–30% of nominal costs here)
        d = (0.1 + 0.2 * self.rng.random((n, n))) * c_bar

        return c_bar, d

    # Robust: batch generation
    def generate_batch(self, n, num_instances=100):
        instances = []
        for _ in range(num_instances):
            c_bar, d = self.generate_instance(n)
            instances.append((c_bar, d))
        return instances

    
class StochasticInstanceGenerator(RobustInstanceGenerator):  
    def __init__(self, seed=None):
        super().__init__(seed)
        self.mu = lambda i,j: 5 + i + 2*j
        self.sigma = lambda i,j: 1 + 0.1*(i + j)

    def generate_instance(self, n, k=30):
        """
        Generate k cost matrices of size n x n, sampled from 
        N(mu_ij, sigma_ij^2) for each entry.
        """
        scenarios = []

        # Base mu and sigma matrices
        mu_matrix = np.array([[self.mu(i,j) for j in range(n)] for i in range(n)])
        sigma_matrix = np.array([[self.sigma(i,j) for j in range(n)] for i in range(n)])

        for _ in range(k):
            # Sample from normal for each entry
            noise = self.rng.normal(loc=0, scale=sigma_matrix, size=(n,n))
            c = mu_matrix + noise

            # Ensure non-negative costs
            c = np.clip(c, 0, None)
            scenarios.append(c)

        return scenarios
    
    def generate_batch(self, n, num_instances=100, k=30):
        instances = []
        for _ in range(num_instances):
            scenarios = self.generate_instance(n, k)
            instances.append(scenarios)
        return instances
