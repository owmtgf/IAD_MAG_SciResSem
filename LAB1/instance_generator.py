import numpy as np


class InstanceGenerator:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    # Robust: generate single instance
    def generate_robust_instance(self, n):
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
    def generate_robust_batch(self, n, num_instances=100):
        instances = []
        for _ in range(num_instances):
            c_bar, d = self.generate_robust_instance(n)
            instances.append((c_bar, d))
        return instances

    # Stochastic: generate scenarios
    def generate_stochastic_instance(self, n, k=30):
        """
        Returns:
            scenarios: list of k cost matrices
        """
        # base matrix
        mean = self.rng.integers(10, 50, size=(n, n))

        scenarios = []
        for _ in range(k):
            # noise (from normal distribution)
            noise = self.rng.normal(loc=0, scale=5, size=(n, n))
            c = mean + noise

            # guarantee non-negativity
            c = np.clip(c, 1, None)
            scenarios.append(c)
        return scenarios

    # Stochastic: batch
    def generate_stochastic_batch(self, n, num_instances=100, k=30):
        instances = []
        for _ in range(num_instances):
            scenarios = self.generate_stochastic_instance(n, k)
            instances.append(scenarios)
        return instances
