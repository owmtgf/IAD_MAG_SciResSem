from instance_generator import InstanceGenerator
from solvers import RobustAssignment, StochasticAssignment


if __name__ == "__main__":

    # robust
    gen = InstanceGenerator(seed=42)
    instances = gen.generate_robust_batch(n=10, num_instances=100)

    for c_bar, d in instances:
        model = RobustAssignment(c_bar, d, Gamma=3)
        sol, obj = model.solve()
        print(obj)

    # stochastic
    gen = InstanceGenerator(seed=42)
    instances = gen.generate_stochastic_batch(n=10, num_instances=100, k=30)

    for scenarios in instances:
        # scenarios = [c^1, c^2, ..., c^30]
        model = StochasticAssignment()
        sol, obj = model.solve()
        print(obj)
        