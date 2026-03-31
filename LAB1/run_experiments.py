from instance_generator import RobustInstanceGenerator, StochasticInstanceGenerator
from solvers import RobustAssignment, StochasticAssignment


if __name__ == "__main__":

    # robust
    gen = RobustInstanceGenerator(seed=42)
    instances = gen.generate_batch(n=10, num_instances=100)

    for c_bar, d in instances:
        model = RobustAssignment(c_bar, d, Gamma=3)
        sol, obj = model.solve()
        print(obj)

    # stochastic
    gen = StochasticInstanceGenerator(seed=42)
    instances = gen.generate_batch(n=10, num_instances=100, k=30)

    for cost_matrix in instances:
        model = StochasticAssignment(cost_matrix)
        print("RN SAA solution:")
        sol, obj = model.solve_risk_neutral()
        print(obj)
        print("RA SAA solution:")
        sol, obj = model.solve_risk_averse()
        print(obj)