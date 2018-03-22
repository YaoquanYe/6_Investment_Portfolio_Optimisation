import numpy as np
from timeit import default_timer as timer

class DataSet:
    def __init__(self, N, K, epsilon = 0.0, delta = 1.0):
        """Creates a random instance. Replace this with an actual instance read in from a text file."""
        self.mu = np.random.rand(N)
        self.sigma = np.random.rand(N,N)
        # This code assumes one epsilon and one delta for all assets.
        self.epsilon = epsilon
        self.delta = delta
        if K * epsilon > 1.0:
            print("Epsilon is too large")
            raise ValueError
        if K * delta < 1.0:
            print("Delta is too small")
            raise ValueError
        self.F = 1.0 - K * epsilon

        
class Solution:
    def __init__(self, N, K):
        """Creates a random solution"""
        self.Q = np.random.permutation(N)[:K]
        self.s = np.random.rand(K)
        self.w = np.zeros(N)
        self.obj1 = np.nan
        self.obj2 = np.nan


def evaluate(solution, l, dataset, best_solutions):
    """solution is a Solution, l is an integer index into Lambda and Best_value_found, 
    dataset is a DataSet, best_solutions is a list of improved Solution(s)"""
    improved = False
    w = solution.w
    L = solution.s.sum()
    w_temp = dataset.epsilon + solution.s * dataset.F / L
    is_too_large = (solution.s > dataset.delta)
    while is_too_large.sum() > 0:
        R = solution.Q[is_too_large]
        is_not_too_large = np.logical_not(is_too_large)
        L = solution.s[is_not_too_large].sum()
        F_temp = 1.0 - (dataset.epsilon * is_not_too_large.sum() + dataset.delta * is_too_large.sum())
        w_temp = dataset.epsilon + solution.s * F_temp / L
        w_temp[R] = dataset.delta
    # Re-init the w values to zero
    w[:] = 0
    # Assign the new values
    w[solution.Q] = w_temp
    if np.any(w < 0.0) or not np.isclose(w.sum(), 1) or np.sum(w > 0.0) != K:
        if np.any(w < 0.0):
            print("There's a negative proportion in solution: " + str(w))
        elif not np.isclose(w.sum(), 1):
            print("Proportions don't sum up to 1 (" + str(w.sum()) + ") in solution: " + str(w))
        else:
            print("More than " + str(K) + " assets selected (" + str(np.sum(w > 0.0)) + ") in solution: " + str(w))
        raise ValueError
    # CoVar = sum of (w * transpose of w * sigma)
    solution.obj1 = np.sum((w * w.reshape((w.shape[0], 1))) * dataset.sigma)
    solution.obj2 = np.sum(w * dataset.mu)
    f = Lambda[l] * solution.obj1 - (1 - Lambda[l]) * solution.obj2
    solution.s = w - dataset.epsilon
    if f < Best_value_found[l]:
        improved = True
        Best_value_found[l] = f
        best_solutions.append(solution)
    return improved, f


start = timer()
# Total number of assets.
N = 32
# Number of assets to include in the portfolio
K = 10

dataset = DataSet(N, K, epsilon = 0.01)
# An array of weights to weight the two objectives..
Lambda = np.array([0.5])
# Best value found for each weight.
Best_value_found = np.array(Lambda * [np.inf])
# List of best solutions ever found.
Best_solutions = []
maxEvals = 1000 * N
# This is NOT a random search. It just generates random solutions and evaluates
# them but it doesn't keep track of the best.
for i in range(maxEvals):
    s = Solution(N,K)
    improved, f = evaluate(s, 0, dataset, Best_solutions)
    if improved:
        print(str(i) + ": " + str(f) + " (" + str(s.obj1) + ", " + str(s.obj2) +")")
    elif i % (1000*N) == 0:
        print(str(i) + ": " + str(f))
end = timer()
print("# Time for " + str(maxEvals) + " : " + str(end - start))

    