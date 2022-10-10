import numpy as np
import numba
import math
import time

ALPH = 0.99
N = 10000
TN = 100.0
AMOUNT_OF_ITERATIONS = 100000000


class Solution(object):
    def __init__(self):
        self.plan: np.array = np.arange(0, N, 1)
        self.energy: int = check_collision(self.plan)


@numba.njit(fastmath=True, cache=True)
def check_collision(plan: np.array) -> int:
    energy: int = 0
    for i in numba.prange(N):
        for j in numba.prange(i+1, N):
            if abs(i - j) == abs(plan[i] - plan[j]):
                energy += 1
    return energy


@numba.njit(fastmath=True, cache=True)
def otjig(working_plan, working_energy, best_plan, best_energy, t):
    new_plan = working_plan.copy()
    new_energy = working_energy

    ind1 = np.random.randint(N)
    ind2 = np.random.randint(N)
    vspom_per = new_plan[ind1]
    new_plan[ind1] = new_plan[ind2]
    new_plan[ind2] = vspom_per
    new_energy = check_collision(new_plan)

    if new_energy < working_energy or math.exp((working_energy - new_energy) / t) > np.random.random(1):
        working_energy = new_energy
        working_plan = new_plan.copy()

        if working_energy < best_energy:
            best_energy = working_energy
            best_plan = working_plan.copy()

    return working_plan.copy(), working_energy, best_plan.copy(), best_energy


def main():
    working_solution = Solution()
    best_solution = Solution()
    t: float = TN
    counter = 0

    print(best_solution.plan)
    print(best_solution.energy)

    for i in range(AMOUNT_OF_ITERATIONS):
        if counter % AMOUNT_OF_ITERATIONS % 1000 == 0 and counter != 0:
            print(best_solution.plan)
            print("Best energy: ", best_solution.energy)
            print("Temperature: ", t)
            print("Iteration: ", counter)

        (working_solution.plan, working_solution.energy,
        best_solution.plan, best_solution.energy) = otjig(working_solution.plan,
                                                           working_solution.energy,
                                                           best_solution.plan,
                                                           best_solution.energy,
                                                           t)
        counter += 1
        t *= ALPH

        if best_solution.energy == 0:
            print(best_solution.plan)
            print("Best energy: ", best_solution.energy)
            print("Temperature: ", t)
            print("Iteration: ", counter)
            break


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print("\n TIME: ", time.perf_counter() - start, ".sec")
