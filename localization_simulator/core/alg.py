from time import perf_counter
from .inf import fim, isotropic
import numpy as np

def dummyFim(x,p,d,sol,iso,var):
    return sum([p[s][0] for s in sol])

def greedy(k,x,p,d,iso,var):
    solution = set()

    for _ in range(k):
        inf_max = 0
        j_best = None
        
        for j in range(len(p)):
            inf_j = fim(x,p,d,solution.union({j}),iso,var)
            print(f"Set:{solution.union({j})} and inf_j:{inf_j} ")

            if inf_j > inf_max:
                inf_max = inf_j
                j_best = j

        solution = solution.union({j_best})

    return solution




if __name__ == "__main__":

    x = np.array([(5,5),(7,7)])
    p = np.array([(8,5), (6,8), (6,14)])
    d = np.array([[3.1,3.12,9.15],[2.12,1.5,7.17]]) 
    iso = isotropic(2,0.5)

    print(greedy(2,x,p,d,iso,0.5))




