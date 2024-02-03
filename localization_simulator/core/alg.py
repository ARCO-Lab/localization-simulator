from time import perf_counter
from .inf import fim, isotropic
import numpy as np
import pandas as pd
from itertools import combinations

def brute(k,x,p,d,iso,var):
    start = perf_counter()
    candidates = list(combinations(range(len(p)),k))

    inf_max = 0
    solution = set()

    for c in candidates:
        inf_j = fim(x,p,d,c,iso,var)
        # print(f"Set:{c} and inf_j:{inf_j} ")

        if inf_j > inf_max:
            inf_max = inf_j
            solution = c
    
    stop = perf_counter()
    print(stop-start)
    print(inf_max)
    return solution

def greedy(k,x,p,d,iso,var):
    start = perf_counter()
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

    stop = perf_counter()
    print(stop-start)
    print(inf_max)
    return solution


if __name__ == "__main__":


    k = 4

    p = np.column_stack((np.random.randint(0,100,30),np.random.randint(0,100,30)))
    x = np.array([(1.5,1.5),(2,2.5),(2.5,1.5)])
    # p = np.array([(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)])
    iso = isotropic(2,2)
    variance = 0.25
    anc = [0,1,2,3,4,5,6,7,8]

    def addNoise(x, p, variance):
        d = np.array([[np.linalg.norm(i - j) for j in p] for i in x])
        noise = np.random.normal(0, np.sqrt(variance), size=np.shape(d))
        return d + noise

    d = addNoise(x,p,variance)

    print(brute(k,x,p,d,iso,variance))
    print(greedy(k,x,p,d,iso,variance))



    # def sanity(anc):
    #     ancComb = [set(x) for i in range(len(anc) + 1) for x in combinations(anc, i)]
    #     infTable = {str(comb): [] for comb in ancComb}
    #     for _ in range(10):
    #         d = addNoise(x,p,variance)
    #         for s in ancComb:
    #             i = fim(x,p,d,s,iso,variance)
    #             infTable[str(s)].append(i)

    #     df = pd.DataFrame(infTable)
    #     df.index += 1
    #     mean = df.mean()
    #     quartiles = df.quantile([0.25,0.5,0.75])
    #     summary = pd.DataFrame({
    #         'Mean': mean,
    #         '25th Percentile':quartiles.loc[0.25],
    #         '50th Percentile':quartiles.loc[0.5],
    #         '75th Percentile':quartiles.loc[0.75]
    #     })
    #     return df, summary

    # print(sanity(anc)[0])
    # print(sanity(anc)[1])

    # print(greedy(2,x,p,d,iso,variance))




