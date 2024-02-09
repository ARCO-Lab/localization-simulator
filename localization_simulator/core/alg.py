from time import perf_counter
from .inf import fim, isotropic
# from ..utils.result import Parameters
import numpy as np
import pandas as pd
from itertools import combinations
import cma

def brute(param):
    start = perf_counter()
    candidates = list(combinations(range(len(param.p)),param.k))

    inf_max = float('-inf')
    solution = set()

    for c in candidates:
        inf_j = fim(param.x,param.p,param.d,c,param.iso_var,param.sensor_var)
        # print(f"Set:{c} and inf_j:{inf_j} ")

        if inf_j > inf_max:
            inf_max = inf_j
            solution = c
    
    stop = perf_counter()
    print("time: " + str(stop-start))
    print("max inf gain: " + str(inf_max))
    return [solution,inf_max,0,stop-start]

def greedy(param):
    start = perf_counter()
    solution = set()

    for _ in range(param.k):
        inf_max = float('-inf')
        j_best = None
        
        for j in range(len(param.p)):
            inf_j = fim(param.x,param.p,param.d,solution.union({j}),param.iso_var,param.sensor_var)
            # print(f"Set:{solution.union({j})} and inf_j:{inf_j} ")

            if inf_j > inf_max:
                inf_max = inf_j
                j_best = j

        solution = solution.union({j_best})

    stop = perf_counter()
    print("time: " + str(stop-start))
    print("max inf gain: " + str(inf_max))
    return [solution,inf_max,0,stop-start]

def cma_es(param):
    start = perf_counter()
    inf_max = float('-inf') 

    def obj_function(candidates):
        nonlocal inf_max
        selected_indices = np.argsort(-candidates)[:param.k]
        inf_gain = fim(param.x, param.p, param.d, selected_indices, param.iso_var, param.sensor_var)

        if inf_gain > inf_max:
            inf_max = inf_gain
        return -inf_gain
    
    seed = 0.5 * np.ones(len(param.p))
    sigma = 0.3

    params = {'maxiter': param.max_generations, 'popsize': param.pop_size}
    es = cma.CMAEvolutionStrategy(seed, sigma, params)
    es.optimize(obj_function)

    continuous_sol = es.result.xbest
    solution = np.argsort(-continuous_sol)[:param.k]
    stop = perf_counter()
    print("time: " + str(stop-start))
    print("max inf gain: " + str(inf_max))
    return [solution,inf_max,0,stop-start]

if __name__ == "__main__":


    # k = 4
    # p = np.column_stack((np.random.randint(0,100,30),np.random.randint(0,100,30),np.random.randint(0,100,30)))
    # x = np.array([(1.5,1.5,1.5),(2,2.5,2.5),(2.5,1.5,1.5)])
    # # p = np.array([(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)])
    # iso = isotropic(3,2)
    # variance = 0.25

    k = 3
    p = np.column_stack((np.random.randint(0,50,10),np.random.randint(0,50,10)))
    x = np.column_stack((np.random.randint(0,50,2),np.random.randint(0,50,2)))
    iso = isotropic(2,2)
    variance = 1

    def addNoise(x, p, variance):
        d = np.array([[np.linalg.norm(i - j) for j in p] for i in x])
        noise = np.random.normal(0, np.sqrt(variance), size=np.shape(d))
        return d + noise
    
    d = addNoise(x,p,variance)

    param = Parameters((50,50),k,x,p,d,iso,variance)
    # print(brute(k,x,p,d,iso,variance))
    print(greedy(param))
    print(cma_es(param))

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
