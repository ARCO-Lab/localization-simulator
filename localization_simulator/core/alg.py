from time import perf_counter
from .inf import fim, isotropic
from ..utils.result import Parameters
import numpy as np
import pandas as pd
from itertools import combinations
import cma
import random
from ..utils.nls import NLS

def random_set(param):
    print(f"len Param P: {len(param.p)}")
    start = perf_counter()
    solution = set()
    while len(solution) < param.k:
        solution.add(random.randint(0,len(param.p)-1))
    inf = fim(param.x,param.p,param.d,solution,param.iso_var,param.sensor_var)
    stop = perf_counter()
    return [solution,inf,0,stop-start]

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

def bruteRMSE(param, initial):
    nls = NLS(None,None,None,1e-9,None,None)
    start = perf_counter()
    candidates = list(combinations(range(len(param.p)),param.k))

    rmse_min = float('inf')
    solution = set()

    for c in candidates:
        print(c)
        ancC = np.array([param.p[i] for i in c])
        print(d)
        print([[d[i][j] for j in c] for i in range(len(param.x))])
        rmse = np.sqrt(np.mean([nls.rmse(initial[i],param.x[i], [d[i][j] for j in c], ancC, param.sensor_var, param.iso_var) for i in range(len(param.x))]))

        if rmse < rmse_min:
            rmse_min = rmse
            solution = c
    
    stop = perf_counter()
    print("time: " + str(stop-start))
    print("min Rmse: " + str(rmse_min))
    return [solution,rmse_min,0,stop-start]


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

def greedyTrace(param):
    start = perf_counter()
    solution = set()

    for _ in range(param.k):
        inf_max = float('-inf')
        j_best = None
        
        for j in range(len(param.p)):
            inf_j = fim(param.x,param.p,param.d,solution.union({j}),param.iso_var,param.sensor_var, trace=True)
            # print(f"Set:{solution.union({j})} and inf_j:{inf_j} ")

            if inf_j > inf_max:
                inf_max = inf_j
                j_best = j

        solution = solution.union({j_best})

    stop = perf_counter()
    print("time: " + str(stop-start))
    print("max inf gain: " + str(inf_max))
    return [solution,inf_max,0,stop-start]

def greedyMinEig(param):
    start = perf_counter()
    solution = set()

    for _ in range(param.k):
        inf_max = float('-inf')
        j_best = None
        
        for j in range(len(param.p)):
            inf_j = fim(param.x,param.p,param.d,solution.union({j}),param.iso_var,param.sensor_var, trace=True)
            # print(f"Set:{solution.union({j})} and inf_j:{inf_j} ")

            if inf_j > inf_max:
                inf_max = inf_j
                j_best = j

        solution = solution.union({j_best})

    stop = perf_counter()
    print("time: " + str(stop-start))
    print("max inf gain: " + str(inf_max))
    return [solution,inf_max,0,stop-start]

def greedyAncMax(param):
    start = perf_counter()
    solution = set()

    connections = np.sum(~np.isnan(param.d), axis=0)

    for _ in range(param.k):
        ancMax = np.argmax(connections)
        solution.add(ancMax)
        connections[ancMax] = -1

    inf = fim(param.x,param.p,param.d,solution,param.iso_var,param.sensor_var)
    stop = perf_counter()
    return [solution,inf,0,stop-start]

def greedyPosGuarantee(param):
    start = perf_counter()
    solution = set()

    covered = np.zeros(np.shape(param.d)[0])
    connections = np.sum(~np.isnan(param.d), axis=1)

    for _ in range(param.k):
        if np.all(covered==1):
            ancMax = np.argmax(connections)
            solution.add(ancMax)
            connections[ancMax] = -1
        else:
            m = -1
            j_best = None
            i_best = None
            for i,j in enumerate(param.d):
                c = np.count_nonzero(covered != ~np.isnan(j))
                if c > m:
                    m = c
                    j_best = ~np.isnan(j)
                    i_best = i
            covered[j_best] = 1
            solution.add(i_best)
            connections[i_best] = -1

    inf = fim(param.x, param.p, param.d, solution, param.iso_var, param.sensor_var)
    stop = perf_counter()

    return [solution, inf, 0, stop - start]

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

    k = 4
    p = np.column_stack((np.random.randint(0,50,40),np.random.randint(0,50,40)))
    x = np.column_stack((np.random.randint(0,50,20),np.random.randint(0,50,20)))
    iso = isotropic(2,2)
    variance = 1
    cutoff = 20

    def addNoise(p, x, variance, cutoff):
        d = np.array([[np.linalg.norm(i - j) if np.linalg.norm(i - j) < cutoff else np.nan for j in p] for i in x])
        noise = np.random.normal(0, np.sqrt(variance), size=np.shape(d))
        noisy_d = d + noise
        noisy_d[np.isnan(d)] = np.nan 
        return noisy_d
    
    d = addNoise(p,x,variance,cutoff)

    # d = data = np.array([[2, 3, np.nan],[3, 4, np.nan],[np.nan, np.nan, 2]])

    param = Parameters((50,50),k,x,p,d,iso,variance)
    noise = np.random.normal(0, np.sqrt(param.iso_var[0][0]), (len(param.x), 2))
    initial = param.x + noise
    # print(greedyAncMax(param))
    # print(greedyPosGuarantee(param))
    print(bruteRMSE(param,initial))


    # # print(brute(k,x,p,d,iso,variance))
    # print(greedy(param))
    # print(cma_es(param))

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

