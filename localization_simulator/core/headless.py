from ..utils.result import Parameters, Result
from .inf import isotropic, shiftedPos
import numpy as np
from matplotlib import pyplot as plt
from .alg import brute, greedy, cma_es, random_set, greedyAncMax, greedyTrace, greedyPosGuarantee
from ..utils.nls import NLS


def run(map, k=5, cutoff=250, prior=8, ancList=None):
        """ Performs the simulation without any visualization
        """
        # print(f"ISOTROPIC VARIANCE: {map.isotropic}")
        res = Result("Random","Greedy","Measurement Greedy","Coverage Greedy")
        res2 = Result("Brute-force","Greedy","CMA-ES")

        map.k = k
        map.isotropic = isotropic(map.nDim,prior)

        runs = 1
        # anchorLocations = np.array([a.location for a in map.anchors])

        

        # d = map.addNoise(anchorLocations,map.poses,map.variance, cutoff)

        # nls = NLS(map.points,map.gradNorms,anchorLocations,variance=0.01,tolerance=1e-9)

        # param = Parameters(map.dim,map.k,map.poses,anchorLocations,d,map.isotropic,map.variance)
        # noise = np.random.normal(0, np.sqrt(map.isotropic[0][0]), (len(map.poses), 2))
        # # noise = np.zeros(np.shape(map.poses))
        # initial = map.poses + noise

        plotinfBrute = []
        plotinfrand = []
        plotinfgreedy = []
        plotinfcma = []
        plotinfgreedyMax = []
        plotrmserand = []
        plotrmsegreedy = []
        plotrmsecma = []
        plotrmsegreedyMax = []
        plotrmsegreedyCov = []
        plotrmseBrute = []
        plotGradNorm = []
        plotHesEig = []
        plotrmseTest = []

        plotrmsegreedytrace = []

        minlist = []
        maxlist = []
        saddlelist = []

        avgRmseRand = []
        avgRmseGreedy = []
        avgRmseGreedyMax = []
        avgRmseGreedyCov = []
        avgRmseCma = []
        avgRmseBrute = []

        avgRunRand = []
        avgRunGreedy = []
        avgRunGreedyMax = []
        avgRunGreedyCov = []
        avgRunCma = []
        avgRunBrute = []


        for segment in range(0,runs):
            # anchorLocations =  np.column_stack((np.random.randint(10,1050,50),np.random.randint(0,610,50)))
            if ancList:
                anchorLocations = ancList[segment]
            d = map.addNoise(anchorLocations,map.poses,map.variance, cutoff)
            # d = map.addNoiseRmse(anchorLocations,map.poses,map.variance)

            nls = NLS(map.points,map.gradNorms,anchorLocations,variance=0.01,tolerance=1e-9)

            param = Parameters(map.dim,map.k,map.poses,anchorLocations,d,map.isotropic,map.variance)
            noise = np.random.normal(0, np.sqrt(map.isotropic[0][0]), (len(map.poses), 2))
            initial = map.poses + noise


            # print("____________________________________________________________________________________________________________________________________________________________________")
            # print("\nRANDOM")
            # print("_____________")
            # resRandom = random_set(param)
            # resRandom[0] = sorted(resRandom[0])
            # print(resRandom[0])
            # aRandom = np.array([anchorLocations[i] for i in resRandom[0]])
            # # plotinfrand.append(resRandom[1])
            # rmseRand = np.sqrt(np.mean([nls.rmse(initial[i],map.poses[i], [d[i][j] for j in resRandom[0]], aRandom, map.variance, map.isotropic) for i in range(len(map.poses))]))
            # plotrmserand.append(rmseRand)
            # avgRmseRand.append(rmseRand)
            # avgRunRand.append(resRandom[3])
            # print(f"RMSE: {rmseRand}")
            # print("____________________________________________________________________________________________________________________________________________________________________")


            print("\nBrute")
            print("_____________")
            resBrute = brute(param)
            resBrute[0] = sorted(resBrute[0])
            print(resBrute[0])
            aBrute = np.array([anchorLocations[i] for i in resBrute[0]])
            plotinfBrute.append(resBrute[1])
            rmseBrute = np.sqrt(np.mean([nls.rmse(initial[i],map.poses[i],[d[i][j] for j in resBrute[0]], aBrute, map.variance, map.isotropic) for i in range(len(map.poses))]))
            plotrmseBrute.append(rmseBrute)
            avgRmseBrute.append(rmseBrute)
            avgRunBrute.append(resBrute[3])
            print(f"RMSE: {rmseBrute}")
            print("____________________________________________________________________________________________________________________________________________________________________")

            print("\nGreedy")
            print("_____________")
            resGreedy = greedy(param)
            resGreedy[0] = sorted(resGreedy[0])
            print(resGreedy[0])
            aGreedy = np.array([anchorLocations[i] for i in resGreedy[0]])
            plotinfgreedy.append(resGreedy[1])
            rmseGreedy = np.sqrt(np.mean([nls.rmse(initial[i],map.poses[i], [d[i][j] for j in resGreedy[0]], aGreedy, map.variance, map.isotropic) for i in range(len(map.poses))]))
            plotrmsegreedy.append(rmseGreedy)
            avgRmseGreedy.append(rmseGreedy)
            avgRunGreedy.append(resGreedy[3])
            print(f"RMSE: {rmseGreedy}")
            print("____________________________________________________________________________________________________________________________________________________________________")

 
            # print("\nGreedyAncMax")
            # print("_____________")
            # resGreedyMax = greedyAncMax(param)
            # resGreedyMax[0] = sorted(resGreedyMax[0])
            # print(resGreedyMax[0])
            # aGreedyMax = np.array([anchorLocations[i] for i in resGreedyMax[0]])
            # # plotinfgreedyMax.append(resGreedyMax[1])
            # rmseGreedyMax = np.sqrt(np.mean([nls.rmse(initial[i],map.poses[i], [d[i][j] for j in resGreedyMax[0]], aGreedyMax, map.variance, map.isotropic) for i in range(len(map.poses))]))
            # plotrmsegreedyMax.append(rmseGreedyMax)
            # avgRmseGreedyMax.append(rmseGreedyMax)
            # avgRunGreedyMax.append(resGreedyMax[3])
            # print(f"RMSE: {rmseGreedyMax}")
            # print("____________________________________________________________________________________________________________________________________________________________________")           

            # print("\nGreedyCov")
            # print("_____________")
            # resGreedyCov = greedyPosGuarantee(param)
            # resGreedyCov[0] = sorted(resGreedyCov[0])
            # print(resGreedyCov[0])
            # aGreedyCov = np.array([anchorLocations[i] for i in resGreedyCov[0]])
            # # plotinfgreedyMax.append(resGreedyMax[1])
            # rmseGreedyCov = np.sqrt(np.mean([nls.rmse(initial[i],map.poses[i], [d[i][j] for j in resGreedyCov[0]], aGreedyCov, map.variance, map.isotropic) for i in range(len(map.poses))]))
            # plotrmsegreedyCov.append(rmseGreedyCov)
            # avgRmseGreedyCov.append(rmseGreedyCov)
            # avgRunGreedyCov.append(resGreedyCov[3])
            # print(f"RMSE: {rmseGreedyCov}")
            # print("____________________________________________________________________________________________________________________________________________________________________")           

            print("\nCMA")
            print("_____________")
            resCmaes = cma_es(param)
            resCmaes[0] = sorted(resCmaes[0])
            print(resCmaes[0])
            aCmaes = np.array([anchorLocations[i] for i in resCmaes[0]])
            plotinfcma.append(resCmaes[1])
            rmseCma = np.sqrt(np.mean([nls.rmse(initial[i],map.poses[i],[d[i][j] for j in resCmaes[0]], aCmaes, map.variance, map.isotropic) for i in range(len(map.poses))]))
            plotrmsecma.append(rmseCma)
            avgRmseCma.append(rmseCma)
            avgRunCma.append(resCmaes[3])
            print(f"RMSE: {rmseCma}")
            print("____________________________________________________________________________________________________________________________________________________________________")     


        print("1")
        infB = np.mean(plotinfBrute)
        infG = np.mean(plotinfgreedy)
        infC = np.mean(plotinfcma)

        print("2")
        infBstd = np.std(plotinfBrute)
        infGstd = np.std(plotinfgreedy)
        infCstd = np.std(plotinfcma)

        print("3")
        avgB = np.mean(avgRmseBrute)
        avgG = np.mean(avgRmseGreedy)
        avgC = np.mean(avgRmseCma)

        print("4")
        stdB = np.std(avgRmseBrute)
        stdG = np.std(avgRmseGreedy)
        stdC = np.std(avgRmseCma)

        print("5")
        runB = np.mean(avgRunBrute)
        runG = np.mean(avgRunGreedy)
        runC = np.mean(avgRunCma)


        print("6")
        runBstd = np.std(avgRunBrute)
        runGstd = np.std(avgRunGreedy)
        runCstd = np.std(avgRunCma)


        # res.add("Random",[0,0,0,avgR,stdR,runR,runRstd])
        # res.add("Greedy",[0,0,0,avgG,stdG,runG,runGstd])
        # res.add("Measurement Greedy",[0,0,0,avgGM,stdGM,runGM,runGMstd])
        # res.add("Coverage Greedy",[0,0,0,avgGC,stdGC,runGC,runGCstd])
        # res.toLatex(f"k{k}c{cutoff}p{prior}")

        res.add("Random",[0,infB,infBstd,avgB,stdB,runB,runBstd])
        res.add("Greedy",[0,infG,infGstd,avgG,stdG,runG,runGstd])
        res.add("CMA-ES",[0,infC,infCstd,avgC,stdC,runC,runCstd])
        res.toLatex(f"part1_k{k}c{cutoff}p{prior}")

        
        # return [plotrmserand,plotrmsegreedy,plotrmsegreedyMax,plotrmsegreedyCov]
        return [infB,infG,infC]
    
def headless(map):           
        # print("\nGreedyTrace")
        # print("_____________")
        # resGreedyTrace = greedyTrace(param)
        # resGreedyTrace[0] = sorted(resGreedyTrace[0])
        # print(resGreedyTrace[0])
        # aGreedyTrace = np.array([anchorLocations[i] for i in resGreedyTrace[0]])

        # plotrmsegreedytrace.append(np.sqrt(np.mean([nls.rmse(map.poses[i], initial[i], [d[i][j] for j in resGreedyTrace[0]], aGreedyTrace, map.variance, map.isotropic) for i in range(len(map.poses))])))
        # print(f"RMSE: {np.sqrt(np.mean([nls.rmse(map.poses[i], initial[i], [d[i][j] for j in resGreedyTrace[0]], aGreedyTrace, map.variance, map.isotropic) for i in range(len(map.poses))]))}")
        # print("____________________________________________________________________________________________________________________________________________________________________")   

        # print("\nTEST")
        # print("_____________")
        # sol = [0,1,2,3,4]
        # aSol = np.array([anchorLocations[i] for i in sol])


        # plotrmseTest.append(np.sqrt(np.mean([nls.rmse(initial[i],map.poses[i], [d[i][j] for j in sol], aSol, map.variance, map.isotropic) for i in range(len(map.poses))])))
        # print(f"RMSE: {np.sqrt(np.mean([nls.rmse(initial[i],map.poses[i], [d[i][j] for j in sol], aSol, map.variance, map.isotropic) for i in range(len(map.poses))]))}")
        # print("____________________________________________________________________________________________________________________________________________________________________")    


        # print("\nSANITY")
        # print("_____________")
        # sol = [i for i in range(0,segment+1)]
        # aSol = np.array([anchorLocations[i] for i in sol])

        # val = np.mean([nls.gradNorm(map.poses[i], initial[i], [d[i][j] for j in sol], aSol, map.variance, map.isotropic) for i in range(len(map.poses))])
        # print(val)
        # plotGradNorm.append(val)

        # rmseSan = np.sqrt(np.mean([nls.rmse(initial[i],map.poses[i], [d[i][j] for j in sol], aSol, map.variance, map.isotropic) for i in range(len(map.poses))]))
        # print(f"RMSE:{rmseSan}")
        # plotrmseSan.append(rmseSan)
        # hesArray = np.array([nls.hesEig(map.poses[i], initial[i], [d[i][j] for j in sol], aSol, map.variance, map.isotropic) for i in range(len(map.poses))])
        # val = np.min(hesArray)
        # if np.all(hesArray >= 0):
        #     plotHesEig.append(1)
        #     minlist.append((segment,rmseSan))
        # elif np.all(hesArray <= 0):
        #     plotHesEig.append(-1)
        #     maxlist.append((segment,rmseSan))
        # else:
        #     plotHesEig.append(0)
        #     saddlelist.append((segment,rmseSan))

        # print(val)
        # plotHesEig.append(val)

        
        # print("\nCMA")
        # print("_____________")
        # resCmaes = cma_es(param)
        # resCmaes[0] = sorted(resCmaes[0])
        # print(resCmaes[0])
        # aCmaes = np.array([anchorLocations[i] for i in resCmaes[0]])

        # plotinfcma.append(resCmaes[1])
        # plotrmsecma.append(np.sqrt(np.mean([nls.rmse(map.poses[i], initial[i], [d[i][j] for j in resCmaes[0]], aCmaes, map.variance, map.isotropic) for i in range(len(map.poses))])))
        # print(f"RMSE: {np.sqrt(np.mean([nls.rmse(map.poses[i], initial[i], [d[i][j] for j in resCmaes[0]], aCmaes, map.variance, map.isotropic) for i in range(len(map.poses))]))}")
        # print("____________________________________________________________________________________________________________________________________________________________________")


    # plt.gca().set_xscale('log')


    # if len(minlist) > 0:
    #     x,y = zip(*minlist)
    #     plt.scatter(x,y,label="Min")
    # if len(maxlist) > 0:
    #     x,y = zip(*maxlist)
    #     plt.scatter(x,y,label="Max")
    # if len(saddlelist) > 0:
    #     x,y = zip(*saddlelist)
    #     plt.scatter(x,y,label="Saddle")

    # plt.scatter([i for i in range(runs)],plotHesEig,label="HesEig")
    # plt.scatter([i for i in range(runs)],plotrmseSan,label="sanity")
    # plt.scatter(plotrmserand,plotinfrand, label="Random")
    # plt.scatter(plotrmsecma,plotinfcma, label="CMA")
    # plt.scatter(plotrmsegreedy,plotinfgreedy, label="Greedy")
    # plt.scatter(plotrmsegreedyMax, plotinfgreedyMax, label="Greedy Max")
    # plt.scatter(plotrmseTest,[50 for i in range(len(plotrmseTest))], label="Sanity Test")

    # fig, ([ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9]) = plt.subplots(3, 3)

    # for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]:
    #     ax.xaxis.set_visible(False)

    # ax1.set_title("K = 5") 
    # ax2.set_title("Cutoff = 150") 
    # ax3.set_title("Prior Variance = 5") 
    # ax4.set_title("K = 10") 
    # ax5.set_title("Cutoff = 300") 
    # ax6.set_title("Prior Variance = 10") 
    # ax7.set_title("K = 15") 
    # ax8.set_title("Cutoff = 450") 
    # ax9.set_title("Prior Variance = 15") 


    # ret = map.run()
    # sns.boxplot(data=[ret[0],ret[1],ret[2],ret[3]], ax=ax1)
    # ret = map.run(cutoff=150)
    # sns.boxplot(data=[ret[0],ret[1],ret[2],ret[3]], ax=ax2)
    # ret = map.run(prior=5)
    # sns.boxplot(data=[ret[0],ret[1],ret[2],ret[3]], ax=ax3)
    # ret = map.run(k=10)
    # sns.boxplot(data=[ret[0],ret[1],ret[2],ret[3]], ax=ax4)
    # ret = map.run(cutoff=300)
    # sns.boxplot(data=[ret[0],ret[1],ret[2],ret[3]], ax=ax5)
    # ret = map.run(prior=10)
    # sns.boxplot(data=[ret[0],ret[1],ret[2],ret[3]], ax=ax6)
    # ret = map.run(k=15)
    # sns.boxplot(data=[ret[0],ret[1],ret[2],ret[3]], ax=ax7)
    # ret = map.run(cutoff=450)
    # sns.boxplot(data=[ret[0],ret[1],ret[2],ret[3]], ax=ax8)
    # ret = map.run(prior=15)
    # sns.boxplot(data=[ret[0],ret[1],ret[2],ret[3]], ax=ax9)


    # default_palette = sns.color_palette()
    # legend_labels = ["Random","Greedy","Measurement Greedy","Coverage Greedy"]
    # legend_handles = [plt.Line2D([0], [0], marker='o', color=color, label=label, markerfacecolor=color, markersize=10) 
    #                 for label, color in zip(legend_labels, default_palette[:len(legend_labels)])]

    # fig.legend(handles=legend_handles, labels=legend_labels, loc='lower center', ncol=4)

    # fig.text(0.06, 0.5, 'RMSE', va='center', rotation='vertical')
    
    # plt.show()

    ancList =  [np.column_stack((np.random.randint(10,1050,20),np.random.randint(0,610,20))) for i in range(3)]



    plotb = [0]
    plotg = [0]
    plotc = [0]

    for i in range(1,3):
        ret = map.run(k=i,ancList=ancList)
        plotb.append(ret[0])
        plotg.append(ret[1])
        plotc.append(ret[2])

    def bound(x):
        return (1 - (1 / np.e)) * x

    x = np.linspace(0, 2, 3)

    plotbound = [bound(x) for x in plotb]

    # Plot the value as a dashed line
    plt.plot(x, plotb,color='red', label="Brute-force")
    plt.plot(x, plotc,color='green', label="CMA-ES")
    plt.plot(x, plotg,color='orange', label="Greedy")
    plt.plot(x, plotbound, linestyle='--', color='blue', label=r'$1 - \frac{1}{e} \cdot x$')
    plt.xlabel('K')
    plt.ylabel('Information Gain')
    plt.grid(True)
    plt.legend()
    plt.show()


    # res.toLatex("test2")