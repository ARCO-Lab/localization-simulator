import pandas as pd
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from ..core.inf import isotropic, addNoise
from ..core.alg import greedy,brute,cma_es, random_set
import copy

class Parameters():
    def __init__(self,dim,k,x,p,d,iso_var,sensor_var,pop_size=100,max_generations=100,sigma=0.3) -> None:
        self.dim = dim
        self.k = k
        self.x = x
        self.p = p
        self.d = d
        self.iso_var = iso_var
        self.sensor_var = sensor_var
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.sigma = sigma
    
    def visualize(self, img=None):
        if img:
            i = plt.imread(img)
            plt.imshow(i)
        plt.scatter(self.p[:,0],self.p[:,1],color="r",marker = "x",s=10)
        plt.plot(self.x[:,0], self.x[:,1],'--cD',markersize=3)
        for i in self.x:
            plt.gca().add_patch(Circle((i[0], i[1]), self.iso_var[0][0], color='c', fill=False))
        plt.gca().set_aspect('equal')    
        return plt
    
    def checkFolder(self):
        path = f"{os.getcwd()}/results"
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folder 'results' created successfully.")
    
    def writeTxt(self, name=None):
        path = f"{os.getcwd()}/results"
        self.checkFolder()
        name = f"{name}.txt" if name else f"{datetime.now().strftime('%B_%d_%Y_%I_%M_%S')}.txt"
        with open(f"{path}/{name}", 'w') as f:
            f.write(f"dim: {self.dim}\n")
            f.write(f"# of poses: {len(self.x)}\n")
            f.write(f"# of anchors: {len(p)}\n")
            f.write(f"k: {self.k}\n")
            f.write(f"iso_var: {self.iso_var[0][0]}\n")
            f.write(f"sensor_var: {self.sensor_var}\n")
            f.write(f"pop_size: {self.pop_size}\n")
            f.write(f"max_generations: {self.max_generations}\n")
            f.write(f"sigma: {self.sigma}\n")

            f.write("\nx:\n")
            for i, item in enumerate(self.x, 1):
                f.write(f"{i} - {item}\n")

            f.write("\np:\n")
            for i, item in enumerate(self.p, 1):
                f.write(f"{i} - {item}\n")

    def to_dataframe(self):
        params_data = {
            'Parameter': ['dim', 'k', 'iso_var', 'sensor_var', 'pop_size', 'max_generations', 'sigma'],
            'Value': [self.dim, self.k, self.iso_var[0][0], self.sensor_var, self.pop_size, self.max_generations, self.sigma]
        }
        df_params = pd.DataFrame(params_data)
        df_x = pd.DataFrame(self.x, columns=["x","y"] if len(self.dim) == 2 else ["x","y","z"])
        df_p = pd.DataFrame(self.p, columns=["x","y"] if len(self.dim) == 2 else ["x","y","z"])

        return df_params, df_x, df_p

    def writeCSV(self):
        path = f"{os.getcwd()}/results"
        self.checkFolder()
        name = f"{name}.txt" if name else f"{datetime.now().strftime('%B_%d_%Y_%I_%M_%S')}.txt"
        df_params, df_x, df_p = self.to_dataframe()

        with open(f"{path}/{name}", 'a') as f:
            f.write("Parameters:\n")
            for _, row in df_params.iterrows():
                f.write(f"{row['Parameter']}, {row['Value']}\n")
            f.write("\n")

            f.write("x:\n")
            f.write(df_x.to_string(index=False) + "\n\n")

            f.write("p:\n")
            f.write(df_p.to_string(index=False) + "\n\n")

 
class Result():

    def __init__(self) -> None:
        pass


class Result():

    def __init__(self, *args) -> None:
        self.algs = args
        columns = ['Algorithm','Information', 'RMSE', 'Runtime']
        subcolumns = pd.MultiIndex.from_product([columns, ['mean', 'std']])
        self.df = pd.DataFrame(index=self.algs ,columns=subcolumns)
        self.time = f"{datetime.now().strftime('%B_%d_%Y_%I_%M_%S')}"
    
    def add(self,algName,result):
        self.df.loc[algName] = [algName] + result
    
    def checkFolder(self):
        path = f"{os.getcwd()}/results"
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folder 'results' created successfully.")

    def toCSV(self, name=None):
        path = f"{os.getcwd()}/results"
        self.checkFolder()
        name = f"{name}.csv" if name else f"{self.time}.csv"
        self.df.to_csv(f"{path}/{name}",index=False)
    
    def toLatex(self, name=None):
        path = f"{os.getcwd()}/results"
        self.checkFolder()
        name = f"{name}.tex" if name else f"{self.time}.tex"
        with open(f"{path}/{name}","w") as f:
            f.write(self.df.to_latex(index=False))


if __name__ == "__main__":
    r = Result("Random","Brute-force","Greedy","CMA-ES")
    r.add("Greedy",[1,2,3,4,5,6,7])
    print(r.df)
    # r.toCSV("test")
    r.toLatex("test5")

    # dim = (1059,641)
    # k = 2
    # p = np.column_stack((np.random.randint(10,1050,50),np.random.randint(0,610,50)))
    # # x = np.column_stack((np.random.randint(0,1059,10),np.random.randint(0,641,10)))
    # x = np.array([(1030,0),(1030,150),(970,160),(960,220),(850,230),
    #               (880,390),(840,520),(1030,510),(1030,600),(710,600),
    #               (719,531),(715,430),(587,422),(576,335),(555,254),
    #               (496,258),(490,396),(435,398),(413,423),(285,421),
    #               (290,593),(154,565),(137,425),(110,420),(82,225),
    #               (108,197),(105,178),(130,178),(140,106),(81,0)
    #               ])
    # iso = isotropic(2,10)
    # variance = 5
    
    # d = addNoise(x,p,variance)

    # param = Parameters(dim,k,x,p,d,iso,variance)
    # param2, param3 = copy.deepcopy(param), copy.deepcopy(param)

    # # param.writeTxt("2dcase - all 3")
    # yb,yg,yc = param.visualize("assets/factorylayout1.jpg"),param2.visualize("assets/factorylayout2.jpg"),param3.visualize("assets/factorylayout3.jpg")

    # bset = brute(param)
    # gset = greedy(param2)
    # cset = cma_es(param3)

    # bsetList = np.array([(p[i][0],p[i][1]) for i in bset])
    # gsetList = np.array([(p[i][0],p[i][1]) for i in gset])
    # csetList = np.array([(p[i][0],p[i][1]) for i in cset])

    # # brute
    # cir = []
    # for i in bsetList:
    #     circle = yb.Circle((i),radius=10,color="m",fill=True)
    #     yb.gca().add_patch(circle)
    #     cir.append(circle)
    # yb.scatter(bsetList[:,0],bsetList[:,1],color="r",marker = "x",s=10)
    # yb.savefig(f"assets/{datetime.now().strftime('%B_%d_%Y_%I_%M_%S')}_brute.png")
    
    # for c in cir:
    #     c.remove()

    # greedy
    # cir = []
    # for i in gsetList:
    #     circle = yg.Circle((i),radius=10,color="y",fill=True)
    #     yg.gca().add_patch(circle)
    #     cir.append(circle)
    # yg.scatter(param.p[:,0],param.p[:,1],color="r",marker = "x",s=10)
    # yg.savefig(f"assets/{datetime.now().strftime('%B_%d_%Y_%I_%M_%S')}_greedy.png")
    
    # for c in cir:
    #     c.remove()

    # # cma
    # cir = []
    # for i in csetList:
    #     circle = yc.Circle((i),radius=10,color="dodgerblue",fill=True)
    #     yc.gca().add_patch(circle)
    #     cir.append(circle)
    # yc.scatter(csetList[:,0],csetList[:,1],color="r",marker = "x",s=10)
    # yc.savefig(f"assets/{datetime.now().strftime('%B_%d_%Y_%I_%M_%S')}_cmaes.png")
    
    # for c in cir:
    #     c.remove()
    
    
    
    

