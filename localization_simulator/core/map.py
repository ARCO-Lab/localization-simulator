"""
A module for the map and assembling components for a simulation.

Todo:
    * Decouple other entities from maps and make it simpler
    * Allow for multiple trajectories
    * Organize imports
    * CSV
"""

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib
import pandas as pd
import random
# import seaborn as sns

from ..utils.plot import Plot
from .inf import isotropic, shiftedPos
from .component import Anchor
from ..utils.trajectory import Traj2D, Traj3D
from ..utils.nls import NLS
from .error import Error
import tkinter as tk
from tkinter import ttk
from .alg import brute, greedy, cma_es, random_set, greedyAncMax
from ..utils.result import Parameters, Result

class Map:
    """Module for creating simulation maps.

    Attributes:
        dim (tuple[int]): The dimensions of the map.
        nDim (int): The dimensionality of the map (2 or 3).
        fig (matplotlib.figure.Figure): Matplotlib figure for visualization.
        ax (matplotlib.axes._axes.Axes): Matplotlib axes for visualization.
        trajectory (Traj2D or Traj3D): The trajectory for the simulation (either 2D or 3D).
        points (list[list[numpy.ndarray(float)]]): The list of points for each pose.
        gradNorms (list[list[float]]): The list of gradient norms for each pose.
        isotropic (numpy.ndarray(int)): Isotropic covarriance matrix for the prior
    """
    def __init__(self, dim) -> None:
        """Init method

        Args:
            dim (tuple[int]): The dimension of the map to be created.
        """
        self.dim = dim
        self.nDim = len(dim)
        # self.fig, self.ax = Plot.create2D(self.dim) if self.nDim==2 else Plot.create3D(self.dim)
        self.trajectory = None
        self.points = []
        self.gradNorms = []
        self.isotropic = None
        self.k = None
        self.variance = None
    
    def setK(self, k):
        self.k = k
    
    def setVariance(self, variance):
        self.variance = variance
    
    def setIsotropic(self,isotropicVariance):
        self.isotropic = isotropic(self.nDim,isotropicVariance)
    
    def placeAnchor(self, anchorList):
        """Method for placing anchors in a map.

        Args:
            anchorList (list[Anchor]): The list of anchors to place in a map.
        """
        self.anchors = anchorList
        # if self.nDim == 2:
        #     for a in anchorList:
        #         self.ax.add_patch(Rectangle(a.location,0.5,0.5,fc="black",ec=a.clr))
        # else:
        #     for a in anchorList:
        #         self.ax.scatter(a.location[0], a.location[1], a.location[2], c='black', edgecolors=a.clr ,marker='o', s=100)
            
    def loadTraj(self, poses, interval):
        """Method for loading a trajectory into a map

        Args:
            poses (list[tuple[float]]): The desired poses for the trajectory.
            interval (int): The number of substeps in between each pose. Can be thought as the number of points in a straight line between poses.
        """
        self.poses = poses
        if self.nDim == 2:
            self.trajectory = Traj2D(poses, interval)
        else:
            self.trajectory = Traj3D(poses, interval)
        self.trajectory.generateTraj()

    def createWindow(self):
        """Creates a window to display pose tracking information using Tkinter.

        The window currently displays the pose number as well as the distance and likelihood of observing the distance in the probability density for each anchor.

        Returns:
            (tuple[tk.Tk, tk.Label, ttk.Treeview]): The main tkinter window, the title widgetm and the treeview widget.
        """
        root = tk.Tk()
        root.title('Pose Tracker')

        value_label = tk.Label(root, text="Pose Tracker", font=("Arial",30)).grid(row=0, columnspan=4)

        cols = ('Pose', 'Anchor', 'Distance', 'Probability Density')
        listBox = ttk.Treeview(root, columns=cols, show='headings')
        for col in cols:
            listBox.heading(col, text=col)    
        listBox.grid(row=1, column=0, columnspan=2)

        return root, value_label, listBox

    @staticmethod
    def addNoise(p, x, variance, cutoff):
        d = np.array([[np.linalg.norm(i - j) if np.linalg.norm(i - j) < cutoff else np.nan for j in p] for i in x])
        noise = np.random.normal(0, np.sqrt(variance), size=np.shape(d))
        noisy_d = d + noise
        noisy_d[np.isnan(d)] = np.nan 
        return noisy_d

    @staticmethod
    def addNoiseRmse(p,x,variance):
        d = np.array([[np.linalg.norm(i - j) for j in p] for i in x])
        noise = np.random.normal(0, np.sqrt(variance), size=np.shape(d))
        return d + noise

    def headless(self):
        """ Performs the simulation without any visualization
        """
        # print(f"ISOTROPIC VARIANCE: {self.isotropic}")
        res = Result("Brute-force","Greedy","CMA-ES")
        runs = 10
        # anchorLocations = np.array([a.location for a in self.anchors])
        cutoff = 200

        anchorLocations =  np.column_stack((np.random.randint(10,1050,50),np.random.randint(0,610,50)))
        # d = self.addNoise(anchorLocations,self.poses,self.variance, cutoff)

        # nls = NLS(self.points,self.gradNorms,anchorLocations,variance=0.01,tolerance=1e-6)

        # param = Parameters(self.dim,self.k,self.poses,anchorLocations,d,self.isotropic,self.variance)
        # noise = np.random.normal(0, np.sqrt(self.isotropic[0][0]), (len(self.poses), 2))
        # initial = self.poses + noise

        plotinfrand = []
        plotinfgreedy = []
        plotinfcma = []
        plotinfgreedyMax = []
        plotrmserand = []
        plotrmsegreedy = []
        plotrmsecma = []
        plotrmsegreedyMax = []
        plotrmseSan = []


        for segment in range(0,runs):

            d = self.addNoise(anchorLocations,self.poses,self.variance, cutoff)  
            # drmse = self.addNoiseRmse(anchorLocations,self.poses,self.variance)

            nls = NLS(self.points,self.gradNorms,anchorLocations,variance=0.01,tolerance=1e-6)

            param = Parameters(self.dim,self.k,self.poses,anchorLocations,d,self.isotropic,self.variance)
            noise = np.random.normal(0, np.sqrt(self.isotropic[0][0]), (len(self.poses), 2))
            initial = self.poses + noise

            # resRandom = random_set(param)
            # resRandom[0] = sorted(resRandom[0])
            # print("\nRANDOM")
            # print("_____________")
            # resRandom[0] = sorted(resRandom[0])
            # print(resRandom[0])
            # aRandom = np.array([anchorLocations[i] for i in resRandom[0]])
            
            # plotinfrand.append(resRandom[1])

            # plotrmserand.append(np.sqrt(np.mean([nls.rmse(self.poses[i], initial[i], [d[i][j] for j in resRandom[0]], aRandom, self.variance, self.isotropic) for i in range(len(self.poses))])))
                            
            # # print(f"Inf gain: {resRandom[1]}")
            # print(f"RMSE: {np.sqrt(np.mean([nls.rmse(self.poses[i], initial[i], [d[i][j] for j in resRandom[0]], aRandom, self.variance, self.isotropic) for i in range(len(self.poses))]))}")
            # print("____________________________________________________________________________________________________________________________________________________________________")


            # print("\nBrute")
            # print("_____________")
            # resBrute = brute(param)
            # resBrute[0] = sorted(resBrute[0])
            # print(resBrute[0])
            # aBrute = np.array([anchorLocations[i] for i in resBrute[0]])
            # print(f"RMSE: {np.sqrt(np.mean([nls.rmse(self.poses[i], initial[i], [d[i][j] for j in resBrute[0]], i, aBrute, self.variance, self.isotropic) for i in range(len(self.poses))]))}")
            # print("____________________________________________________________________________________________________________________________________________________________________")

            # print("\nGreedy")
            # print("_____________")
            # resGreedy = greedy(param)
            # resGreedy[0] = sorted(resGreedy[0])
            # print(resGreedy[0])
            # aGreedy = np.array([anchorLocations[i] for i in resGreedy[0]])

            # plotinfgreedy.append(resGreedy[1])
            # plotrmsegreedy.append(np.sqrt(np.mean([nls.rmse(self.poses[i], initial[i], [d[i][j] for j in resGreedy[0]], aGreedy, self.variance, self.isotropic) for i in range(len(self.poses))])))
            # print(f"RMSE: {np.sqrt(np.mean([nls.rmse(self.poses[i], initial[i], [d[i][j] for j in resGreedy[0]], aGreedy, self.variance, self.isotropic) for i in range(len(self.poses))]))}")
            # print("____________________________________________________________________________________________________________________________________________________________________")

 
            # print("\nGreedyAncMax")
            # print("_____________")
            # resGreedyMax = greedyAncMax(param)
            # resGreedyMax[0] = sorted(resGreedyMax[0])
            # print(resGreedyMax[0])
            # aGreedyMax = np.array([anchorLocations[i] for i in resGreedyMax[0]])

            # plotinfgreedyMax.append(resGreedyMax[1])
            # plotrmsegreedyMax.append(np.sqrt(np.mean([nls.rmse(self.poses[i], initial[i], [d[i][j] for j in resGreedyMax[0]], aGreedyMax, self.variance, self.isotropic) for i in range(len(self.poses))])))
            # print(f"RMSE: {np.sqrt(np.mean([nls.rmse(self.poses[i], initial[i], [d[i][j] for j in resGreedyMax[0]], aGreedyMax, self.variance, self.isotropic) for i in range(len(self.poses))]))}")
            # print("____________________________________________________________________________________________________________________________________________________________________")           


            print("\nSANITY")
            print("_____________")
            sol = [i for i in range(0,segment+1)]
            aSol = np.array([anchorLocations[i] for i in sol])
            print(f"RMSE: {np.sqrt(np.mean([nls.rmse(self.poses[i], initial[i], [d[i][j] for j in sol], aSol, self.variance, self.isotropic) for i in range(len(self.poses))]))}")
            plotrmseSan.append(np.sqrt(np.mean([nls.rmse(self.poses[i], initial[i], [d[i][j] for j in sol], aSol, self.variance, self.isotropic) for i in range(len(self.poses))])))
            # print("\nCMA")
            # print("_____________")
            # resCmaes = cma_es(param)
            # resCmaes[0] = sorted(resCmaes[0])
            # print(resCmaes[0])
            # aCmaes = np.array([anchorLocations[i] for i in resCmaes[0]])

            # plotinfcma.append(resCmaes[1])
            # plotrmsecma.append(np.sqrt(np.mean([nls.rmse(self.poses[i], initial[i], [d[i][j] for j in resCmaes[0]], aCmaes, self.variance, self.isotropic) for i in range(len(self.poses))])))
            # print(f"RMSE: {np.sqrt(np.mean([nls.rmse(self.poses[i], initial[i], [d[i][j] for j in resCmaes[0]], aCmaes, self.variance, self.isotropic) for i in range(len(self.poses))]))}")
            # print("____________________________________________________________________________________________________________________________________________________________________")


        plt.gca().set_xscale('log')

        plt.scatter([i for i in range(runs)],plotrmseSan,label="sanity")
        # plt.scatter(plotrmserand,plotinfrand, label="Random")
        # plt.scatter(plotrmsecma,plotinfcma, label="CMA")
        # plt.scatter(plotrmsegreedy,plotinfgreedy, label="Greedy")
        # plt.scatter(plotrmsegreedyMax, plotinfgreedyMax, label="Greedy Max")
        
        plt.legend()
        plt.show()


        # res.toLatex("test2")

    def visualize2D(self):
        """Visualizes the trajectory (2D) using Matplotlib animation
        """
        nls = NLS(self.points,self.gradNorms,np.array([a.location for a in self.anchors]), variance=0.01, tolerance=0.1)

        def show(listBox, send, num):
            """Populate the treeview widget with values at each timestep

            Args:
                listBox (ttk.Treeview): The Treeview widget.
                send (list): Values to be displayed in the Treeview widget.
                num (int): The timestep
            """
            for i in send:
                listBox.insert("", "end", values=(num,i[0], i[1], i[2]))

        def update(num):
            """Updates the line depicting the trajectory after each animation frame.

            Args:
                num (int): the frame number representing the timestep to be visualized.

            Returns:
                tuple[matplotlib.lines.Line2D,matplotlib.text.Text]: the updated line and title.
            """
            data = self.trajectory.df.iloc[num:num+1]
            ln.set_data(data.x, data.y)
            priorPos = shiftedPos((data.x.item(),data.y.item()),self.isotropic)
            ln_p.set_data(priorPos)

            if num % self.trajectory.interval == 0:
                c = np.random.rand(3,)
                ln.set_color(c)
                ln_p.set_color(c)
            title.set_text('2D Test, pose={}'.format(num))

            guess = np.array([np.random.uniform(0, self.dim[0]), np.random.uniform(0, self.dim[1])])
            nls.process(priorPos, guess)

            send = []
            for a in self.anchors:
                dist = a.getDist(priorPos)
                send.append([a.name,dist,a.error.getPDF(dist)])

            show(listBox, send, num)
            return ln,ln_p,title,

        title = self.ax.text(0.5,0.90, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=self.ax.transAxes, ha="center")

        ln, = self.ax.plot(self.trajectory.df.x, self.trajectory.df.y, 'ro')
        ln_p, = self.ax.plot([],[], 'rx')

        value_root, value_label, listBox = self.createWindow()
        
        ani = FuncAnimation(self.fig, update, frames=self.trajectory.interval*(len(self.trajectory.poses)-1),
                            blit=True, interval=500//self.trajectory.interval*(len(self.trajectory.poses)-1), repeat=False)
        
        plt.show()
        value_root.mainloop()
    
    def visualize3D(self):
        """Visualizes the trajectory (3D) using Matplotlib animation
        """
    
        def show(listBox, send, num):
            """Populate the treeview widget with values at each timestep

            Args:
                listBox (ttk.Treeview): The Treeview widget.
                send (list): Values to be displayed in the Treeview widget.
                num (int): The timestep
            """
            for i in send:
                listBox.insert("", "end", values=(num, i[0], i[1], i[2]))

        def update(num):
            """Updates the trajectory graph after each animation frame.

            Args:
                num (int): the frame number to be visualized.

            Returns:
               (tuple[mpl_toolkits.mplot3d.art3d.Line3D,matplotlib.text.Text]) : the updated graph and title.
            """
            data=self.trajectory.df.iloc[num:num+1]
            graph.set_data (data.x, data.y)
            graph.set_3d_properties(data.z)
            if num % self.trajectory.interval == 0:
                graph.set_color(np.random.rand(3,))
            title.set_text('3D Test, pose={}'.format(num))

            send = []
            for a in self.anchors:
                dist = a.getDist(self.trajectory.df.iloc[num])
                send.append([a.name,dist,a.error.getPDF(dist)])

            show(listBox, send, num)

            return graph, title,

        title = self.ax.text2D(0.05,0.95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=self.ax.transAxes)

        graph, = self.ax.plot(self.trajectory.df.x, self.trajectory.df.y, self.trajectory.df.z, linestyle="", marker="o")

        value_root, value_label, listBox = self.createWindow()

        ani = FuncAnimation(self.fig, update, self.trajectory.interval*(len(self.trajectory.poses)-1), interval=500//self.trajectory.interval*(len(self.trajectory.poses)-1), blit=True, repeat=False)

        plt.show()
        value_root.mainloop()

if __name__ == "__main__":
    p = np.array([(0,0),(2,2),(4,4),(100,100),(5,5)])
    x = np.array([(3.9,3.9),(4.5,4.5)])
    variance = 1
    cutoff = 10
    print(Map.addNoise(p,x,0,cutoff))


    
