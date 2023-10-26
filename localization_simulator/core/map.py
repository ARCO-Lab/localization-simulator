from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import pandas as pd

from .component import Anchor
from ..utils.trajectory import Traj2D, Traj3D

class Map:
    
    def __init__(self, dim) -> None:
        self.dim = dim
        self.nDim = len(dim)
        self.fig, self.ax = Plot.create2D(self.dim) if self.nDim==2 else Plot.create3D(self.dim)
        self.trajectory = None

    # PLACEHOLDER UNTIL SUITABLE REPLACEMENT IS FOUND TO REPRESENT ANCHORS ON 3D PLOT
    def plt_sphere(self,list_center, list_radius):
        for c, r in zip(list_center, list_radius):
            
            # draw sphere
            u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
            x = r*np.cos(u)*np.sin(v)
            y = r*np.sin(u)*np.sin(v)
            z = r*np.cos(v)

            self.ax.plot_surface(x-c[0], y-c[1], z-c[2], color="green", alpha=0.5*np.random.random()+0.5)
    
    def placeAnchor(self, anchorList):
        if self.nDim == 2:
            for a in anchorList:
                self.ax.add_patch(Rectangle(a.location,0.5,0.5,fc="black",ec=a.clr))
        else:
            for a in anchorList:
                self.ax.scatter(a.location[0], a.location[1], a.location[2], c='black', edgecolors=a.clr ,marker='o', s=100)


    def loadTraj(self, poses, interval):
        if self.nDim == 2:
            self.trajectory = Traj2D(poses, interval)
        else:
            self.trajectory = Traj3D(poses, interval)
        self.trajectory.generateTraj()

    def visualize2D(self):

        def update(num):
            data = self.trajectory.df.iloc[num:num+1]
            ln.set_data(data.x, data.y)
            if num % self.trajectory.interval == 0:
                ln.set_color(np.random.rand(3,))
            title.set_text('2D Test, pose={}'.format(num))
            return ln,title,


        title = self.ax.text(0.5,0.90, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=self.ax.transAxes, ha="center")

        ln, = self.ax.plot(self.trajectory.df.x, self.trajectory.df.y, 'ro')
        
        ani = FuncAnimation(self.fig, update, frames=self.trajectory.interval*(len(self.trajectory.poses)-1),
                            blit=True, interval=500//self.trajectory.interval*(len(self.trajectory.poses)-1), repeat=False)
        plt.show()

    
    def visualize3D(self):

        def update(num):
            data=self.trajectory.df.iloc[num:num+1]
            graph.set_data (data.x, data.y)
            graph.set_3d_properties(data.z)
            if num % self.trajectory.interval == 0:
                graph.set_color(np.random.rand(3,))
            title.set_text('3D Test, pose={}'.format(num))
            return graph, title,

        title = self.ax.text2D(0.05,0.95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=self.ax.transAxes)


        graph, = self.ax.plot(self.trajectory.df.x, self.trajectory.df.y, self.trajectory.df.z, linestyle="", marker="o")

        ani = FuncAnimation(self.fig, update, self.trajectory.interval*(len(self.trajectory.poses)-1), interval=500//self.trajectory.interval*(len(self.trajectory.poses)-1), blit=True, repeat=False)

        plt.show()

class Plot:

    @staticmethod
    def create2D(dim):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0,dim[0])
        ax.set_ylim(0,dim[1])
        return fig, ax

    @staticmethod
    def create3D(dim):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.axes.set_xlim3d(left=0, right=dim[0]) 
        ax.axes.set_ylim3d(bottom=0, top=dim[1]) 
        ax.axes.set_zlim3d(bottom=0, top=dim[2]) 
        return fig, ax




if __name__ == "__main__":
    # anchorList = [Anchor((3,3),0,0,0,"red"),Anchor((14,14),0,0,0,"blue"),Anchor((18,12),0,0,0,"red")]
    # m = Map((20,20))
    # m.placeAnchor(anchorList)
    # m.loadTraj([(0,0),(8,8),(6,6),(4,5),(10,4)],6)
    # m.visualize2D()

    anchorList = [Anchor((3,3,3),0,0,0,"red"),Anchor((14,14,14),0,0,0,"blue"),Anchor((18,12,13),0,0,0,"red")]
    m = Map((20,20,20))
    m.placeAnchor(anchorList)
    m.loadTraj([(0,0,0),(8,8,8),(6,6,7),(4,5,4),(10,4,2)],12)
    m.visualize3D()