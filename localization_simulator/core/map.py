from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import pandas as pd
import argparse

from .component import Anchor
from ..utils.trajectory import Traj2D, Traj3D
from .noise import Noise
import tkinter as tk
from tkinter import ttk

class Map:
    
    def __init__(self, dim) -> None:
        self.dim = dim
        self.nDim = len(dim)
        self.fig, self.ax = Plot.create2D(self.dim) if self.nDim==2 else Plot.create3D(self.dim)
        self.trajectory = None
    
    def placeAnchor(self, anchorList):
        self.anchors = anchorList
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

    def createWindow(self):
        root = tk.Tk()
        root.title('Pose Tracker')

        value_label = tk.Label(root, text="Pose Tracker", font=("Arial",30)).grid(row=0, columnspan=4)

        cols = ('Pose', 'Anchor', 'Distance', 'Bias')
        listBox = ttk.Treeview(root, columns=cols, show='headings')
        # set column headings
        for col in cols:
            listBox.heading(col, text=col)    
        listBox.grid(row=1, column=0, columnspan=2)

        return root, value_label, listBox

    def visualize2D(self):

        def show(listBox, send, num):
            for i in send:
                listBox.insert("", "end", values=(num,i[0], i[1], i[2]))

        def update(num):
            data = self.trajectory.df.iloc[num:num+1]
            ln.set_data(data.x, data.y)
            if num % self.trajectory.interval == 0:
                ln.set_color(np.random.rand(3,))
            title.set_text('2D Test, pose={}'.format(num))

            send = []
            for a in self.anchors:
                send.append([a.name,np.linalg.norm(self.trajectory.df.iloc[num]-a.location),a.noise(np.linalg.norm(self.trajectory.df.iloc[num]-a.location))])

            show(listBox, send, num)
            return ln,title,


        title = self.ax.text(0.5,0.90, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=self.ax.transAxes, ha="center")

        ln, = self.ax.plot(self.trajectory.df.x, self.trajectory.df.y, 'ro')

        value_root, value_label, listBox = self.createWindow()
        
        ani = FuncAnimation(self.fig, update, frames=self.trajectory.interval*(len(self.trajectory.poses)-1),
                            blit=True, interval=500//self.trajectory.interval*(len(self.trajectory.poses)-1), repeat=False)
        
        plt.show()
        value_root.mainloop()
    
    def visualize3D(self):
    
        def show(listBox, send, num):
            for i in send:
                listBox.insert("", "end", values=(num, i[0], i[1], i[2]))

        def update(num):
            data=self.trajectory.df.iloc[num:num+1]
            graph.set_data (data.x, data.y)
            graph.set_3d_properties(data.z)
            if num % self.trajectory.interval == 0:
                graph.set_color(np.random.rand(3,))
            title.set_text('3D Test, pose={}'.format(num))

            send = []
            for a in self.anchors:
                send.append([a.name,np.linalg.norm(self.trajectory.df.iloc[num]-a.location),a.noise(np.linalg.norm(self.trajectory.df.iloc[num]-a.location))])

            show(listBox, send, num)

            return graph, title,

        title = self.ax.text2D(0.05,0.95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=self.ax.transAxes)


        graph, = self.ax.plot(self.trajectory.df.x, self.trajectory.df.y, self.trajectory.df.z, linestyle="", marker="o")

        value_root, value_label, listBox = self.createWindow()

        ani = FuncAnimation(self.fig, update, self.trajectory.interval*(len(self.trajectory.poses)-1), interval=500//self.trajectory.interval*(len(self.trajectory.poses)-1), blit=True, repeat=False)

        plt.show()
        value_root.mainloop()

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


def parseArgs():
        retArgs = []
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--2", "-dim2", default = False, required = False, action="store_true", help="Flag to show 2D.")
        parser.add_argument("--3", "-dim3", default = False, required = False, action="store_true", help="Flag to show 3D.")
        args = parser.parse_args()
        for arg in vars(args):
             retArgs.append(getattr(args, arg))
        return retArgs
 


if __name__ == "__main__":

    def routine2d():
        anchorList2d = [Anchor("a",(3,3),0,0,Noise.gonzalez,"red"),Anchor("b",(14,14),0,0,Noise.gonzalez,"blue"),Anchor("c",(18,12),0,0,Noise.gonzalez,"red")]
        m = Map((20,20))
        m.placeAnchor(anchorList2d)
        m.loadTraj([(0,0),(8,8),(6,6),(4,5),(10,4)],6)
        m.visualize2D()

    def routine3d():
        anchorList3d = [Anchor("a",(3,3,3),0,0,Noise.gonzalez,"red"),Anchor("b",(14,14,14),0,0,Noise.gonzalez,"blue"),Anchor("c",(18,12,13),0,0,Noise.gonzalez,"red")]
        m = Map((20,20,20))
        m.placeAnchor(anchorList3d)
        m.loadTraj([(0,0,0),(8,8,8),(6,6,7),(4,5,4),(10,4,2)],12)
        m.visualize3D()

    use2d, use3d = parseArgs()

    if use2d == use3d:
        np.random.choice([routine2d,routine3d])()
    elif use2d:
        routine2d()
    else:
        routine3d()


    
