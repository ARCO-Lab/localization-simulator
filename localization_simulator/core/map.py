from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import pandas as pd
import argparse

from .component import Anchor
from ..utils.trajectory import Traj2D, Traj3D
from ..utils.nls import NLS
from .error import Error
import tkinter as tk
from tkinter import ttk

class Map:
    
    def __init__(self, dim) -> None:
        self.dim = dim
        self.nDim = len(dim)
        self.fig, self.ax = Plot.create2D(self.dim) if self.nDim==2 else Plot.create3D(self.dim)
        self.trajectory = None
        self.points = []
        self.gradNorms = []
    
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

        cols = ('Pose', 'Anchor', 'Distance', 'P(dist; mean, std_dev)')
        listBox = ttk.Treeview(root, columns=cols, show='headings')
        # set column headings
        for col in cols:
            listBox.heading(col, text=col)    
        listBox.grid(row=1, column=0, columnspan=2)

        return root, value_label, listBox

    def visualize2D(self):

        nls = NLS(self.points,self.gradNorms, np.array([a.location for a in self.anchors]),variance=0.01, max_error=0.1)

        def show(listBox, send, num):
            for i in send:
                listBox.insert("", "end", values=(num,i[0], i[1], i[2]))

        def update(num):
            data = self.trajectory.df.iloc[num:num+1]
            ln.set_data(data.x, data.y)
            if num % self.trajectory.interval == 0:
                ln.set_color(np.random.rand(3,))
            title.set_text('2D Test, pose={}'.format(num))

            # guess = np.array([np.random.uniform(0, self.dim[0]), np.random.uniform(0, self.dim[1])])
            guess = np.array([10.39,4.22])
            nls.process(np.array([data.x.item(),data.y.item()]), guess)

            send = []
            for a in self.anchors:
                dist = a.getDist(self.trajectory.df.iloc[num])
                send.append([a.name,dist,a.error.getPDF(dist)])

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
 
import matplotlib.pyplot as plt

def postPlot(points, gradNorms):
    for i, (p, gn) in enumerate(zip(points, gradNorms), start=1):
        x, y = zip(*p)
        fig, axs = plt.subplots(3)
        fig.suptitle(f"Approx for {x[0]},{y[0]}")

        axs[0].scatter(x[1], y[1], color='green', marker='o', label="Initial")

        axs[0].scatter(x[-1], y[-1], color='blue', marker='x', label="Final")

        axs[0].scatter(x[2:-1], y[2:-1], color='red', marker='x')

        for i, (xi, yi) in enumerate(zip(x[2:-1], y[2:-1]), start=1):
            axs[0].text(xi, yi, str(i), ha='right', va='bottom')

        axs[1].plot(range(1, len(gn) + 1), gn, color='purple', marker='o', label='Grad Norms')

        log_gn = np.log(gn)
        axs[2].plot(range(1, len(log_gn) + 1), log_gn, color='orange', marker='o', label='Log Grad Norms')

        plt.show(block=False)
        input("Press a key ")
        plt.close()


if __name__ == "__main__":

    def routine2d():
        anchorList2d = [Anchor("a",(3,3),0,(0,3),"red"),Anchor("b",(14,14),0,(0,3),"blue"),Anchor("c",(18,12),0,(0,3),"red")]
        m = Map((20,20))
        m.placeAnchor(anchorList2d)
        m.loadTraj([(0,0),(8,8),(6,6),(4,5),(10,4)],6)
        m.visualize2D()
        postPlot(m.points,m.gradNorms)

    def routine3d():
        anchorList3d = [Anchor("a",(3,3,3),0,(0,3),"red"),Anchor("b",(14,14,14),0,(0,3),"blue"),Anchor("c",(18,12,13),0,(0,3),"red")]
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


    
