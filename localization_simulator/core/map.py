from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import pandas as pd

from .plot import Plot
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

            guess = np.array([np.random.uniform(0, self.dim[0]), np.random.uniform(0, self.dim[1])])
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

if __name__ == "__main__":
    pass


    
