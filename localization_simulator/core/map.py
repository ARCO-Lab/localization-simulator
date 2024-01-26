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
import pandas as pd
import random

from .plot import Plot
from .inf import isotropic, newAnchorPos
from .component import Anchor
from ..utils.trajectory import Traj2D, Traj3D
from ..utils.nls import NLS
from .error import Error
import tkinter as tk
from tkinter import ttk

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
        self.fig, self.ax = Plot.create2D(self.dim) if self.nDim==2 else Plot.create3D(self.dim)
        self.trajectory = None
        self.points = []
        self.gradNorms = []
        self.isotropic = None
    
    def placeAnchor(self, anchorList):
        """Method for placing anchors in a map.

        Args:
            anchorList (list[Anchor]): The list of anchors to place in a map.
        """
        self.anchors = anchorList
        self.isotropic = isotropic(self.nDim,random.uniform(1,2))
        if self.nDim == 2:
            for a in anchorList:
                self.ax.add_patch(Rectangle(a.location,0.5,0.5,fc="black",ec=a.clr))
        else:
            for a in anchorList:
                self.ax.scatter(a.location[0], a.location[1], a.location[2], c='black', edgecolors=a.clr ,marker='o', s=100)
            
    def loadTraj(self, poses, interval):
        """Method for loading a trajectory into a map

        Args:
            poses (list[tuple[float]]): The desired poses for the trajectory.
            interval (int): The number of substeps in between each pose. Can be thought as the number of points in a straight line between poses.
        """
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

    def visualize2D(self):
        """Visualizes the trajectory (2D) using Matplotlib animation
        """
        nls = NLS(self.points,self.gradNorms,variance=0.01, tolerance=0.1)

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

            newAncs = []
            for i in newAnchorPos([a.location for a in self.anchors], self.isotropic):
                newAncs.append(i)
            anc.set_data([i[0] for i in newAncs],[i[1] for i in newAncs])

            if num % self.trajectory.interval == 0:
                ln.set_color(np.random.rand(3,))
            title.set_text('2D Test, pose={}'.format(num))

            guess = np.array([np.random.uniform(0, self.dim[0]), np.random.uniform(0, self.dim[1])])
            nls.process(np.array([data.x.item(),data.y.item()]), guess, np.array(newAncs))

            send = []
            for a, a_new in zip(self.anchors, newAncs):
                print(f"{a.location} vs {a_new}")
                dist = a.getDist(self.trajectory.df.iloc[num], tuple(a_new))
                send.append([a.name,dist,a.error.getPDF(dist)])

            show(listBox, send, num)
            return ln,anc,title,

        title = self.ax.text(0.5,0.90, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=self.ax.transAxes, ha="center")

        ln, = self.ax.plot(self.trajectory.df.x, self.trajectory.df.y, 'ro')
        anc, = self.ax.plot([],[], 'bx')

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
    pass


    
