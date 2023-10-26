import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd



class Traj2D:

    def __init__(self, poses, interval) -> None:
        self.poses = poses
        self.interval = interval
        self.data = {
            "x":[],
            "y":[]
        }
        self.df = None
    
    def generateTraj(self):
        for cur in range(len(self.poses)-1):
            self.data["x"].extend(np.linspace(self.poses[cur][0], self.poses[cur+1][0], self.interval))
            self.data["y"].extend(np.linspace(self.poses[cur][1], self.poses[cur+1][1], self.interval))
        self.df = pd.DataFrame(self.data)

    def visualizeTraj(self):

        def init():
            ax.set_xlim(0, max([i[0] for i in self.poses]))
            ax.set_ylim(0, max([i[1] for i in self.poses]))
            return ln,
        
        def update(num):
            data = self.df.iloc[num:num+1]
            ln.set_data(data.x, data.y)
            if num % self.interval == 0:
                ln.set_color(np.random.rand(3,))
            title.set_text('2D Test, pose={}'.format(num))
            return ln,title,
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        title = ax.text(0.5,0.90, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")

        ln, = ax.plot(self.df.x, self.df.y, 'ro')
        
        ani = FuncAnimation(fig, update, frames=self.interval*(len(self.poses)-1),
                            init_func=init, blit=True, interval=500//self.interval*(len(self.poses)-1), repeat=False)
        plt.show()


class Traj3D(Traj2D):
    def __init__(self, poses, interval) -> None:
        super().__init__(poses, interval)
        self.data = {
            "x":[],
            "y":[],
            "z":[]
        }
    
    def generateTraj(self):
        for cur in range(len(self.poses)-1):
            self.data["x"].extend(np.linspace(self.poses[cur][0], self.poses[cur+1][0], self.interval))
            self.data["y"].extend(np.linspace(self.poses[cur][1], self.poses[cur+1][1], self.interval))
            self.data["z"].extend(np.linspace(self.poses[cur][2], self.poses[cur+1][2], self.interval))
        self.df = pd.DataFrame(self.data)

    def visualizeTraj(self):


        def update_graph(num):
        
            data=self.df.iloc[num:num+1]
            graph.set_data (data.x, data.y)
            graph.set_3d_properties(data.z)
            if num % self.interval == 0:
                graph.set_color(np.random.rand(3,))
            title.set_text('3D Test, pose={}'.format(num))
            return graph, title,


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        title = ax.text2D(0.05,0.95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes)


        graph, = ax.plot(self.df.x, self.df.y, self.df.z, linestyle="", marker="o")

        ani = FuncAnimation(fig, update_graph, self.interval*(len(self.poses)-1), interval=500//self.interval*(len(self.poses)-1), blit=True, repeat=False)

        plt.show()

if __name__ == "__main__":
    t = Traj2D([(0,0),(8,8),(6,6),(4,5),(10,4)],6)
    # t = Traj3D([(0,0,0),(8,8,8),(6,6,7),(4,5,4),(10,4,2)],12)
    t.generateTraj()
    t.visualizeTraj()