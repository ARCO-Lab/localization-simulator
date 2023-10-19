import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


class Traj2D:

    def __init__(self, poses, interval) -> None:
        self.poses = poses
        self.interval = interval
        self.x = []
        self.y = []
    
    def generateTraj(self):
        for cur in range(len(self.poses)-1):
            self.x.append(np.linspace(self.poses[cur][0], self.poses[cur+1][0], self.interval))
            self.y.append(np.linspace(self.poses[cur][1], self.poses[cur+1][1], self.interval))

    def visualizeTraj(self):
        fig, ax = plt.subplots()
        xdata, ydata = [x for i in self.x for x in i], [y for i in self.y for y in i]
        ln, = plt.plot(xdata, ydata, 'ro')

        def init():
            ax.set_xlim(0, max([i[0] for i in self.poses]))
            ax.set_ylim(0, max([i[1] for i in self.poses]))
            return ln,
        
        def update(frame):
            ln.set_data(xdata[frame], ydata[frame])
            if frame % self.interval == 0:
                ln.set_color(np.random.rand(3,))
            return ln,
        
        ani = FuncAnimation(fig, update, frames=self.interval*(len(self.poses)-1),
                            init_func=init, blit=True, interval=500, repeat=False)
        plt.show()


class Traj3D(Traj2D):
    def __init__(self, poses, interval) -> None:
        super().__init__(poses, interval)
        self.z = []
    
    def generateTraj(self):
        for cur in range(len(self.poses)-1):
            self.x.append(np.linspace(self.poses[cur][0], self.poses[cur+1][0], self.interval))
            self.y.append(np.linspace(self.poses[cur][1], self.poses[cur+1][1], self.interval))
            self.z.append(np.linspace(self.poses[cur][2], self.poses[cur+1][2], self.interval))

    def visualizeTraj(self):

        xdata,ydata,zdata = [x for i in self.x for x in i], [y for i in self.y for y in i],[z for i in self.z for z in i]

        global col 

        def update(frame):
            ax.cla()

            if frame % self.interval == 0:
                global col
                col = np.random.rand(3,)
            ax.scatter(xdata[frame], ydata[frame], zdata[frame], c=col)

            ax.set_xlim(0, max([i[0] for i in self.poses]))
            ax.set_ylim(0, max([i[1] for i in self.poses]))
            ax.set_zlim(0, max([i[2] for i in self.poses]))


        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ani = FuncAnimation(fig = fig, func = update, frames = self.interval*(len(self.poses)-1), interval = 500, repeat=False)

        plt.show()


t = Traj3D([(0,0,0),(8,8,8),(6,6,6)],6)
t.generateTraj()
t.visualizeTraj()