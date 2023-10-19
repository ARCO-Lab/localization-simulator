from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.patches import Circle

from component import Anchor

# def test():
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')

#     def gen(n):
#         phi = 0
#         while phi < 2*np.pi:
#             yield np.array([np.cos(phi), np.sin(phi), phi])
#             phi += 2*np.pi/n

#     def update(num, data, line):
#         line.set_data(data[:2, :num])
#         line.set_3d_properties(data[2, :num])

#     N = 100
#     data = np.array(list(gen(N))).T
#     print(data[0, 0:1][0])
#     line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

#     # Setting the axes properties
#     ax.set_xlim3d([-10.0, 1.0])
#     ax.set_xlabel('X')

#     ax.set_ylim3d([-1.0, 1.0])
#     ax.set_ylabel('Y')

#     ax.set_zlim3d([0.0, 10.0])
#     ax.set_zlabel('Z')

#     ani = animation.FuncAnimation(fig, update, N, fargs=(data, line), interval=10000/N, blit=False)
#     #ani.save('matplot003.gif', writer='imagemagick')
#     return ani
#     # plt.show()


# class Map:
    
#     def __init__(self, dim) -> None:
#         self.dim = dim
#         self.nDim = len(dim)
#         self.fig, self.ax = Plot.create2D(self.dim) if self.nDim==2 else Plot.create3D(self.dim)
    
#     def placeAnchor(self, anchorList):
#         for a in anchorList:
#             self.ax.add_patch(Circle(a.location,3))


# class Plot:

#     @staticmethod
#     def create2D(dim):
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.set_xlim(0,dim[0])
#         ax.set_ylim(0,dim[1])
#         return fig, ax

#     @staticmethod
#     def create3D():
#         pass

# test()

# anchorList = [Anchor((10,10),0,0,0),Anchor((60,60),0,0,0),Anchor((80,190),0,0,0)]
# m = Map((100,200))
# m.placeAnchor(anchorList)


# t = np.linspace(0, 10, 50)
# g = -7
# v0 = 40
# z = g * t**2 / 2 + v0 * t

# scat = m.ax.scatter(t[0], z[0], c="r", s=5)

# def update(frame):
#     # for each frame, update the data stored on each artist.
#     x = t[:frame]
#     y = z[:frame]
#     # update the scatter plot:
#     data = np.stack([x, y]).T
#     scat.set_offsets(data)
#     return scat

# ani = animation.FuncAnimation(fig=m.fig, func=update, frames=50, interval=100)
# plt.show()

# a = test()
# plt.show()