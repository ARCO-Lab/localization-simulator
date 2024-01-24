from matplotlib import pyplot as plt
import numpy as np

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

def postPlot(points, gradNorms):
    for i, (p, gn) in enumerate(zip(points, gradNorms), start=1):
        x, y = zip(*p) 

        plt.scatter(x[1], y[1], color='green', marker='o', label="Initial")
        plt.scatter(x[-1], y[-1], color='blue', marker='x', label="Final")
        plt.scatter(x[2:-1], y[2:-1], color='red', marker='x')
        for i, (xi, yi) in enumerate(zip(x[2:-1], y[2:-1]), start=1):
           plt.text(xi, yi, str(i), ha='right', va='bottom')
        plt.title(f"Approx for {x[0]},{y[0]}")
        plt.legend()
        # plt.show()
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        fig, axs = plt.subplots(2)
        fig.suptitle(f"Approx for {x[0]},{y[0]}")
        axs[0].plot(range(1, len(gn) + 1), gn, color='purple', marker='o', label='Grad Norms')
        log_gn = np.log(gn)
        axs[1].plot(range(1, len(log_gn) + 1), log_gn, color='orange', marker='o', label='Log Grad Norms')

        plt.show(block=False)
        # input("Press a key ")
        plt.pause(2)
        plt.close()