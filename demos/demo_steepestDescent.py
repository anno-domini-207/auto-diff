import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../AnnoDomini')))

import AnnoDomini.AutoDiff as AD
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
from steepest_descent import SteepestDescent
from matplotlib import pyplot as plt

def f(args):
    [x,y] = args
    ans = 100*(y-x**2)**2 + (1-x)**2
    return ans

x0 = [2,1]
sd = SteepestDescent(f, x0)
root = sd.find_root()
traj = sd.xs

x = [traj[i][0] for i in range(len(traj))]
y = [traj[i][1] for i in range(len(traj))]
traj2 = np.concatenate((np.array(x).reshape(-1,1),np.array(y).reshape(-1,1)), axis=1)
# print(root)
# print(traj2)

def plot_descent(ans):
    X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-2, 8, 100))
    Z = f(np.array([X,Y]))
    fig = plt.subplots(1,1, figsize = (10,7))
    plt.contour(X, Y, Z)
    plt.plot(ans[:,0], ans[:,1], "-.", label="Trajectory")
    plt.scatter(root[0],root[1], label="Root", c="red")
    plt.scatter(-1,1, label="Initial Guess", c ="orange")
    plt.title("Convergence of Steepest Descent on Rosenbrock Function")
    plt.xlim(-3, 3)
    plt.ylim(-2, 8)
    plt.legend()
    #plt.show()
    plt.savefig("steepestDescent.png")

plot_descent(traj2)
