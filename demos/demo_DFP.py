import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../AnnoDomini')))

import AnnoDomini.AutoDiff as AD
import numpy as np
from matplotlib import pyplot as plt
from DFP import DFP


def f(args):
    [x, y] = args
    return np.e**(x+1) + np.e**(-y+1) + (x-y)**2

x0 = [2, 0]
sd = DFP(f, x0)
root = sd.find_root()
print(root)
traj = sd.xs



def plot_dfp(ans):
    X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-2, 8, 100))
    Z = f(np.array([X, Y]))
    xmesh, ymesh = np.mgrid[-4:4:80j, -4:4:80j]
    fmesh = f(np.array([xmesh, ymesh]))
    fig = plt.figure()
    plt.title('DPF Path for $f(x,y) = e^{x+1} + e^{1-y} + {(x-y)}^2$ Starting at [2,0]')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.contour(xmesh, ymesh, fmesh, 50)
    it_array = np.array(ans)
    # plt.plot(it_array.T[0], it_array.T[1], "x-")
    plt.plot(it_array.T[0], it_array.T[1], "x-", label="Path")
    plt.plot(it_array.T[0][0], it_array.T[1][0], 'xr', label='Initial Guess', markersize=12)
    plt.plot(it_array.T[0][-1], it_array.T[1][-1], 'xg', label='Solution', markersize=12)
    plt.legend()
    plt.show()


plot_dfp(traj)
