import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Optimization')))

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

x0 = [0,1]
sd = SteepestDescent(f, x0)
print(sd.find_root())
