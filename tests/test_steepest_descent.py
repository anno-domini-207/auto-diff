import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../AnnoDomini')))

#from AnnoDomini.AutoDiff import AutoDiff as AD
from AnnoDomini.AutoDiff import AutoDiff as AD
from steepest_descent import SteepestDescent
import numpy as np

def test_initial_guess1():
    x0 = [2,1]
    def f(args):
        [x,y] = args
        ans = 100*(y-x**2)**2 + (1-x)**2
        return ans
    demo = SteepestDescent(f,x0)
    root_est = demo.find_root()
    real_root = np.array([1.,1.])
    assert real_root.all() == root_est.all()

def test_initial_guess2():
    x0 = [0,1]
    def f(args):
        [x,y] = args
        ans = 100*(y-x**2)**2 + (1-x)**2
        return ans
    demo = SteepestDescent(f,x0)
    root_est = demo.find_root()
    real_root = np.array([1.,1.])
    assert real_root.all()  == root_est.all()

def test_initial_guess3():
    x0 = [-1,1]
    def f(args):
        [x,y] = args
        ans = 100*(y-x**2)**2 + (1-x)**2
        return ans
    demo = SteepestDescent(f,x0)
    root_est = demo.find_root()
    real_root = np.array([1.,1.])
    assert real_root.all()  == root_est.all() 
