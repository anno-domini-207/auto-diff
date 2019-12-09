import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from AnnoDomini.AutoDiff import AutoDiff as AD
from AnnoDomini.DFP import DPF

def f(args):
    [x, y] = args
    return np.e**(x+1) + np.e**(-y+1) + (x-y)**2




def test_dfp():
    demo = DPF(f, np.array([0, 0]))
    roots = demo.dpf()
    assert ((roots[0] + 0.438378) < 1e-6 and (roots[1] - 0.438378) < 1e-6)
    demo = DPF(f, np.array([1, 1]))
    roots = demo.dpf()
    assert ((roots[0] + 0.438378) < 1e-6 and (roots[1] - 0.438378) < 1e-6)
    demo = DPF(f, np.array([10, -10]))
    roots = demo.dpf()
    assert ((roots[0] + 0.438378) < 1e-6 and (roots[1] - 0.438378) < 1e-6)



