import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AnnoDomini.AutoDiff import AutoDiff as AD
from AnnoDomini.BFGS import BFGS

def f(args):
    [x, y] = args
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2




def test_bfgs():
    demo = BFGS(f, np.array([-1, 1]))
    roots = demo.bfgs()
    assert ((roots[0] - 1) < 1e-6 and (roots[1] - 1) < 1e-6)
    demo = BFGS(f, np.array([0, 1]))
    roots = demo.bfgs()
    assert ((roots[0] - 1) < 1e-6 and (roots[1] - 1) < 1e-6)
    demo = BFGS(f, np.array([2, 1]))
    roots = demo.bfgs()
    assert ((roots[0] - 1) < 1e-6 and (roots[1] - 1) < 1e-6)



