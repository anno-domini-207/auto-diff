import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import AnnoDomini.AutoDiff as AD
import numpy as np
from scipy import linalg as la
class Newton:
    def __init__(self, func, x0, alpha=0.5, tol=10**(-8), maxiter=50):
        # When `val` is a list of `AutoDiff` objects (we assume/expect a list of homogeneous objects)
        self.f = func
        if not isinstance(x0, list):
            x0 = [x0]
        self.xold = x0
        self.xnew = x0
        self.tol = tol
        self.maxiter = maxiter
        self.alpha = alpha
        self.m = len(x0) if isinstance(x0, list) else 1

    def check_convergence(self):
        if la.norm(self.xnew - self.xold) < self.tol:
            return True
        else:
            return False

    def update_x(self, *df):
        self.xnew = self.xold - (self.f(*df).der) ** -1 * self.alpha * self.f(*self.xold)

    def find_root(self):
        for i in range(self.maxiter):
            if self.m == 1:
                df = AD.AutoDiff(self.xold)
                self.update_x(df)
            else: #pragma: no cover
                dfs = [0] * self.m
                for i in range(self.m):
                    ders = [0] * self.m
                    ders[i] = 1
                    dfs[i] = AD.AutoDiff(self.xold[i], ders)
                self.update_x(*dfs)
            if self.check_convergence():
                return self.xnew
            else:
                self.xold = self.xnew
        return self.xnew