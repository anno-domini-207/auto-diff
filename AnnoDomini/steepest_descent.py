import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from numpy import linalg as la
import AnnoDomini.AutoDiff as AD
import numpy as np
import scipy.optimize

class SteepestDescent:
    def __init__(self, func, x0, step=1, tol=10**(-30), maxiter=2000):
        self.f = func
        self.xk = x0
        self.xk1 = x0
        self.tol = tol
        self.maxiter = maxiter
        self.step = step
        self.vars = []
        variables = enumerate(x0)
        for i, variable in variables:
            guesses = np.zeros(len(x0))
            guesses[i] = 1
            self.vars.append(AD.AutoDiff(variable, guesses))
        self.vars = self.f(self.vars)

    def update_xk(self):
        self.xk1 = self.xk - self.step*self.vars.der
        new_vars_list = []
        new_vars = enumerate(self.xk1)
        for i, variable in new_vars:
            guesses = np.zeros(len(self.xk1))
            guesses[i] = 1
            new_vars_list.append(AD.AutoDiff(variable, guesses))
        new_vars_list = self.f(new_vars_list)

        self.xk = self.xk1
        self.vars = new_vars_list

    def find_root(self):
        i = 0
        self.xs = [self.xk]
        while self.step > self.tol and i <= self.maxiter:
            sk = self.vars.der
            dif = self.xk - self.step*sk
            dif_f = self.f(dif)
            if dif_f < self.f(self.xk):
                # print("here")
                self.xs.append(self.xk1)
                self.update_xk()
            else:
                self.step = self.step/2.

        return self.xk1

# def f(args):
#     [x,y] = args
#     ans = 100*(y-x**2)**2 + (1-x)**2
#     return ans
#
# x0 = [0,1]
# sd = SteepestDescent(f, x0)
# print(sd.find_root())
