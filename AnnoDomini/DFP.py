import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from numpy import linalg as la
import AnnoDomini.AutoDiff as AD
import numpy as np


class DPF:
    def __init__(self, f, x0, niter=20000, tol=10 ** (-8)):
        self.f = f
        self.vars = []
        variables = enumerate(x0)
        for i, variable in variables:
            guesses = np.zeros(len(x0))
            guesses[i] = 1
            self.vars.append(AD.AutoDiff(variable, guesses))
        self.vars = f(self.vars)
        self.Bk = np.eye(x0.shape[0])
        self.xk = x0  # x0 = (n,1) vector
        self.xk1 = x0
        self.tol = tol
        self.count = 0;
        self.niter = niter
        self.xs = []

    def check_convergence(self):
        if la.norm(self.xk1 - self.xk) < self.tol:
            return True
        else:
            return False

    def update_B(self, yk, sk):
        I = (np.eye(yk.shape[0]))
        t1 = np.outer(yk, yk.T)
        b1 = np.dot(yk.T, sk)
        gamma = 1 / b1
        t2 = I - gamma * (np.outer(yk, sk.T))
        t3 = np.dot(t2, self.Bk)
        t4 = I - gamma * (np.outer(sk, yk.T))
        t5 = np.dot(t3, t4)
        self.Bk = t5 + gamma * t1

    def dpf(self):
        count = 0
        while count < self.niter:
            self.xs.append(self.xk)
            count = count + 1
            gradient = self.vars.der
            sk = la.solve(-self.Bk, gradient)
            self.xk1 = self.xk + sk
            new_vars_list = []
            new_vars = enumerate(self.xk1)
            for i, variable in new_vars:
                guesses = np.zeros(len(self.xk1))
                guesses[i] = 1
                new_vars_list.append(AD.AutoDiff(variable, guesses))
            new_vars_list = self.f(new_vars_list)
            yk = new_vars_list.der - gradient
            if self.check_convergence():
                break
            self.update_B(yk, sk)
            # break
            self.xk = self.xk1
            self.vars = new_vars_list
        self.count = count
        return self.xk
