import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Optimization')))

from numpy import linalg as la
import AnnoDomini.AutoDiff as AD
import numpy as np
import numpy



class DPF:
    def __init__(self, f, x0, niter=2000, tol=10 ** (-8)):
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
        self.niter = niter

    def check_convergence(self):
        if la.norm(self.xk1 - self.xk) < self.tol:
            return True
        else:
            return False

    def update_B(self, yk, sk):
        I = (np.eye(yk.shape[0]))
        t1 = np.outer(yk, yk.T)
        b1 = np.dot(yk.T, sk)
        # b1 = 1 / np.dot(yk.T, sk)
        gamma = 1 / b1

        t2 = I - gamma*(np.outer(yk, sk.T))
        # print(yk.shape, sk.shape, gamma.shape)
        # print(self.Bk.shape, b1)

        t3 = np.dot(t2, self.Bk)
        t4 = I - gamma*(np.outer(sk, yk.T))
        # print(t4.shape)
        t5 = np.dot(t3, t4)
        self.Bk = t5 + gamma*t1

    def dpf(self):
        count = 0
        while count < self.niter:
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
            new_vars_list = f(new_vars_list)
            yk = new_vars_list.der - gradient
            if self.check_convergence():
                break
            self.update_B(yk, sk)
            # break
            self.xk = self.xk1
            self.vars = new_vars_list
        print(count)
        return self.xk


def f(args):
    [x, y] = args
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


test = DPF(f, np.array([-1, 1])).dpf()

print(test)
