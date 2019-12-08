from numpy import linalg as la
import AnnoDomini.AutoDiff as AD
import numpy as np
import numpy

'''Note: So far, this is BFGS for vector inputs, and we have to adjust for the different
number of input and output types and possible scalar and multivariable input cases.'''


class BFGS:
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
        t1 = np.outer(yk, yk.T)
        b1 = np.dot(yk.T, sk)
        t2 = np.dot(np.outer(np.dot(self.Bk, sk), sk.T), self.Bk)
        b2 = np.dot(np.dot(sk.T, self.Bk), sk)
        self.Bk = self.Bk + (t1 / b1 - t2 / b2)

    def bfgs(self):
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
            self.xk = self.xk1
            self.vars = new_vars_list
        return self.xk


def f(args):
    [x, y] = args
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


test = BFGS(f, np.array([-1, 1])).bfgs()

print(test)
