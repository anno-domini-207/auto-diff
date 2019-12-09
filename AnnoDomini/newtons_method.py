import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import AnnoDomini.AutoDiff as AD
import numpy as np
from scipy import linalg as la
class Newton:
    def __init__(self, func, x0, alpha=1, tol=10**(-8), maxiter=50):
        # When `val` is a list of `AutoDiff` objects (we assume/expect a list of homogeneous objects)
        self.f = func
        self.xold = x0
        self.xnew = x0
        self.tol = tol
        self.maxiter = maxiter
        self.alpha = alpha

    def check_convergence(self):
        if la.norm(self.xnew - self.xold) < self.tol:
            return True
        else:
            return False

    def update_x(self,df):
        self.xnew = self.xold - self.alpha * self.f(self.xold)/ self.f(df).der

    def find_root(self):
        for i in range(self.maxiter):
            df = AD.AutoDiff(self.xold)
            self.update_x(df)
            if self.check_convergence():
                return self.xnew
            else:
                self.xold = self.xnew
        return self.xnew

# if __name__ == '__main__':
#
#     import numpy as np
#     from matplotlib import pyplot as plt
#
#     f = lambda x: np.sin(x) + x * np.cos(x)
#     x0 = -3
#     N = Newton(f,x0)
#     ans = N.scalar_newton()
#
#     # plot solution
#     xs = np.linspace(-7,5,100)
#     plt.plot(xs, f(xs), label="f")
#     plt.scatter(ans, f(ans),label="Root", color = 'black')
#     plt.scatter(x0, f(x0),label="initial", color = 'red')
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.title("Visual of Newton's Method on $sin(x) + x * cos(x)$")
#     plt.axhline(y = 0, color = 'red')
#     plt.legend()
#     #plt.savefig('newtons_method.png')
