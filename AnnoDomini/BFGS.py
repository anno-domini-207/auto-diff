from numpy import linalg as la
import AnnoDomini.AutoDiff as AD
import numpy

'''Note: So far, this is BFGS for vector inputs, and we have to adjust for the different
number of input and output types and possible scalar and multivariable input cases.'''

class BFGS:
    def __init__(self,f,x0, niter=2000, tol=10**(-8)):
        self.f = func
        self.Bk = np.eye(x0.shape[0])
        self.xk = x0 #x0 = (n,1) vector
        self.xk1 = x0
        self.tol = tol
        self.niter = niter

    def check_convergence(self):
        if la.norm(self.xk1 - self.xk) < self.tol:
            return True
        else:
            return False

    def update_B(self,yk,sk):
        t1 = np.outer(yk,yk.T)
        b1 = np.dot(yk.T,sk)
        t2 = np.dot(np.outer(np.dot(Bk, sk), sk.T), Bk)
        b2 = np.dot(np.dot(sk.T, Bk),sk)
        self.Bk = self.Bk + (t1/b1 -  t2/b2)

    def bfgs(self):
        i = 0
        while i < self.niter:
            sk = la.solve(self.Bk, -self.J(xk))
            self.xk1 = self.xk + sk

            df1 = AD.AutoDiff(self.xk)
            df2 = AD.AutoDiff(self.xk1)
            # J(self.xk) = (self.f(df1).der
            # J(self.xk1) = (self.f(df2).der
            yk = (self.f(df2).der - (self.f(df2).der

            if self.check_convergence():
                break

            self.update_B(yk,sk)
            self.xk = self.xk1

            i += 1
        return self.xk
