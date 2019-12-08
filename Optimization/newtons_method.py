from numpy import linalg as la
import AnnoDomini.AutoDiff as AD

'''Note: So far, this is Newton for scara inputs, and we have to adjust for the different
number of input and output types, and multivariable case.'''

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
        self.xnew = self.xold - self.alpha * self.f(xold)/ self.f(df).der

    def scalar_newton(self):
        for i in range self.maxiter:
            df = AD.AutoDiff(self.xold)
            self.update_x(df)
            if self.check_convergence():
                return self.xnew
            else:
                self.xold = self.xnew
        return self.xnew
