from numpy import linalg as la
import AnnoDomini.AutoDiff as AD

'''Note: Below, I use the scipy.optimize.line_search. Not sure how we want to handle this -
 because we have to pass in a function for the derivitive, and I am not sure how to do that
 with our package.An alternative algorithm for steepest descent that doesnt need scipy.optimize is given below:

def steepestDescent(f, Df, x0, step=1, tol=.0001, maxiters=50):
    """Use the Method of Steepest Descent to find the minimizer x of the convex
    function f:Rn -> R.

    Parameters:
        f (function Rn -> R): The objective function.
        Df (function Rn -> Rn): The gradient of the objective function f.
        x0 ((n,) ndarray): An initial guess for x.
        step (float): The initial step size.
        tol (float): The convergence tolerance.

    Returns:
        x ((n,) ndarray): The minimizer of f.
    """
    i = 0
    xk = x0
    while step > tol and i <= maxiters:
        if f(xk - step*Df(xk)) < f(xk):
            xk1 = xk - step*Df(xk)
            xk = xk1
        else:
            step = step/2.
    return xk1

    '''

class SteepestDescent:
    def __init__(self, func, x0, step=1, tol=10**(-8), maxiter=2000):
        self.f = func
        self.xk = x0
        self.xk1 = x0
        self.tol = tol
        self.maxiter = maxiter
        self.step = step

    def find_sk(self):
        df = AD.AutoDiff(self.xk)
        sk = -self.f(df).der
        return sk

    def update_xk(self,nk,sk):
        self.xk1 = self.xk + nk*sk

    def steepestDescent(f, x0, step=1, tol=10**(-8), maxiters=2000):
        i = 0
        while i < maxiters:
            sk = self.find_sk()
            # need to pass in the function for Df - not sure how to do this with our package
            nk = scipy.optimize.line_search(self.f, Df, xk, sk)[0]
            self.update_xk(self,nk,sk)
            if la.norm(self.xk1 - self.xk) < tol:
                break
            self.xk = self.xk1
            i+=1
        return self.xk
