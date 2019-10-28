# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 00:46:08 2019

@author: DavidYQY
"""
import numpy as np
import functools

class Dual:
    def __init__(self, real=0, dual=1):
        self.real = real
        self.dual = dual
        
    def __add__(self, other):
        otherreal = getattr(other, 'real', other)
        otherdual = getattr(other, 'dual', None)
        ret = Dual(self.real + otherreal, self.dual)
        if otherdual is not None:
            ret.dual += otherdual
        return ret
    def __sub__(self, other):
        otherreal = getattr(other, 'real', other)
        otherdual = getattr(other, 'dual', None)
        ret = Dual(self. real - otherreal, self.dual)
        if otherdual is not None:
            ret.dual -= otherdual
        return ret
    def __mul__(self, other):
        otherreal = getattr(other, 'real', other)
        otherdual = getattr(other, 'dual', None)
        ret = Dual(self.real * otherreal, self.dual * otherreal)
        if otherdual is not None:
            ret.dual += self.real*otherdual
        return ret
    def __truediv__(self, other):
        otherreal = getattr(other, 'real', other)
        otherdual = getattr(other, 'dual', None)
        ret = Dual(self. real / otherreal, self.dual / otherreal)
        if otherdual is not None:
            ret.dual -= self.real*otherdual/(otherreal*otherreal)
        return ret
    
    def __pow__(self, other):
        if isinstance(other, int):
            n = other
            ret = Dual(self.real **n, n * self.real **(n-1) * self.dual)
            return ret

    def __radd__(self, other):
        return self.__add__(other)
    def __rsub__(self, other):
        return -self.__sub__(other)
    def __rmul__(self, other):
        return self.__mul__(other)

    def __rpow__(self, other):
        x = Dual(other, nvars=self.nvars)
        return x.__pow__(self)

    def __iadd__(self, other):
        otherreal = getattr(other, 'real', other)
        otherdual = getattr(other, 'dual', None)
        self.real += otherreal
        if otherdual is not None:
            self.dual += other.dual
        return self
    def __isub__(self, other):
        otherreal = getattr(other, 'real', other)
        otherdual = getattr(other, 'dual', None)
        self.real -= otherreal
        if otherdual is not None:
            self.dual -= other.dual
        return self
    # object.__imul__(self, other)
    # object.__itruediv__(self, other)
    # object.__ifloordiv__(self, other)
    # object.__imod__(self, other)
    # object.__ipow__(self, other[, modulo])
    # object.__ilshift__(self, other)
    # object.__irshift__(self, other)
    # object.__iand__(self, other)
    # object.__ixor__(self, other)
    # object.__ior__(self, other)

    def __neg__(self):
        return Dual(-self.real, -self.dual)
    # object.__pos__(self)
    # object.__abs__(self)
    # object.__invert__(self)

    def __str__(self):
        return "Dual({},  {})".format(self.real, self.dual)
    def __repr__(self):
        return self.__str__()

    # f(a + be) = f(a) + b fprime(a) e

    def exp(x):
        expa = np.exp(x.real)
        ret = Dual(expa, x.dual*expa)
        return ret
    
    def log(x):
        ret = Dual(np.log(x.real), x.dual / x.real)
        return ret
    
    def sin(x):
        ret = Dual(np.sin(x.real), x.dual * np.cos(x.real))
        return ret
    
    def cos(x):
        ret = Dual(np.cos(x.real), -x.dual * np.sin(x.real))
        return ret
    
    def sqrt(x):
        sqrta = np.sqrt(x.real)
        ret = Dual(sqrta, x.dual * 0.5/sqrta)
        return ret

def grad(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        newargs = [arg for arg in args]
        for k,arg in enumerate(args):
            newargs[k] = Dual(arg)
        args = newargs
        return func(*args, **kwargs).dual
    return wrapper

def f(x):
    return x**4
def f2(x):
    return x-np.exp(-2 * (np.sin(4*x)) **2)

#x - np.exp(-2 * (np.sin(4*x)) ** 2)
df = grad(f)
print(df(3))

df = grad(f2)
print(df(np.pi/16))#[1 + 8/e] 

