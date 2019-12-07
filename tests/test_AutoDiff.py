import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from AnnoDomini.AutoDiff import AutoDiff as AD
from AnnoDomini.AutoDiff import AutoDiff as AD

import numpy as np


def test_repr():
    x = AD(1.5)
    assert x.__repr__() == '====== Function Value(s) ======\n1.5\n===== Derivative Value(s) =====\n1.0\n'
    x = AD(2, [1,0])
    y = AD(3, [0,1])
    f = AD([x + y, x*y])
    assert f.__repr__() == '====== Function Value(s) ======\n[5 6]\n===== Derivative Value(s) =====\n[[1 1]\n [3 2]]\n'
    #assert x.__repr__() == 'Function Value: 1.5 | Derivative Value: 1.0'


def test_neq():
    x = AD(1.5,1)
    y = AD(1.5,2)
    assert x != y
    
def test_negative():
    x = AD(10)
    f = -x
    assert f == AD(-10, -1)

def test_add():
    x = AD(2, 1)
    f = x + 4
    assert f == AD(6, 1)
    y = AD(4, 2)
    assert x + y == AD(6, 3)
    f2 = x + 2 + x + 4
    assert f2 == AD(10, 2)


def test_radd():
    x = AD(2)
    f = 4 + x
    assert f == AD(6, 1)
    assert x == AD(2, 1)


def test_sub():
    x = AD(5)
    f = x - 6
    assert f == AD(-1, 1)
    y = AD(4, 2)
    assert x - y == AD(1, -1)
    f2 = x - x - x - 1
    assert f2 == AD(-6, -1)
    assert x == AD(5, 1)


def test_rsub():
    x = AD(5)
    f = 6 - x
    assert f == AD(1, -1)
    assert x == AD(5)


def test_mul():
    x = AD(5, 1)
    f1 = x * 3
    assert f1 == AD(15, 3)
    y = AD(10, 2)
    f2 = x * y
    assert f2 == AD(50, 20)  # 1 * 10 + 5 * 2 = 20
    f3 = x * x
    assert f3 == AD(25, 10)


def test_rmul():
    x = AD(5, 1)
    f = 2 * x
    assert f == AD(10, 2)
    assert x == AD(5, 1)


def test_truediv():
    x = AD(3)
    f = x / 2
    assert np.round(f.val, 2) == 1.5
    assert np.round(f.der, 2) == 0.5
    y = AD(3)
    f2 = x / y
    assert f2 == AD(1.0, 0.0)
    
    with np.testing.assert_raises(ZeroDivisionError):
        f3 = x/0
    
    with np.testing.assert_raises(ZeroDivisionError):
        z = AD(0, 3)
        f4 = x/z
    


def test_rtruediv():
    x = AD(2)
    f = 1 / x
    assert np.round(f.val, 2) == 0.5
    assert np.round(f.der, 2) == -0.25

    f3 = 2 / x / x
    assert f3 == AD(0.50, -0.50)

    with np.testing.assert_raises(ZeroDivisionError):
        z = AD(0, 3)
        f4 = 10/z
        
def test_pow():
    x = AD(2)
    f = x ** 3 + 3* x**2 + 3 * x + 1 # (x+1)**3
    assert f == AD(27,27)

def test_rpow():
    x = AD(3)
    f = 2 ** x# 2**x * log(2)
    assert f.val == 8
    assert np.round(f.der,2) == 5.55 # 2**3 * ln(2)
    
    f2 = 0 ** x
    assert f2.val == 0
    assert f2.der == 0
    x = AD(0)
    f3 = 0 ** x
    assert f3 == AD(1,0)
    
    with np.testing.assert_raises(ZeroDivisionError):
        x = AD(-1,1)
        f4 = 0 ** x
    
    with np.testing.assert_raises(ValueError):
        f3 = (-1) ** x

def test_sqrt():
    x = AD(4)
    f = np.sqrt(x)
    assert f == AD(2, 0.25)


def test_sin():
    x = AD(np.pi / 2)
    f = 2 * np.sin(x)
    assert np.round(f.val, 2) == 2.0
    assert np.round(f.der, 2) == 0.0


def test_cos():
    x = AD(np.pi / 2)
    f = 2 * np.cos(x)
    assert np.round(f.val, 2) == 0.0
    assert np.round(f.der, 2) == -2.0


def test_tan():
    x = AD(np.pi / 4)
    f = 3 * np.tan(x)
    assert np.round(f.val, 2) == 3.0
    assert np.round(f.der, 2) == 6.0
    with np.testing.assert_raises(ValueError):
        x = AD(np.pi / 2)
        f = np.tan(x)


def test_arcsin():
    with np.testing.assert_raises(ValueError):
        x = AD(-2)
        f = np.arcsin(x)
    with np.testing.assert_raises(ZeroDivisionError):
        x = AD(-1)
        f = np.arcsin(x)
    x = AD(0)
    f = np.arcsin(x)
    assert np.round(f.val, 2) == 0.00
    assert np.round(f.der, 2) == 1.00
    
def test_arccos():
    with np.testing.assert_raises(ValueError):
        x = AD(-2)
        f = np.arccos(x)
    with np.testing.assert_raises(ZeroDivisionError):
        x = AD(-1)
        f = np.arccos(x)
    x = AD(0)
    f = np.arccos(x)
    assert np.round(f.val, 2) == 1.57
    assert np.round(f.der, 2) == -1.00

def test_arctan():
    x = AD(1)
    f = np.arctan(x)
    assert np.round(f.val, 2) == 0.79
    assert np.round(f.der, 2) == 0.50

def test_sinh():
    x = AD(0.5)
    f = np.sinh(x)
    assert np.round(f.val, 2) == 0.52
    assert np.round(f.der, 2) == 1.13

def test_cosh():
    x = AD(0.5)
    f = np.cosh(x)
    assert np.round(f.val, 2) == 1.13
    assert np.round(f.der, 2) == 0.52

def test_log():
    x = AD(-1)
    with np.testing.assert_raises(ValueError):
        f = np.log(x)
    x = AD(1)
    f = np.log(x)
    assert f ==  AD(0,1)

def test_exp():
    x = AD(5)
    f = np.exp(x)
    assert np.round(f.val,2) == 148.41
    assert np.round(f.der,2) == 148.41

def test_logistic():
    a = 0.5
    loga = 0.5 + 0.5 * np.tanh(a/2)
    x = AD(a)
    f = x.logistic()
    assert np.round(f.val,2) == np.round(loga,2)
    assert np.round(f.der, 2) == np.round(np.exp(a) / (np.exp(a)+1)**2, 2)
    
def test_r2_to_r1():
    # f(x,y) = cos(x) + exp(y)
    x = AD(np.pi/2, [1,0])
    y = AD(1, [0,1])
    f = np.cos(x) + np.exp(y)
    assert np.round(f.val,2) == 2.72
    assert len(f.der) == 2
    assert np.round(f.der[0],2) == -1.0
    assert np.round(f.der[1],2) == 2.72

def test_r1_to_r2():
    # f(x) = (x^2, sin(x))
    x = AD(np.pi/2, 1)
    f = AD([x**2, np.sin(x)])
    assert len(f.val) == 2
    assert np.round(f.val[0], 2) == 2.47
    assert np.round(f.val[1], 2) == 1.00
    assert len(f.der) == 2
    assert np.round(f.der[0], 2) == 3.14
    assert np.round(f.der[1], 2) == 0

def test_r2_to_r2():
    # f(x,y) = (x + y, x * y)
    x = AD(2, [1,0])
    y = AD(3, [0,1])
    f = AD([x + y, x*y])
    assert len(f.val) == 2
    assert (f.val == np.array([5,6])).all()
    assert (f.der == np.array([[1,1],[3,2]])).all()
    

#test_rm_to_r1()
#test_r1_to_rn()
#test_r2_to_r2()
#test_repr()


