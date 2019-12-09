# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:53:09 2019

@author: DavidYQY
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../AnnoDomini')))

import AnnoDomini.AutoDiff as AD
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
from newtons_method import Newton

def single_var():
    f = lambda x: np.sin(x) + x * np.cos(x)
    x0 = -3
    
    demo = Newton(f,x0)
    ans = demo.find_root()
    
    # plot solution
    xs = np.linspace(-7,5,100)
    plt.plot(xs, f(xs), label="f")
    plt.scatter(ans, f(ans),label="Root", color = 'black')
    plt.scatter(x0, f(x0),label="initial", color = 'red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Visual of Newton's Method on $sin(x) + x * cos(x)$")
    plt.axhline(y = 0, color = 'red')
    plt.legend()
    plt.show()

def multi_var():
    def f(x,y):
        return x ** 2 + y ** 2 - 3 * x * y - 4 # (x-y)^2 = 9
    
    x0 = 1.0
    y0 = -2.0
    init_vars = [x0, y0]
    
    demo = Newton(f,init_vars)
    ans = demo.find_root()
    print(ans)
    delta = 0.025 
    lam1 = np.arange(-3, 3, delta) 
    lam2 = np.arange(-5, 3, delta) 
    Lam1, Lam2 = np.meshgrid(lam1, lam2) 
    value = Lam1 ** 2 + Lam2 ** 2 - 3 * Lam1 * Lam2 -4
    fig = plt.subplots(1,1, figsize = (10,7))
    CS = plt.contour(Lam1, Lam2, value,levels = 30) 

    plt.scatter(x0,y0,color = "red",label = "Initialization")
    plt.scatter(ans[0],ans[1],color = "green",label = "root found") 
    plt.clabel(CS, inline=1, fontsize=10) 
    plt.xlabel('x') 
    plt.ylabel('y') 
    #plt.axis('equal') 
    #plt.xlim(-3, 3)
    #plt.ylim(-5, 3)
    plt.legend() 
    plt.title('Level Curve of $x^2 + y^2 - 3*x*y - 4$ wrt x and y')
    plt.savefig('newton_multivar.png')


single_var()
multi_var()

