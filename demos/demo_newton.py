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
