# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:53:09 2019

@author: DavidYQY
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Optimization')))

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


# def newtons_method(f, x0, iters = 100, tol = 1e-6, alpha = 1):
#     """Use Newton's method to approximate a root.
#
#     Inputs:
#         f (function): A function to handle.
#         x0 (float): Initial guess.
#         iters (int): Maximum number of iterations before the function
#             returns. Defaults to 100.
#         tol (float): The function returns when the difference between
#             successive approximations is less than tol.
#         alpha (float): Defaults to 1.  Allows backstepping.
#
#     Returns:
#         A float that is the root that Newton's method finds
#     """
#     # Newton's Method on Scalar Input
#     xold = x0
#     for i in range(iters):
#         # compute derivative via AutoDiff
#         temp = AD.AutoDiff(xold)
#         df = f(temp)
#
#         #solve for x_k1
#         xnew = xold - alpha * f(xold)/df.der
#         if la.norm(xnew - xold) < tol:
#             return xnew
#         else:
#             xold = xnew
#
#     return xnew
#
# ans = newtons_method(f,x0)
#
# # plot solution
# xs = np.linspace(-7,5,100)
# plt.plot(xs, f(xs), label="f")
# plt.scatter(ans, f(ans),label="Root", color = 'black')
# plt.scatter(x0, f(x0),label="initial", color = 'red')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Visual of Newton's Method on $sin(x) + x * cos(x)$")
# plt.axhline(y = 0, color = 'red')
# plt.legend()
# plt.show()
