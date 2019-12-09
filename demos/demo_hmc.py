# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 20:37:19 2019

@author: for_y
"""

import matplotlib.pyplot as plt
import numpy as np
from AnnoDomini.hamilton_mc import HMC

def demo_normal():
    def norm_function(mu = 0, var = 1):
        def norm(x):
            denom = (2*np.pi*var)**.5
            num = np.exp(-(x-mu)**2/(2*var))
            return num/denom
        return norm
    
    start_point = -10.0 # start from far apart
    func = norm_function(1,1)
    
    chain,accepts_ratio = HMC(target_pdf = func, burn_in=200, thinning=2,chain_len=10000, q_init=[start_point],epsilon = 0.05)
    print("Accepts ratio = {}".format(accepts_ratio))
    print(chain.shape)
    
    q = chain[:,0]
    fig,ax = plt.subplots(1,1,figsize = (8,5))
    x = np.linspace(-4,4)
    ax.plot(x,func(x),color = "black",label = "actual pdf")
    ax.hist(q,bins = 50, density = True, color = "blue",alpha = 0.3, label = "histogram of samples")
    ax.set_title("Actual pdf vs sampling by hamiltonian monte carlo")
    ax.legend()
    plt.savefig('hmc_simulation_normal.png')

def demo_weibull():
    
    def neg_log_weibull(lam = 1, k = 0.5):
        def w(x):
            if x > 0:
                return -(np.log(k / lam) + (k-1) * np.log(x/lam) - (x/lam) ** k)
            else:
                return float('inf')
        return w
    
#    def weibull(lam = 1, k = 0.5):
#        def w(x):
#            if x > 0:
#                return (k / lam) * (x/lam) ** (k-1) * np.exp(-(x/lam) ** k)
#            else:
#                return 0
#        return w
    
    start_point = 1.0 # start from far apart
    func = neg_log_weibull()
    chain,accepts_ratio = HMC(U = func, burn_in=200, thinning=1,chain_len=10000, q_init=[start_point],epsilon = 0.03)
    print("Accepts ratio = {}".format(accepts_ratio))
    print(chain.shape)
    
    q = chain[:,0]
    fig,ax = plt.subplots(1,1,figsize = (8,5))
    x = np.linspace(0.01,6)
    y = np.array(list(map(func,x)))
    ax.plot(x,np.exp(-y),color = "black",label = "actual pdf")
    ax.hist(q,bins = 50, density = True, color = "blue",alpha = 0.3, label = "histogram of samples")
    ax.set_title("Actual pdf vs sampling by hamiltonian monte carlo")
    ax.set_ylim(0,2.5)
    ax.legend()
    plt.savefig('hmc_simulation_weibull_1_05.png')
    
    
    start_point = 1.0 # start from far apart
    func = neg_log_weibull(k = 1.5)
    chain,accepts_ratio = HMC(U = func, burn_in=0, thinning=1,chain_len=10000, q_init=[start_point],epsilon = 0.02)
    print("Accepts ratio = {}".format(accepts_ratio))
    print(chain.shape)
    
    q = chain[:,0]
    fig,ax = plt.subplots(1,1,figsize = (8,5))
    x = np.linspace(0.01,6)
    y = np.array(list(map(func,x)))
    ax.plot(x,np.exp(-y),color = "black",label = "actual pdf")
    ax.hist(q,bins = 50, density = True, color = "blue",alpha = 0.3, label = "histogram of samples")
    ax.set_title("Actual pdf vs sampling by hamiltonian monte carlo")
    ax.legend()
    plt.savefig('hmc_simulation_weibull_1_15.png')

demo_normal()
demo_weibull()


