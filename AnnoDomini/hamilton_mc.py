# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 16:05:51 2019

@author: for_y
"""

import AnnoDomini.AutoDiff as AD
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def HMC(q_init, target_pdf = None, D = None, U = None, 
          chain_len = 1000, T = 5, burn_in = 0, thinning = 1, epsilon = 0.1, randomseed = 2019):
    
    """Use hamiltonian monte carlo to sample from a certain distribution.

    Inputs:
        q_init (list): A list representing Initial point
        target_pdf (target probability density function): A pdf to sample from
        D(float): dimension of the q_init
        U(-log probility pdf)： if target_pdf is provided, it could be automatically inferred; otherwise can be specified directly
        chain_len(float): length of hamiltonian monte carlo chain
        T(float): length of leapfrog in HMC
        burn_in, thinning(float): burn in and thinning to the chain
        epsilon (float): step length in the HMC

    Returns:
        A chain of samples which sample from the target probability density function
    """
    
    np.random.seed(randomseed)
    if D is None:
        if isinstance(q_init, float):
            D = 1
        else:
            D = len(q_init)
            q_init = np.array(q_init)
        
    if target_pdf:
        U = lambda q: -np.log(target_pdf(q))
    
    def K(p):
        p = p.reshape(-1,1)
        return 0.5 * p.T.dot(p)
    
    H = lambda q,p: U(q) + K(p)
        
    accepts = 0
    q_current = q_init
    
    output = []
    for iter in tqdm(range(chain_len)):

        # kick off
        p_current = np.random.multivariate_normal(np.zeros(D), np.identity(D), 1)[0]
        p_current = p_current.reshape(q_current.shape)
        
        # simulate movement
        q_prev, p_prev = q_current, p_current
        
        for t in range(T):
            dU = U(AD.AutoDiff(q_prev)) # use our AD class
            
            if isinstance(dU, AD.AutoDiff):
                p_half = p_prev - epsilon/2 * dU.der
            else:
                p_half = p_prev - epsilon/2 * dU
                
            q_propose = q_prev + epsilon * p_prev
            dU = U(AD.AutoDiff(q_propose))
            if isinstance(dU, AD.AutoDiff):
                p_propose = p_half - epsilon/2 * dU.der
            else:
                p_propose = p_half - epsilon/2 * dU
            
            
            if np.sum(np.isnan(*q_propose))>0 or np.sum(np.isnan(p_propose))>0:
                #print(q_propose,p_propose)
                print("overflow error!")
                return -1,-1
            
            q_prev, p_prev = q_propose, p_propose
        
        # reverse momentum
        p_propose = -p_propose
        
        # correction for simulation error
        alpha = min(1, np.exp(H(q_current,p_current) - H(q_propose, p_propose)))
        if np.random.uniform() <= alpha:
            output.append(q_propose)
            q_current, p_current = q_propose, p_propose
            accepts += 1
        else:
            output.append(q_current)
    temp = np.array(output)
    return temp[burn_in::thinning], accepts * 1. / chain_len

if __name__ == '__main__':
    def norm_function(x):
        var = 1
        denom = (2*np.pi*var)**.5
        num = np.exp(-x**2/2)
        return num/denom
    
    start_point = -10.0 # start from far apart
    chain,accepts_ratio = HMC(target_pdf = norm_function, burn_in=200, thinning=2,chain_len=10000, q_init=[start_point],epsilon = 0.05)
    print("Accepts ratio = {}".format(accepts_ratio))
    print(chain.shape)
    
    q = chain[:,0]
    fig,ax = plt.subplots(1,1,figsize = (8,5))
    x = np.linspace(-4,4)
    ax.plot(x,norm_function(x),color = "black",label = "actual pdf")
    ax.hist(q,bins = 50, density = True, color = "blue",alpha = 0.3, label = "histogram of samples")
    ax.set_title("Actual pdf vs sampling by hamiltonian monte carlo")
    ax.legend()
    #plt.savefig('hmc_simulation.png')