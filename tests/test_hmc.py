# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:37:00 2019

@author: for_y
"""

import numpy as np
from AnnoDomini.hamilton_mc import HMC, describe
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_hmc():
    def norm_function(x):
        var = 1
        denom = (2*np.pi*var)**.5
        num = np.exp(-x**2/2)
        return num/denom
    
    start_point = 0
    chain,accepts_ratio = HMC(target_pdf = norm_function, burn_in=0, thinning=1,chain_len=100, q_init=[start_point],epsilon = 0.05)
    assert len(chain) == 100
    q = chain[:,0]
    assert max(q) < 10
    assert np.isnan(q).sum() == 0
    
    
    def neg_log_weibull(lam = 1, k = 0.5):
        def w(x):
            if x > 0:
                return -(np.log(k / lam) + (k-1) * np.log(x/lam) - (x/lam) ** k)
            else:
                return float('inf')
        return w
    
    start_point = 3
    func = neg_log_weibull(k = 1.5)
    chain,accepts_ratio = HMC(U = func, burn_in=0, thinning=1,chain_len=100, q_init=[start_point],epsilon = 0.02)
    assert len(chain) == 100
    q = chain[:,0]
    assert max(q) < 10
    assert min(q) > 0
    assert np.isnan(q).sum() == 0
    
    
