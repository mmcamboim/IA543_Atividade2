# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 10:00:25 2021

@author: mcamboim
"""
import numpy as np

def f(x1,x2):
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

def grad_f(x1,x2):
    nabla_f = np.zeros(2).reshape(2,1)
    nabla_f[0] = -400*x1*(x2 - x1**2) - 2*(1-x1)
    nabla_f[1] = 200*(x2-x1**2)
    return nabla_f

def hessi_f(x1,x2):
    F = np.zeros(4).reshape(2,2)
    F[0,0] = 1200*x1**2 - 400*x2 + 2
    F[0,1] = -400*x1
    F[1,0] = -400*x1
    F[1,1] = 200
    return F
