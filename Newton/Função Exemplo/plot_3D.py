# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 18:26:30 2021

@author: mcamboim
"""

import matplotlib.pyplot as plt
import numpy as np

from example_function import grad_f,f

def plot_contour(xk,k):
    x_min = -10.0
    x_max = 10.0
    x1 = np.arange(x_min,x_max,0.1)
    x2 = np.arange(x_min,x_max,0.1)
    x1,x2 = np.meshgrid(x1,x2)
    z = f(x1,x2)

    levels = np.arange(np.min(z), np.max(z), 15.0)


    fig = plt.figure(figsize=(12,6),dpi=150)
    ax = plt.axes()
    #ax.contour(x1, x2, z,cmap='viridis')
    ax.set_xlim([x_min,x_max])
    ax.set_ylim([x_min,x_max])
    ax.contour(x1, x2, z, levels = levels,cmap='viridis')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.plot(xk[0,0:k+1],xk[1,0:k+1],c='k',lw=2)
