# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 09:28:54 2021

@author: mcamboim
"""
import matplotlib.pyplot as plt
import numpy as np

from rosenbrock_function import grad_f,f
from plot_3D import plot_contour

plt.close('all')
plt.rcParams.update({'font.size': 16})
plt.rcParams['axes.linewidth'] = 2

# Initialization =============================================================
x1_0,x2_0 = (-1.5,0.5)
max_error = 1e-2 # Mudar esse nome 
max_iter = 200000

x0 = np.array([x1_0,x2_0]).reshape(2,1)
nabla_f = grad_f(x0[0],x0[1])

xk = np.zeros((2,max_iter))
xk[:,0] = x0[:,0]
mod_grad = np.zeros(max_iter)
mod_grad[0] = np.sqrt(nabla_f.T @ nabla_f)

# Gradient Search ============================================================
k = 0
while np.sqrt(nabla_f.T @ nabla_f) >  max_error:
    # Optimum Step -----------------------------------------------------------
    alpha_0 = 0.0
    d0 = x0 - nabla_f * alpha_0
    nabla_d0 = grad_f(d0[0],d0[1])
    lambda_0 = nabla_d0.T @ -nabla_f

    alpha_1 = 1.0
    d1 = x0 - nabla_f * alpha_1
    nabla_d1 = grad_f(d1[0],d1[1])
    lambda_1 = nabla_d1.T @ -nabla_f

    alpha_opt = (alpha_1 - alpha_0) * lambda_1 / (lambda_0 - lambda_1) + alpha_1 
    
    # Att Result
    x0 = x0 - nabla_f * alpha_opt
    nabla_f = grad_f(x0[0],x0[1])
    
    # Saving Iteration
    k += 1
    xk[:,k] = x0[:,0]
    mod_grad[k] = np.sqrt(nabla_f.T @ nabla_f)
    if(k >= max_iter - 1):
        break
    
    
print("Otimização Finalizada")
print(f"Iterações Executadas: {k} Iterações")
print(f"Valor Analítico: X = ({np.round(1.0,6)},{np.round(1.0,6)})")
print(f"Valor Calculado: X = ({np.round(x0[0,0],6)},{np.round(x0[1,0],6)})")
print(f"Erro Obtido: X = ({np.round(x0[0,0] - 16/14,6)},{np.round(x0[1,0] - 5/7,6)})")

plt.figure(figsize=(12,6),dpi=150)
plt.plot(mod_grad,lw=3,c='blue')
plt.xlim(0,k)
plt.grid(True,ls='dotted')
plt.ylabel('|| \u2207f($x_1$,$x_2$) ||')
plt.xlabel('Iteração [N]')

plot_contour(xk,k)

