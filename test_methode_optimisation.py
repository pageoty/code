# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:21:37 2020

@author: Yann Pageot
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import csv
from scipy.optimize import minimize
from ResultsUtils import *
from sklearn.metrics import *
from scipy.optimize import linprog
from scipy import optimize



def rosen(x):
     """The Rosenbrock function
     f(x,y)=(1-x)²+100(y-x²)²
     """
     return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
 
 
# def nash (s):
#     """The Nash function"""
#    # 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)
#     nash=1 - sum((s-x)**2)/sum((x-np.mean(x))**2)
#     return nash 
    
def RMSE(x,**args) :
    x_data = args[0]
    y_data = args[1]
    rmse= mean_squared_error(x_data,y_data,squared=False)
    return rmse
    
    

if __name__ == "__main__":
    
    
    x0 = [1,10]# valeur des paramétres qui vont varier/ que l'on souhaite optimiser 
    # résultat de l'optimisation des paramétes qui vont étre intérfegrd dzns SAMIR via csv
    # Lancement de samir 
    # résultats SAMIR 
    res = minimize(rosen,x0, method='Nelder-Mead')# " module d'optimisation "
    res.x
    # Calcule d'un RMSE / nash => conservation ou poubelle 
    

    
#     x=np.array([5.0, 0.1, 2.0, 1.6, 1.9])
#     x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])# état initial
#     # resrmse = minimize(RMSE,x0, method='nelder-mead',options={'xatol': 1e-8, 'disp': False})
#     # resnash = minimize(nash,x0, method='nelder-mead',options={'xatol': 1e-8, 'disp': True})
#     # print(resrmse)
#     # print(resnash)
# # options={'xatol': 1e-8, 'disp': True}

#     # fun = lambda x,s: mean_squared_error(x,s,squared=False)
#     fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
#     cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
#             {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
#             {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})

#     bnds = ((0, None), (0, None))

#     res = minimize(fun, (x0), method='SLSQP',
#                constraints=cons)
    
# =============================================================================
# Test pour comprendre
# =============================================================================
import random
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt
 
 
def linear_law(x, slope, scale):
    return slope*x + scale
 
def evaluate(x, *args):
    """Function computing square error"""
    x_data = args[0]
    y_data = args[1]
    # y_estimated = linear_law(x_data, x[0], x[1])# dans notre cas sortie du modèle
    return mean_squared_error(y_data, y_data,squared=False)
 
# Generate pseudo random data
slope, scale = 2.0, 10.0
x_data = np.arange(1, 100, 1)
y_data = np.array([slope*i+scale+random.randrange(-10,10) for i in x_data])
 
# x0 = [slope, scale]
x0 = [10.0, 1]
slope_estv= minimize(evaluate, x0, args=(x_data, y_data),method='nelder-mead')
slope_est, scale_est = fmin(evaluate, x0, args=(x_data, y_data), xtol=1e-8, disp=True,)

# print (slope_est, scale_est)
# y_est = linear_law(x_data, slope_est, scale_est)
 
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(x_data, y_data, color='red', label='Raw data')
# ax.plot(x_data, y_est, color='blue', label='Fitted law')
# ax.legend(loc='lower right')
# plt.grid()
# plt.show()


