# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:20:21 2024

@author: u0084712
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from scipy import interpolate
from sklearn.linear_model import QuantileRegressor

############
### DATA ###
############
x1 = np.array([0, 4, 6, 10, 15, 20])
y1 =  np.array([18, 17.5, 13, 12, 8, 10]);

scale = 2
x2 = np.array([0,10.5,28])
y2= scale*np.array([18.2,10.6,10.3]);

# y2 is the target function and y1 is the function to be shifted
f = y1;
InterPolationFunction = interpolate.interp1d(x2, y2)
g = InterPolationFunction(x1)
n = len(g)

#######################################
### Solution 1: Quantile regression ###
#######################################
QuantileReg = QuantileRegressor(quantile=0.5, fit_intercept=False).fit(f.reshape(-1, 1), g)
aSol1 = QuantileReg.coef_[0]
print(aSol1)

######################################
### Solution 2: Linear programming ###
######################################
# Let the vector x stack: a, u_1, ..., u_n
# fVector'*x should equal sum_i u_i
fVector = np.ones((n+1,1))
fVector[0] = 0

# Stack of linear constraints in matrix notation
A = np.block([[-f.T, -np.eye(n)], [f.T, -np.eye(n)]])
b = np.vstack( (-g.reshape(n,1),g.reshape(n,1)) )

# Solver
LinprogOutput = linprog(fVector, A_ub=A, b_ub=b)
aSol2 = LinprogOutput.x[0]
print(aSol2)

############
### PLOT ###
############
plt.figure()
plt.plot(x1, f, label = "original f")
plt.plot(x1, g, label = "g (interpolation)")
plt.plot(x1, aSol1*f, label = "a*f")
plt.plot(x2,y2,label="shifted")
plt.legend()
plt.show()