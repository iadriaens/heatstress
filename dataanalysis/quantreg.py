# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:25:06 2024

@author: u0084712
"""

#%% import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog



#%% define functions

# function to define the loss function of quantile regression in a linear programming way
# give more weight to first 15 days.
def qreg(order, X, y, n_diff, tau1, tau2, plotbool):
    """
	quantile regression with number of x values fixed = first ndiff days
	in a seperate tau1
    
    X = dim 
    y = dmy 
    order = order of polynomial function to use in the regression
    
    tau1 = first tau =  quantile for first n_diff measurements (e.g. 0.1)
    tau2 = second tau = quantile for len(X)-n measurements (e.g. 0.7)
    """
    
    # convert X to matrix
    x = pd.DataFrame(X,columns=["dim"])
    x["ones"] = np.ones(len(x))
    x=x[["ones","dim"]]
    if order > 1:
        for i in range(2,order+1):
            print(i)
            name = ["dim"+str(i)]
            x = pd.concat([x,pd.DataFrame(data=X.values**i,columns=name)],axis=1)

    # sizes to create matrices
    n = len(x)
    m = order+1

    # define function to optimize: equality constraints belonging to quantile regression (Aeq*x=beq)
    Aeq = np.block([np.eye(n),-np.eye(n),x])  # (n * 2n+m) linear equality constraints
    beq = y
    lb=np.hstack([np.zeros(n),np.zeros(n),-np.inf*np.ones(m)])   
    ub=np.inf*np.ones(m+2*n)
    bounds = np.vstack([lb,ub]).T
    
    # define function vector with quantiles as the objective function = normal qreg    
    f = np.hstack([tau2*np.ones(n),(1-tau2)*np.ones(n),np.zeros(m)])
    
    # adjusted qreg
    f2 = np.hstack([tau1*np.ones(n_diff),tau2*np.ones(n-n_diff),
                    (1-tau1)*np.ones(n_diff),(1-tau2)*np.ones(n-n_diff),np.zeros(m)]);

    # solve linear program -- normal qreg
    out = linprog(f, A_eq=Aeq, b_eq=beq,bounds=bounds)
    bhat = out.x[-m:]
    
    # solve linear program -- adjusted qreg
    out2 = linprog(f2, A_eq=Aeq, b_eq=beq,bounds=bounds)
    bhat2 = out2.x[-m:]
    
    if plotbool == True:
        # plot
        _,ax = plt.subplots(1,1, figsize = (12,6))
        plt.plot(X,y,"o-",color="darkblue", ms=4,label="data")
        plt.plot(X,np.dot(x,bhat),"red",lw=1.5, label="qreg " + "\u03C4" +"$_1$" +" = " + str(tau2) )
        plt.plot(X,np.dot(x,bhat2),"--",lw=1.5, color="magenta",label = "qreg " + "\u03C4" + "$_1$" +" = " + str(tau1) + ", " + "\u03C4" + "$_2$" +" = " + str(tau2) )
        plt.legend()
        ax.set_xlabel('dim')
        ax.set_ylabel('dmy')
          
    return np.dot(x,bhat),np.dot(x,bhat2)
