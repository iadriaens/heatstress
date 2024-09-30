# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:07:07 2023

@author: u0084712

-------------------------------------------------------------------------------
Iterative Wood model for daily milk yield
- function 1: wood model
- function 2: iterative procedure to fit model
- function 3: visualisation


"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels import robust
from statsmodels.robust.scale import huber
from scipy.optimize import linprog


#%% function 1: Wood

def wood(dim, a, b, c):
    """
    Defines the (non-linear) Wood model as y = p1 * np.power(x,p2) * np.exp(-p3*x)

    Within the F2_SelectionLactationHealthy script, this function is used in the F2_5_wm_2step_iter as a callable in the scipy.optimize.curve_fit function.

    Parameters
    ----------
    x : Array (dtype = float)
        Contains the DIM values of each milking session within one (quarter or udder level) lactation
    a : float
        First parameter: represents the milk yield right after calving (the intercept).
        Boundaries: [0 60 or 30]
        Usual offset values: depend on quarter/udder level. Often taking the mean value of the y variable
    b : float
        Second parameter: represents the decreasing part of the lactation curve (after the production peak).
        Boundaries:  [0 0.6]
        Usual offset values: 0.2
    c : float
        Third parameter: represents the increasing part of the lactation curve (before the production peak).
        Boundaries:  [0 0.01]
        Usual offset values: 0.005

    Returns
    -------
    y : Array (dtype = float)
        Predicted standardised daily milk yields of each milking session within one (quarter or udder level) lactation.

    """
    
    y = a * np.power(dim, b) * np.exp(-c*dim)
    
    return y

def woodres(dim,a,b,c,dmy):
    
    y = wood(dim,a,b,c)
    res = dmy-y
    
    return res


"""
for development:
    dim = df["dim"]
    dmy = df["dmy"]
    a0 = woodsettings["init"][0]
    b0 = woodsettings["init"][1]
    c0 = woodsettings["init"][2]
    lb = woodsettings["lb"]
    ub = woodsettings["ub"]
"""


def itw(dim,dmy, a0, b0, c0, lb, ub, plotbool):
    
    df2 = pd.concat([dim,dmy], axis = 1)
    
    # find initial fit of the wood model
    p = curve_fit(wood, dim, dmy,
                  p0 = [a0,b0,c0],
                  bounds=(lb,ub),
                  method='trf')
    a = p[0][0]
    b = p[0][1]
    c = p[0][2]
    
    # calculate residuals and find (robust) std
    res = woodres(dim,a,b,c,dmy)
    try:
        sd = huber(res)[1]      # robust sd
    except:
        sd = res.std()
    mod = wood(dim,a,b,c)   # fitted wood model
    #t = mod - 1.6 * sd      # threshold
    t = mod - 0.6 * sd      # threshold
    
    # find all residuals below threshold of 1.6*sd
    idx_excl = dmy.loc[(dim > 7) & (dmy < t)].index.values
    
    # all residuals included (above the threshold)
    idx = dmy.loc[(dim <= 7) | (dmy >= t)].index.values

    #--------------------------------------------------------------------------
    if plotbool == True:
        # plots
        fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize=(20,9))
        sns.lineplot(data = df2,x = "dim",y = "dmy",
                color ="blue",marker = "o", ms = 4, lw = 1.2 )
        ax.plot(dim,wood(dim, a0, b0, c0),
                color = "grey", linestyle = "--",lw = 0.8)
        ax.plot(dim,mod,
                color = "firebrick", linestyle = "--",lw = 1.8)
        ax.plot(dim,t,
                color = "red", linestyle = ":",lw = 0.4)
        ax.plot(dim[idx_excl],dmy[idx_excl],
                color = "red", marker = "x",lw = 0, ls = "",ms = 3)
        
        ax.legend(labels = ["dmy","_","init","wood1","threshold1","_" ])
    #--------------------------------------------------------------------------
    
    # prepare iterative procedure
    rmse0 = np.sqrt((res*res).mean()) # difference with prev needs to be > 0.10
    rmse1 = rmse0 + 1                 # fictive value to start iterations
    no_iter = 0                       # needs to be smaller than 20
    lendata = len(idx)                # needs to be larger than 50
    
    while (lendata > 50) and (no_iter < 20) and (rmse1-rmse0 > 0.1):
        
        # add to no_iter
        no_iter = no_iter + 1
        print("iter = " + str(no_iter))
        
        # fit model on remaining data, with initial values = a,b,c
        p = curve_fit(wood, dim[idx], dmy[idx],
                      p0 = [a,b,c],
                      bounds=(lb,ub),
                      method='trf')
        a = p[0][0]
        b = p[0][1]
        c = p[0][2]
        
        # calculate residuals and find (robust) std
        res = woodres(dim,a,b,c,dmy)
        try:
            sd = huber(res)[1]      # robust sd
        except:
            sd = res.std()
        mod = wood(dim,a,b,c)           # wood model fitted
        t = mod - 1.6 * sd              # threshold
        
        # find all residuals below threshold of 1.6*sd
        idx_excl = dmy.loc[(dim > 7) & (dmy < t)].index.values
        
        # all residuals included (above the threshold)
        idx = dmy.loc[(dim <= 7) | (dmy >= t)].index.values
        
        #----------------------------------------------------------------------
        if plotbool == True:
            
            ax.plot(dim,mod,
                    color = "magenta", linestyle = "--",lw = 1)
            ax.plot(dim,t,
                    color = "magenta", linestyle = ":",lw = 0.4)
            ax.plot(dim[idx_excl],dmy[idx_excl],
                    color = "red", marker = "x",lw = 0, ls = "",ms = 3)
            if no_iter == 1:
                ax.legend(labels = ["dmy","_","init","wood1","threshold1","_","itw" ])
        #----------------------------------------------------------------------
        
        # update criteria
        rmse1 = rmse0
        rmse0 = np.sqrt((res[idx]*res[idx]).mean())      
        
        lendata = len(idx)
    
    print("no_iter = " + str(no_iter) + ", no_excluded = " + str(len(idx_excl)))
    if plotbool == True:
        
        ax.plot(dim,mod,
                color = "orangered", linestyle = "-",lw = 2)
        ax.set_ylabel("dmy [kg]")
        ax.set_xlabel("dim [d]")
        ax.set_title("no_iter = " + str(no_iter) + ", no_excluded = " + str(len(idx_excl)))
    else:
        ax = None
        
    return ax, p, mod


#%% perturbations

"""
for development:
    dim = df["dim"]
    dmy = df["dmy"]
    itw = df["mod"]

"""


def pert(dim, dmy, itw):
    """
    definitions perturbations: 
        - If less than 5 days below ITW			
  							no perturbation
        - If >= 5 and less than 10	days below ITW	
                never < 0.85*ITW		                very mild perturbation
                1 or 2 days < 0.85*ITW				    mild perturbation
                3 or more days < 0.85*ITW				moderate perturbation
        - If more than 10 days below ITW
                0, 1 or 2 days < 0.85*ITW			    mild perturbation
                3 or more days, 
                    never >3 successive days	        moderate perturbation
                3 or more days, 
                    at least once >3 successive days    severe perturbation
    """
    # create data frame and calculate model residuals
    df = pd.concat([dim,dmy,itw],axis = 1)
    df["res"] = df["dmy"] - df["mod"]
    df["thres"] = df["mod"] * 0.85
    
    # # find std robust of residual time series
    # try:
    #     sd = huber(df["res"])[1]      # robust sd
    # except:
    #     sd = df["res"].std()
        
    # find negative
    df["is_neg"] = 0
    df.loc[df["dmy"] < df["mod"] , "is_neg"] = 1
    
    # find below 1.6*robust_std
    df["is_low"] = 0
    df.loc[df["dmy"] < df["thres"], "is_low"] = 1
    
    # number of consecutive negatives > 5 days   
    df["test"] = df['is_neg'].ne(df['is_neg'].shift()).cumsum()
    starts = df["test"].drop_duplicates()
    df.loc[starts.index.values,"test2"] = 1
    df.loc[(df["is_neg"] == 0) | (df["test2"].isna()),"test2"] = 0
    df.loc[:,"test2"] = df.loc[:,"test2"].cumsum()
    df.loc[(df["is_neg"] == 0),"test2"] = 0
    dur = df[["dim","test2"]].groupby(by = "test2").count().reset_index()
    dur = dur.rename(columns = {"dim" : "pert_len"})
    dur = dur.loc[dur["pert_len"] >=5,:]
    df = pd.merge(df,dur,how = "outer",on = "test2").sort_values(by = "dim").reset_index(drop=1)
    df.loc[(df["is_neg"] == 0)| (df["pert_len"].isna()),"pert_len"] = 0
    df = df.drop(columns = ["test","test2"])

    # correct perturbation/period number
    df.loc[df["pert_len"]==0,"is_neg"] = 0
    df["pert_no"] = df['is_neg'].ne(df['is_neg'].shift()).cumsum()
    
    # divide into mild, moderate or severe perturbation
    df["is_vmild"] = np.nan
    df["is_mild"] = np.nan
    df["is_mod"] = np.nan
    df["is_sev"] = np.nan
    for no in df["pert_no"].drop_duplicates().reset_index(drop=1):
        # print("period = " + str(no))
        
        # if perturbation, quantify mild, moderate, severe
        period = df.loc[df["pert_no"]==no,:]
        
        # length of successive ones
        lengths_low = period["is_low"][period["is_low"]==1].groupby(period["is_low"].eq(0).cumsum()).sum()
        # set criterion for first period lactation: check slope instead of neg?
        if (period.iloc[0].loc["is_neg"] == 1) & (period["dim"].min() < 7):
            slope1 = period["mod"].diff()
            slope2 = period["dmy"].diff()
            slopediff = (slope2-slope1).mean()
            if len(slope2.loc[slope2 < 0]) > 7: 
                df.loc[period.index,"is_sev"] = 1
            elif len(slope2.loc[slope2 < 0]) > 5: 
                df.loc[period.index,"is_mod"] = 1
            elif slopediff < 0:
                df.loc[period.index,"is_mild"] = 1
            else:
                df.loc[period.index,"is_vmild"] = 1
        elif (period.iloc[0].loc["is_neg"]) == 1:  # if a perturbation and not dim<5
            # perturbation length between 5 and 10 days
            if (period["pert_len"].mean() < 10):
                if (period["is_low"].sum() == 0):
                    df.loc[period.index,"is_vmild"] = 1
                elif (lengths_low.sum() < 3):
                    df.loc[period.index,"is_mild"] = 1
                else:
                    df.loc[period.index,"is_mod"] = 1
            # perturbation length 10 days or longer
            else: 
                if (period["is_low"].sum() < 3):
                    df.loc[period.index,"is_mod"] = 1
                elif (lengths_low.max() < 3):
                    df.loc[period.index,"is_mod"] = 1
                else:
                    df.loc[period.index,"is_sev"] = 1
        
        
            
        
    return df[["dim","thres","pert_len","pert_no","is_vmild","is_mild","is_mod","is_sev"]]
        
        

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

    resources:
    https://github.com/iadriaens/quantileRegression/blob/master/quantreg.m
    https://github.com/antononcube/MathematicaForPrediction/blob/master/Documentation/Quantile%20regression%20through%20linear%20programming.pdf
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
    https://nl.mathworks.com/help/optim/ug/linprog.html
    previous implementation in matlab:
    
    """
    
    # convert X to matrix
    x = pd.DataFrame(X,columns=["dim"])
    x["ones"] = np.ones(len(x))
    x=x[["ones","dim"]]
    if order > 1:
        for i in range(2,order+1):
            # print(i)
            name = ["dim"+str(i)]
            x = pd.concat([x,pd.DataFrame(data=X.values**i,columns=name, index=X.index)],axis=1)

    # sizes to create matrices
    n = len(x)
    m = order+1

    # define function to optimize: equality constraints belonging to quantile regression (Aeq*x=beq)
    Aeq = np.block([np.eye(n),-np.eye(n),x])  # (n * 2n+m) linear equality constraints
    beq = y
    lb=np.hstack([np.zeros(n),np.zeros(n),-np.inf*np.ones(m)])   
    ub=np.inf*np.ones(m+2*n)
    bounds = np.vstack([lb,ub]).T
    
    # # define function vector with quantiles as the objective function = normal qreg    
    # f = np.hstack([tau2*np.ones(n),(1-tau2)*np.ones(n),np.zeros(m)])
    dimdiff = n_diff
    n_diff = sum(X<n_diff)
    
    # adjusted qreg objective function
    f = np.hstack([tau1*np.ones(n_diff),tau2*np.ones(n-n_diff),
                    (1-tau1)*np.ones(n_diff),(1-tau2)*np.ones(n-n_diff),np.zeros(m)]);

    # solve linear program -- normal qreg
    out = linprog(f, A_eq=Aeq, b_eq=beq,bounds=bounds)
    bhat = out.x[-m:]
    
    # # solve linear program -- adjusted qreg
    # out2 = linprog(f2, A_eq=Aeq, b_eq=beq,bounds=bounds)
    # bhat2 = out2.x[-m:]
    
    if plotbool == True:
        # plot
        _,ax = plt.subplots(1,1, figsize = (12,6))
        plt.plot(X,y,"o-",color="darkblue", ms=4,label="data")
        # plt.plot(X,np.dot(x,bhat),"red",lw=1.5, label="qreg " + "\u03C4" +"$_1$" +" = " + str(tau2) )
        plt.plot(X,np.dot(x,bhat),"--",lw=2.5, color="red",label = "qreg " + "\u03C4" + "$_1$" +" = " + str(tau1) + ", " + "\u03C4" + "$_2$" +" = " + str(tau2) )
        plt.legend()
        ax.set_xlabel('dim')
        ax.set_ylabel('dmy')
        ax.set_title("quantile regression model, dim < " + str(dimdiff) + " have different \u03C4, polynomial order = " + str(order))
    else:
        ax = 0
    
    # create dataframe outputs with indices of orginal X 
    res = pd.DataFrame(data=np.dot(x,bhat),columns=["mod"],index=X.index)
          
    return ax,res,bhat