# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:04:12 2024

@author: u0084712
"""

import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataanalysis")
os.chdir(path)

#%% import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from dmy_functions import wood, itw, pert
from scipy.optimize import curve_fit
from random import sample
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import openpyxl
from scipy.signal import savgol_filter


# from datetime import date

#%matplotlib qt

#%% set file paths and constants

path_data = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","datapreprocessing",
                    "results","data")
dpath = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataanalysis",
                    "data")


# farm selected
farms = [30,31,33,34,35,38,39,40,43,44,45,46,48]
#farms = [44,45,46,48]

#%% load and process data

for farm in farms:
    milk = pd.read_csv(os.path.join(path,"results","data","milk_itw_farm_" + str(farm)+".txt"),
                       index_col = 0)
    milk["relres"] = milk["res"] / milk["mod"]*100
    milk.loc[milk["relres"].abs() > 100]=0
    milk.loc[milk["parity"]>3,"parity"] = 3
    milk["date"] = pd.to_datetime(milk["date"],format='%Y-%m-%d')
    milk = milk.sort_values(by=["animal_id","date"]).reset_index(drop=1)
    milk = milk.loc[milk["farm_id"]==farm,:].reset_index(drop=1)  
    milk = milk.dropna().reset_index(drop=1)

    cowlac = milk[["animal_id","parity"]].drop_duplicates().reset_index(drop=1)
    milk["sm1"] = 0
    milk["sm2"] = 0

    for i in range(0,len(cowlac)):
        df = milk.loc[(milk["animal_id"] == cowlac["animal_id"][i]) & \
                      (milk["parity"] == cowlac["parity"][i]),:]
        vals = df[["relres"]].dropna().apply(lambda x: savgol_filter(x,3,1)).values
        milk.loc[df.index.values,"sm1"] = vals    
        vals = df[["relres"]].dropna().apply(lambda x: savgol_filter(x,7,2)).values
        milk.loc[df.index.values,"sm2"] = vals

    
    

    #load thi
    thi = pd.read_csv(os.path.join(dpath,"weatherfeatures_" + str(farm) + ".txt"),
                     index_col = 0 )
    thi["date"] = pd.to_datetime(thi["date"],format='%Y-%m-%d')
    data = pd.merge(milk,thi, on = ["farm_id","date"],how="outer")
    data = data.loc[data["animal_id"].isna()==False,:].reset_index(drop=1)
    data["THI"] = data["thi_avg"]
    data.loc[data["thi_avg"]<68,"THI"] = np.nan

    # summary = milk.groupby(by="date").agg({"relres":["mean","std"]})
    _, ax = plt.subplots(nrows = 2, ncols = 1, figsize=(16,7),sharex=True)
    sns.lineplot(data=milk,x="date",y="sm1",hue="parity",estimator="median",errorbar="sd",ax=ax[0])
    sns.lineplot(data=milk,x="date",y="sm2",hue="parity",estimator="median",errorbar="sd",ax=ax[1])
    ax2 = ax[0].twinx()
    ax2.plot(data["date"].drop_duplicates(), data.drop_duplicates(subset="date").loc[:,"THI"],color="b",marker="o",ms=2,lw=1.2)
    ax2.set_ylim(67,82)
    ax[0].plot([data["date"].min(),data["date"].max()],[0, 0],'--',color='r')
    ax[0].set_xlim(data["date"].min(),data["date"].max())
    
# define wood settings
woodsettings = {"init" : [35,0.25,0.003],   # initial values
                "lb" : [0,0,0],             # lowerbound
                "ub" : [100,5,1],           # upperbound
                }

for farm in farms:
    
    # read milk yield data
    milk = pd.read_csv(os.path.join(path_data, "milk_preprocessed_" 
                                    + str(farm) + ".txt"),
                       index_col = 0)
    milk["date"] = pd.to_datetime(milk["date"],format='%Y-%m-%d')
    
    # correct dim = -1
    milk = milk.loc[milk["dim"] >=0].reset_index(drop=1)

    # unique lactations
    cowlac = milk[["animal_id","parity"]].drop_duplicates().reset_index(drop=1)
    rsample_plot = sample(cowlac.index.values.tolist(),10)

    # visualise + model per lactation
    for i in range(0,len(cowlac)):
        # select data
        df = milk.loc[(milk["animal_id"] == cowlac["animal_id"][i]) & \
                      (milk["parity"] == cowlac["parity"][i]),:]
        
        # adjust woodsettings and fit wood model
        woodsettings["init"][0] = df.loc[(df["dim"] > 30) & (df["dim"] < 100),"dmy"].mean()/2        
        woodsettings["ub"][0] = df["dmy"].max()
        p = curve_fit(wood, df["dim"], df["dmy"],
                      p0 = woodsettings["init"],
                      bounds=(woodsettings["lb"],woodsettings["ub"]),
                      method='trf')
        wa = p[0][0]
        wb = p[0][1]
        wc = p[0][2]
        
        if i in rsample_plot:
            print(i)
            plotbool = True
        else:
            plotbool = False
        
        #----------------------------------------------------------------------

        if plotbool == True:
            fig, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize=(20,9))
            ax2 = ax1.twiny()
            sns.lineplot(data=df,x="date",y="dmy",ax = ax2,
                     color ="white",lw = 0)
            sns.lineplot(data=df,x="dim",y="dmy",ax = ax1,
                     color ="mediumblue",marker = "o", ms = 4, lw = 1.2 )
            ax1.plot(df["dim"],
                     wood(df["dim"],woodsettings["init"][0],woodsettings["init"][1],woodsettings["init"][2]),
                     color = "red", linestyle = "--",lw = 0.8)
            ax1.plot(df["dim"],wood(df["dim"],wa,wb,wc),
                     color = "purple", linestyle = "-",lw = 1.8)
            ax1.legend(labels = ["dmy","_","init","wood"])
            ax1.set_title("Wood model, farm " + str(farm) + \
                      ", for cow " + str(cowlac["animal_id"][i]) + \
                          " in parity " + str(cowlac["parity"][i]))
            plt.savefig(os.path.join(path,"results","milk","data_example_wood_farm_" + \
                                     str(farm) + "_cow" + \
                                     str(cowlac["animal_id"][i]) + "_lac" + \
                                     str(cowlac["parity"][i]) + ".tif"))
            plt.close()
        #----------------------------------------------------------------------               
        
        
        # iterative wood model
        ax,p,mod = itw(df["dim"],df["dmy"], 
                       woodsettings["init"][0], woodsettings["init"][1], woodsettings["init"][2],
                       woodsettings["lb"], woodsettings["ub"], plotbool)
        if plotbool == True:
            plt.savefig(os.path.join(path,"results","milk","itw_farm_" + \
                                 str(farm) + "_cow" + \
                                 str(cowlac["animal_id"][i]) + "_lac" + \
                                 str(cowlac["parity"][i]) + ".tif"))
            plt.close()

        # add mod to df and to milk
        mod = mod.rename("mod")
        #df = pd.concat([df,mod], axis = 1)
        milk.loc[df.index.values,"mod"] = mod.values
        milk.loc[df.index.values,"res"] = \
            milk.loc[df.index.values,"dmy"] - milk.loc[df.index.values,"mod"]
        milk.loc[df.index.values,"relres"] = \
            milk.loc[df.index.values,"res"] / milk.loc[df.index.values,"mod"]*100
        milk.loc[milk["relres"].abs() > 100]=0
        
        test = milk.loc[mod.index.values,:].copy()
        # relative residual smoothed
        test["relres"] = test["res"]/test["mod"]*100
        test.loc[test["relres"].abs() > 100]=0
        test["x"] = np.linspace(1,len(test),len(test))
        test["sm"] = test[["relres"]].apply(lambda x: savgol_filter(x,3,1))
        
        if plotbool == True:
            fig, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize=(16,7))
            ax1.plot(test.x,test.relres,marker = 'o', ms=3, color="blue")
            ax1.plot([0, 400],[0, 0],ls='--',lw=0.5, ms=3, color="red")
            ax1.plot(test.x,test.sm,lw=1.2, color="magenta")
        
    # save milk in results - data
    milk.to_csv(os.path.join(path,"results","data","milk_itw_farm_" + str(farm) + ".txt"))

del ax, ax1, ax2, fig, mod, df, wa, wb, wc, p, woodsettings, cowlac, rsample_plot
del plotbool,i,

