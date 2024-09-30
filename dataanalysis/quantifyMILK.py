# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:07:38 2023

@author: u0084712

-------------------------------------------------------------------------------

Quantification and feature calculation of milk yield and perturbations during
    heat stress events.
    
-------------------------------------------------------------------------------
Script contents
- set file paths and constants
- load data
- model milk yield data
- quantify health and perturbations
- design and extract milk yield features

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

# from datetime import date

#%matplotlib qt


#%% set file paths and constants

path_data = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","datapreprocessing",
                    "results","data")

# farm selected
farms = [30,31,33,34,35,38,39,40,43,44,45,46,48]
#farms = [44,45,46,48]

#%% load and process data

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
        
    # save milk in results - data
    milk["res"] = milk["dmy"] - milk["mod"]
    milk.to_csv(os.path.join(path,"results","data","milk_itw_farm_" + str(farm) + ".txt"))

del ax, ax1, ax2, fig, mod, df, wa, wb, wc, p, woodsettings, cowlac, rsample_plot
del plotbool,i,


#%% feature calculation
"""
    find perturbations + health
    Criteria implemented - "no", "very mild", "mild", "moderate", "severe"    
        If less than 5 days below ITW		    = no perturbation
        If >= 5 and less than 10	days below ITW	
            - never < 0.85*ITW					= very mild perturbation
            - 1 or 2 days < 0.85*ITW				= mild perturbation
            - 3 or more days < 0.85*ITW			= moderate perturbation
        If more than 10 days below ITW
            - 0, 1 or 2 days < 0.85*ITW			= mild perturbation
            - 3 or more days, never successive >3 successive days	
                                                 = moderate perturbation
            - 3 or more days, at least once >3 successive days		
                                                = severe perturbation

"""

for farm in farms:
    
    # read milk yield data
    milk = pd.read_csv(os.path.join(path,
                                    "results",
                                    "data",
                                    "milk_itw_farm_" + str(farm) + ".txt"),
                       index_col = 0)
    milk["date"] = pd.to_datetime(milk["date"],format='%Y-%m-%d')
    

    # unique lactations
    cowlac = milk[["animal_id","parity"]].drop_duplicates().reset_index(drop=1)
    rsample_plot = sample(cowlac.index.values.tolist(),3)

    # select lactations and detect perturbations
    for i in range(0,len(cowlac)):
        # select data
        df = milk.loc[(milk["animal_id"] == cowlac["animal_id"][i]) & \
                      (milk["parity"] == cowlac["parity"][i]), \
                          ["dim","dmy","mod"]]
        if len(df["dim"].drop_duplicates()) != len(df["dim"]):
            idx = df["dim"].drop_duplicates().index
            df = df.loc[idx,:]
        
        idx = df.index
            
        if i in rsample_plot:
            print(i)
            plotbool = True
        else:
            plotbool = False
            
        # plot if plotbool = True  || nice curve: farm 48 i = 879
        if plotbool == True:
            fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize=(20,9))
            sns.lineplot(data = df,x = "dim",y = "dmy",
                    color ="blue",marker = "o", ms = 4, lw = 1.2, label = "dmy")
            ax.plot(df["dim"],df["mod"],color = "purple",lw = 2,label = "itw")

        # get perturbations
        pt = pert(df["dim"],df["dmy"],df["mod"])
        df = pd.merge(df,pt, on = "dim").set_index(idx)
        
        # plot perturbations
        if plotbool == True:
            ax.plot(df["dim"],df["thres"],color = "red",lw = 1.2, ls = "--")
            ax.plot(df.loc[df["is_vmild"]==1,"dim"], df.loc[df["is_vmild"]==1,"dmy"],
                    lw = 0, marker = "o", ms = 6, color = "gold", label = "very mild")
            ax.plot(df.loc[df["is_mild"]==1,"dim"], df.loc[df["is_mild"]==1,"dmy"],
                    lw = 0, marker = "s", ms = 6, color = "orange",label = "mild")
            ax.plot(df.loc[df["is_mod"]==1,"dim"], df.loc[df["is_mod"]==1,"dmy"],
                    lw = 0, marker = "*", ms = 6, color = "tomato",label = "moderate")
            ax.plot(df.loc[df["is_sev"]==1,"dim"], df.loc[df["is_sev"]==1,"dmy"],
                    lw = 0, marker = "X", ms = 6, color = "red",label = "severe")
            
            ax.legend()
        
            plt.savefig(os.path.join(path,"results","milk","pert_farm_" + \
                                     str(farm) + "_cow" + \
                                     str(cowlac["animal_id"][i]) + "_lac" + \
                                     str(cowlac["parity"][i]) + ".tif"))
            plt.close()
        
        # save milk in results - data
        milk.loc[df.index.values,"pert_no"] = df["pert_no"].values
        milk.loc[df.index.values,"pert_len"] = df["pert_len"].values
        milk.loc[df.index.values, "is_vmild"] = df["is_vmild"].values
        milk.loc[df.index.values,"is_mild"] = df["is_mild"].values
        milk.loc[df.index.values,"is_mod"] = df["is_mod"].values
        milk.loc[df.index.values,"is_sev"] = df["is_sev"].values
        
    
    milk = milk.fillna(value = 0)
    milk.to_csv(os.path.join(path,"results","data","milk_pert_farm_" + str(farm) + ".txt"))


#%% quantify perturbations: how many, how severe, how long etc per farm

for farm in farms:
    # load data milk yield with perturbations
    milk = pd.read_csv(os.path.join(path,
                                    "results",
                                    "data",
                                    "milk_pert_farm_" + str(farm) + ".txt"),
                       index_col = 0)
    milk["date"] = pd.to_datetime(milk["date"],format='%Y-%m-%d')
    milk["is_pert"] = milk[["is_vmild","is_mild","is_mod","is_sev"]].max(axis=1)
    sep_pert = milk.loc[milk["is_pert"]==1,["animal_id","parity","pert_no"]].drop_duplicates().reset_index(drop=1)
    sep_pert["T"] = 1
    test = sep_pert[["animal_id","parity","T"]].groupby(by = ["animal_id","parity"]).cumsum()
    sep_pert["no"] = test["T"]
    milk = pd.merge(milk,sep_pert[["animal_id","parity","pert_no","no"]],
                    how = "outer",on = ["animal_id","parity","pert_no"])
    del test, sep_pert
    
    # new parity: 1,2,3,4+   - put as PARITY
    milk["newpar"] = milk["parity"]
    milk.loc[milk["newpar"] > 4, "parity"] = 4
    
    
    # number of perturbations per cat per cow-lac
    sumpert = (
              milk.loc[milk["is_pert"]==1,["animal_id","parity","no"]].drop_duplicates()
              .groupby(by = ["animal_id","parity"])
              .max()
              )
    # over all pertubations : duration
    subset = (
              milk.loc[milk["no"].isna() == False, ["animal_id","parity","no","pert_len"]].drop_duplicates()
              .groupby(by = ["animal_id","parity"])
              .mean()
              )
    sumpert["pert_len"] = subset["pert_len"]
    
    # number of perturbations per severity
    subset = (
              milk.loc[milk["no"].isna() == False, ["animal_id","parity","no","is_vmild"]].drop_duplicates()
              .groupby(by = ["animal_id","parity"])
              .sum()
              )
    sumpert["no_vmild"] = subset["is_vmild"]
    subset = (
              milk.loc[milk["no"].isna() == False, ["animal_id","parity","no","is_mild"]].drop_duplicates()
              .groupby(by = ["animal_id","parity"])
              .sum()
              )
    sumpert["no_mild"] = subset["is_mild"]
    subset = (
              milk.loc[milk["no"].isna() == False, ["animal_id","parity","no","is_mod"]].drop_duplicates()
              .groupby(by = ["animal_id","parity"])
              .sum()
              )
    sumpert["no_mod"] = subset["is_mod"]
    subset = (
              milk.loc[milk["no"].isna() == False, ["animal_id","parity","no","is_sev"]].drop_duplicates()
              .groupby(by = ["animal_id","parity"])
              .sum()
              )
    sumpert["no_sev"] = subset["is_sev"]
    
    # duration average of perturbations in each category
    subset = (
              milk.loc[(milk["no"].isna() == False) & (milk["is_vmild"] == 1), 
                       ["animal_id","parity","no","pert_len","is_vmild"]].drop_duplicates()
              .groupby(by = ["animal_id","parity"])
              .mean()
              ).rename(columns = {"pert_len" : "len_vmild"}).reset_index()
    sumpert = pd.merge(sumpert, subset[["animal_id","parity","len_vmild"]],
                       how = "outer",on = ["animal_id","parity"])
    subset = (
              milk.loc[(milk["no"].isna() == False) & (milk["is_mild"] == 1), 
                       ["animal_id","parity","no","pert_len","is_mild"]].drop_duplicates()
              .groupby(by = ["animal_id","parity"])
              .mean()
              ).rename(columns = {"pert_len" : "len_mild"}).reset_index()
    sumpert = pd.merge(sumpert, subset[["animal_id","parity","len_mild"]],
                       how = "outer",on = ["animal_id","parity"])
    subset = (
              milk.loc[(milk["no"].isna() == False) & (milk["is_mod"] == 1), 
                       ["animal_id","parity","no","pert_len","is_mod"]].drop_duplicates()
              .groupby(by = ["animal_id","parity"])
              .mean()
              ).rename(columns = {"pert_len" : "len_mod"}).reset_index()
    sumpert = pd.merge(sumpert, subset[["animal_id","parity","len_mod"]],
                       how = "outer",on = ["animal_id","parity"])
    subset = (
          milk.loc[(milk["no"].isna() == False) & (milk["is_sev"] == 1), 
                   ["animal_id","parity","no","pert_len","is_sev"]].drop_duplicates()
          .groupby(by = ["animal_id","parity"])
          .mean()
          ).rename(columns = {"pert_len" : "len_sev"}).reset_index()
    sumpert = pd.merge(sumpert, subset[["animal_id","parity","len_sev"]],
                   how = "outer",on = ["animal_id","parity"])
    
    # fn
    fn = os.path.join(path,"results","milk","perturbations.xlsx")
    with pd.ExcelWriter(fn, mode = 'a',if_sheet_exists="replace") as writer:
        sumpert.to_excel(writer,sheet_name = "farm_" + str(farm))
        
    # # excel writer
    # writer = pd.ExcelWriter(os.path.join(path,"results","milk","perturbations.xlsx"), engine = 'openpyxl')
    # sumpert.to_excel(writer,sheet_name = "farm_" + str(farm), index=False)
    # # save and close
    # writer.save()
    # writer.close()
    # del writer

    
    # figures
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (15,15))
    
    #1) no perturbations / lactation
    subset = ( 
             milk.loc[milk["no"].isna()==False,["animal_id","parity","no"]]
             .groupby(by =["animal_id","parity"]).max()
             ).reset_index()
    sns.barplot(data = subset,
                x = "parity", y = "no", 
                palette = "flare", errcolor = "grey", errwidth = 1, 
                estimator = "mean",errorbar = "ci", ax = ax[0][0])
    ax[0][0].set_title("avg. number of perturbations")
    ax[0][0].set_ylabel("no.")
    means = subset[['parity','no']].groupby('parity').mean().values
    counts = subset[['parity','no']].groupby(['parity']).count().values
    counts = [str(x) for x in counts.tolist()]
    pos = range(len(counts))
    for tick,label in zip(pos,ax[0][0].get_xticklabels()):
         ax[0][0].text(pos[tick], means[tick] + 0.5, counts[tick],
         horizontalalignment='center', size='x-small', color='k', 
         weight='semibold')
    
    # no of pert / lact/severity
    milk["severity"] = ""
    milk.loc[milk["is_vmild"]==1,"severity"] = "very mild"
    milk.loc[milk["is_mild"]==1,"severity"] = "mild"
    milk.loc[milk["is_mod"]==1,"severity"] = "moderate"
    milk.loc[milk["is_sev"]==1,"severity"] = "severe"
    
    subset = ( 
             milk.loc[milk["no"].isna()==False,["animal_id","parity","severity","no"]].drop_duplicates()
             .groupby(by =["animal_id","parity","severity"]).count()
             ).reset_index()
    sns.barplot(data=subset, x = "parity", y = "no", hue = "severity",
                estimator = "mean",errorbar = "ci",
                palette = "flare", errcolor = "grey", errwidth = 1, 
                hue_order=["very mild","mild", "moderate","severe"],
                ax = ax[0][1])
    ax[0][1].legend(fontsize = "x-small", loc = "upper right")
    ax[0][1].set_title("avg. number of perturbations / severity")
    ax[0][1].set_ylabel("no.")
    ax[0][1].set_ylim(0,6)

    
    # perturbation length
    subset = milk.loc[milk["no"].isna()==False,["animal_id","parity","no","pert_len"]].drop_duplicates()
    sns.barplot(data = subset,
                x = "parity", y = "pert_len", 
                palette = "flare", errcolor = "grey", errwidth = 1, 
                estimator = "mean",errorbar = "ci", ax = ax[1][0])
    
    means = subset[['parity','pert_len']].groupby('parity').mean().values
    counts = subset[['parity','pert_len']].groupby(['parity']).count().values
    counts = [str(x) for x in counts.tolist()]

    # add to the plot
    pos = range(len(counts))
    for tick,label in zip(pos,ax[1][0].get_xticklabels()):
         ax[1][0].text(pos[tick], means[tick] + 1, counts[tick],
         horizontalalignment='center', size='x-small', color='k', 
         weight='semibold')
        
    ax[1][0].set_title("avg. perturbation length")
    ax[1][0].set_ylabel("length [d]")
    ax[1][0].set_ylim(0,20)
    
    # perturbation length per severity
    subset = milk.loc[milk["no"].isna()==False,["animal_id","parity","severity","no","pert_len"]].drop_duplicates()   
    sns.barplot(data = subset,
                x = "parity", y = "pert_len",hue = "severity", 
                hue_order = ["very mild","mild", "moderate","severe"],
                palette = "flare", errcolor = "grey", errwidth = 1, 
                estimator = "mean",errorbar = "ci", ax = ax[1][1])
    ax[1][1].legend(fontsize = "x-small", loc = "upper right")
    ax[1][1].set_title("avg. length of perturbations / severity")
    ax[1][1].set_ylabel("length [d]")
    ax[1][1].set_ylim(0,45)
    
    
    # save and close
    plt.savefig(os.path.join(path,"results","milk","pert_overview_farm_" + \
                             str(farm)+ ".tif"))
    plt.close()

    # severity of perturbations per lact stage/parity
    cowlac = milk.loc[milk["no"].isna()==False,["animal_id","newpar","no"]].drop_duplicates()
    cowlac["dim"] = milk["dim"]
    cowlac["pert_len"] = milk["pert_len"]
    cowlac["ls"] = ''
    cowlac.loc[(cowlac["dim"] + cowlac["pert_len"] / 2) < 22, "ls"] = "0-21"
    cowlac.loc[((cowlac["dim"] + cowlac["pert_len"] / 2) >= 22) & \
               ((cowlac["dim"] + cowlac["pert_len"] / 2) < 61), "ls"] = "22-60"
    cowlac.loc[((cowlac["dim"] + cowlac["pert_len"] / 2) >= 61) & \
               ((cowlac["dim"] + cowlac["pert_len"] / 2) < 121), "ls"] = "61-120"
    cowlac.loc[((cowlac["dim"] + cowlac["pert_len"] / 2) >= 121) & \
               ((cowlac["dim"] + cowlac["pert_len"] / 2) < 201), "ls"] = "121-200"
    cowlac.loc[((cowlac["dim"] + cowlac["pert_len"] / 2) >= 201), "ls"] = ">200"
    cowlac.ls = pd.Categorical(cowlac.ls,
                               categories =["0-21","22-60","61-120","121-200",">200"],
                               ordered = True)
    milk = pd.merge(milk, cowlac[["animal_id","newpar","no","ls"]], how = "outer",
                    on = ["animal_id","newpar","no"])
    
    
    # plot severity per parity and per lactation stage in percentages
    subset = milk.loc[milk["no"].isna()==False,["animal_id","parity","severity","no","ls"]].drop_duplicates()   
    no_pert = subset[["parity","severity","ls"]].groupby(["parity","ls"]).count().reset_index()
    no_pert = no_pert.rename(columns = {'severity':'no_pert'})
    summarypert = subset[["parity","ls","severity","no"]].groupby(["parity","ls","severity"]).count().reset_index()
    summarypert = summarypert.rename(columns = {'no':'count'})
    summarypert = pd.merge(summarypert,no_pert,how = "outer",on=["parity","ls"])
    summarypert["percentage"] = round(summarypert["count"] / summarypert["no_pert"] *100,2)
    summarypert["severity"] = pd.Categorical(summarypert["severity"],
                                             categories=["very mild","mild","moderate","severe"],
                                             ordered = True)
    summarypert = summarypert.sort_values(["parity","ls","severity"]).reset_index(drop=1)
    
    # plot severities
    fig,ax = plt.subplots(1,2,figsize = (12,5))
    sns.barplot(data=summarypert, x="parity",y="percentage",hue = "severity",
                estimator = "mean", errorbar = None, palette = "flare", ax = ax[0])
    ax[0].legend(fontsize = "x-small")
    
    sns.barplot(data=summarypert, x="ls",y="percentage",hue = "severity",
                estimator = "mean", errorbar = None, palette = "flare", ax = ax[1])
    ax[1].legend(fontsize = "x-small")
    
    ax[0].set_xlabel("parity")
    ax[1].set_xlabel("lactation stage, DIM [d]")
    ax[0].set_ylabel("percentage perturbations")
    ax[1].set_ylabel("percentage perturbations")
    ax[0].set_title("severity of perturbations per parity")
    ax[1].set_title("severity of perturbations per lactation stage")
    labels = ["1","2","3","4+"] 
    ax[0].set_xticklabels(labels)
    
    # save and close
    plt.savefig(os.path.join(path,"results","milk","pert_severity_farm_" + \
                             str(farm)+ ".tif"))
    plt.close()

    milk.to_csv(os.path.join(path,"results","data","milk_pert_severity_farm_" + \
                             str(farm)+ ".txt"))
        
        
#%% analysis / quantification with weather features

dpath = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataanalysis",
                    "data")

overview = pd.DataFrame(farms,columns = ["farm_id"])

for farm in farms:
    # load data milk yield with perturbations
    milk = pd.read_csv(os.path.join(path,
                                    "results",
                                    "data",
                                    "milk_pert_severity_farm_" + str(farm) + ".txt"),
                       index_col = 0)
    milk["date"] = pd.to_datetime(milk["date"],format='%Y-%m-%d')
    milk = milk.rename(columns = {"ls" : "pert_ls"})
    milk.pert_ls = pd.Categorical(milk.pert_ls,
                               categories =["0-21","22-60","61-120","121-200",">200"],
                               ordered = True)
    milk.loc[(milk["dim"]) < 22, "ls"] = "0-21"
    milk.loc[(milk["dim"] >= 22) & \
             (milk["dim"] < 61), "ls"] = "22-60"
    milk.loc[(milk["dim"] >= 61) & \
             (milk["dim"] < 121), "ls"] = "61-120"
    milk.loc[(milk["dim"] >= 121) & \
             (milk["dim"] < 201), "ls"] = "121-200"
    milk.loc[(milk["dim"] >= 201), "ls"] = ">200"
    milk.ls = pd.Categorical(milk.ls,
                             categories =["0-21","22-60","61-120","121-200",">200"],
                             ordered = True)

    
    # load thi data
    thi = pd.read_csv(os.path.join(dpath,"weatherfeatures_" + str(farm) + ".txt"),
                     index_col = 0 )
    thi["date"] = pd.to_datetime(thi["date"],format='%Y-%m-%d')
    
    # merge thi with milk on date
    data = pd.merge(milk,thi, on = ["farm_id","date"],how="outer")
    data = data.sort_values(by = ["farm_id","animal_id","newpar","dim"]).reset_index(drop=1)
    # delete weather info when no cow data
    data = data.loc[data["animal_id"].isna() == False,:].reset_index(drop=1)
    
    
    """
    TODO: investigate frequency of perturbations with periods of high THI
        THI high: % days YES/NO perturbation
        THI not high: % days with perturbation
    """
    
    # % days thi high and perturbation
    overview.loc[overview["farm_id"]==farm,"perc_pert_highTHI"] = \
        round(len(data.loc[(data.is_pert==1) & (data.thi_high == 1)]) / \
                len(data.loc[(data.thi_high == 1)]) * 100,2)
    # % days with perturbation in the entire dataset
    overview.loc[overview["farm_id"]==farm,"perc_pert_overall"] = \
        round(len(data.loc[(data.is_pert==1)]) / \
              len(data) * 100,2)
    
    # % days thi high and negative residual
    overview.loc[overview["farm_id"]==farm,"perc_negres_highTHI"] = \
        round(len(data.loc[(data.res<0) & (data.thi_high == 1)]) / \
              len(data.loc[(data.thi_high == 1)]) * 100,2)
    # % days with negative residual
    overview.loc[overview["farm_id"]==farm,"perc_negres_overall"] = \
        round(len(data.loc[(data.res<0)]) / \
              len(data) * 100,2)
            
            
    #model - initially just "thi avg"
    
    """
    model for average effects = Y = DMY corrected for parity- lactation curve
    model for cow-individual effects: DMY corrected for ITW
    
    """
    
    # plot residuals of ITW - these are being modelled 
    data["relres"] = (data["res"]/data["mod"]) * 100
    sns.set_style("darkgrid") 
    fig,ax = plt.subplots(1,2, figsize = (15,8))
    sns.boxplot(data = data, 
                 x = "ls", y = "res", 
                 hue = "parity",
                 ax = ax[0], palette = "dark",
                 showfliers=False)
    sns.boxplot(data = data, 
                x = "ls",y = "relres", 
                hue = "parity", 
                ax = ax[1], palette = "dark",
                showfliers=False)
    ax[0].set_title("absolute residual of ITW")
    ax[0].set_ylabel("residual milk yield [kg]")
    ax[0].set_xlabel("lactation stage, dim [d]")
    ax[0].legend(fontsize = "small")
    ax[1].set_title("relative residual of ITW")
    ax[1].set_ylabel("residual milk yield [%]")
    ax[1].set_xlabel("lactation stage, dim [d]")
    ax[1].legend(fontsize = "small")
    plt.savefig(os.path.join(path,"results","milk","itw_distr_farm_" + \
                             str(farm)+ ".tif"))
    plt.close()

    # non-normality is expected in the residual milk yield data, as the residuals as
    # compared to the expected lactation model are more often negative than positive
    # still, these are more valid to model in comparison with THI than when using raw
    # DIM data that is for example corrected for herd/parity effects of lactation stage,
    # given that individual lactation curves, even per herd/parity, vary lots.
    # first attempt: test with herd/parity corrected lactation curves
    
    herdpar = data[["parity","dim","dmy"]].groupby(by = ["parity","dim"]).mean().reset_index()
    herdpar = herdpar.rename(columns={"dmy":"paravg"})
    data = pd.merge(data,herdpar,on = ["parity","dim"])
    data["res_pc"] = data["dmy"] - data["paravg"]
    
    # plot residuals of parity corrected DMY - can be modelled as well
    sns.set_style("darkgrid") 
    fig,ax = plt.subplots(1,2, figsize = (15,8))
    sns.boxplot(data = data, 
                 x = "ls", y = "res_pc", 
                 hue = "parity",
                 ax = ax[0], palette = "dark",
                 showfliers=False)
    ax[0].set_title("parity-average corrected DMY")
    ax[0].set_ylabel("residual milk yield [kg]")
    ax[0].set_xlabel("lactation stage [d]")
    sns.countplot(data = data,
                  x = "month", 
                  hue = "parity", palette = "dark",
                  ax = ax[1])
    ax[1].set_xlabel("month")
    ax[1].set_title("data distribution over months")
    ax[1].set_ylabel("number of days DMY data")
    labels = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"] 
    ax[1].set_xticklabels(labels)
    plt.savefig(os.path.join(path,"results","milk","parcor_dmy_distr_farm_" + \
                             str(farm)+ ".tif"))
    plt.close()
    
    # here we see that on average, LS 22-60 has negative residuals, while you expect
    # residuals around 0. this means that more "strong" outlying negatieve residuals 
    # are present in the data in this LS. We can take this into account in the model
    # by estimating a fixed effect for LS, but we need to make sure LS and season are
    # not confounded. The right plot of the figure shows this is not the case.
    
    # -----------------------------------------------------------------------------
    # estimate non-linearity of effect of THI first so we can correct for this
    # model first with THI as linear effect, then express residual ifo. thi
    
    # change LS back to non-ordered categorical, idem for parity class
    data.ls = pd.Categorical(data.ls, ordered = False)
    data.parity = pd.Categorical(data.parity, ordered = False)
    data = data.loc[data["thi_avg"].isna()==False,:].reset_index(drop=1)
    data.thi_avg = round(data.thi_avg).astype(int)
    
    # if interaction terms, standardise/scale for convergence
    data["thi_std"] = (data.thi_avg - data.thi_avg.mean()) / data.thi_avg.std()
    
    # year season
    ys = data[["year","month"]].drop_duplicates().reset_index(drop=1)
    ys["season"] = "winter"
    ys.loc[(ys["month"] >= 3) & (ys["month"] < 6), "season"] = "spring"
    ys.loc[(ys["month"] >= 6) & (ys["month"] < 9), "season"] = "summer"
    ys.loc[(ys["month"] >= 9) & (ys["month"] < 12), "season"] = "autumn"
    ys["season"] = pd.Categorical(ys.season,
                                  categories = ["winter","spring","summer","autumn"],
                                  ordered = True)
    yscombi = ys[["year","season"]].drop_duplicates().sort_values(by = ["year","season"]).reset_index(drop=1)
    yscombi["ysclass"] = np.linspace(1, len(yscombi),len(yscombi),endpoint = True, dtype = int)
    ys = pd.merge(ys,yscombi, on = ["year","season"])
    data = pd.merge(data,ys, on = ["year","month"])
    
    # year month combi
    ys = data[["year","month"]].drop_duplicates().reset_index(drop=1)
    ys = ys.sort_values(by = ["year","month"])
    ys["ymclass"] = np.linspace(1, len(ys),len(ys),endpoint = True, dtype = int)
    data = pd.merge(data,ys, on = ["year","month"])
    
    #############################################################################
    # first model: res_pc ~ 1 + thi + ls + parity + thi*ls + thi*parity + ys + (1 + thi | animal_id)
    
    md = smf.mixedlm("res_pc ~ thi_std + C(ls) + C(parity) + thi_std*C(ls) + thi_std*C(parity) + C(ymclass)",
                     data=data,
                     groups = data["animal_id"],
                     re_formula = "~thi_std")
    mdf = md.fit(method=["lbfgs"])
    with open(os.path.join("results","milk","summary_" + str(farm) + ".txt"), 'w') as fh:
        fh.write(mdf.summary().as_text())
    
    print(mdf.summary())
    print(mdf.cov_re)
    
    #TODO calculate R2 adjusted (wald test)
    print(mdf.wald_test_terms(scalar=True))
    
    
    # correlation = cov / sqrt(varx)*sqrt(vary)
    print(
          "correlation random thi slope and intercept = " + \
          str(round(mdf.cov_re.Group.thi_std / (np.sqrt(mdf.cov_re.Group.Group)*(np.sqrt(mdf.cov_re.thi_std.thi_std))),3))
          )
        
    # fitted values
    data["fitted_lme"] = mdf.fittedvalues
    data["residual_lme"] = mdf.resid
    
    # plot residuals ifo thi_std / thi_avg
    fig,ax = plt.subplots(1,1,figsize = (12,6))
    sns.scatterplot(data = data,
                   x = "thi_avg", y = "residual_lme", style = "parity", hue = "parity",
                   palette="rocket",size = 3)
    ax.set_xlim(ax.get_xlim()[0],ax.get_xlim()[1])
    ax.plot([25,78],[0,0],"k--",lw = 1)
    ax.set_title("model residuals ifo of average thi, farm = "+str(farm))
    ax.set_ylabel("residual dmy [kg]")
    ax.set_xlabel("thi")
    ax.legend(labels = ["1","2","3","4","_"],fontsize = "small")
    plt.savefig(os.path.join(path,"results","milk","mdl_res_vs_thi_farm_" + \
                             str(farm)+ ".tif"))
    plt.close()
    
    fig,ax = plt.subplots(2,1,figsize = (17,14))
    sns.boxplot(data = data,
                x = "thi_avg", y = "residual_lme",
                fliersize=1, ax= ax[0])
    ax[0].plot([-0.50,52],[0,0],lw = 1, color = "r", ls = "--")
    ax[0].set_xlim(-0.5,51.5)
    ax[0].set_title("model residuals ifo average thi, farm = " +str(farm))
    ax[0].set_xlabel("average thi")
    ax[0].set_ylabel("dmy - model residuals [kg]")
    sns.countplot(data = data,
                  x = "thi_avg", ax= ax[1])
    ax[1].set_xlim(-0.5,51.5)
    ax[1].set_title("number of observations at each thi, farm = "+str(farm))
    ax[1].set_xlabel("average thi")
    ax[1].set_ylabel("no. of observations")
    plt.savefig(os.path.join(path,"results","milk","mdl_res2_vs_thi_farm_" + \
                             str(farm)+ ".tif"))
    plt.close()
    
    # normality of residuals and predicted vs observed plot
    fig,ax = plt.subplots(1,3,figsize = (20,6))
    sns.distplot(mdf.resid,hist=False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm, ax = ax[0])
    ax[0].set_title("KDE plot of residuals (blue) and normal distribution (black)")
    ax[0].set_xlabel("model residuals")
    ax[0].set_ylabel("density")
    
    pp = sm.ProbPlot(mdf.resid, fit=True)
    qq = pp.qqplot(marker='.', markerfacecolor='b', markeredgecolor='b', alpha=0.3, ax = ax[1])
    sm.qqline(qq.axes[1], line='45', fmt='r--')
    ax[1].set_title("qq-plot, farm = "+str(farm))
    ax[1].set_xlabel("theoretical quantiles (std)")
    ax[1].set_ylabel("sample quantiles")
    
    ax[2].plot(data["fitted_lme"],data["res_pc"],
            lw = 0,
            color = "indigo", marker = "x",ms = 3)
    ax[2].plot([-40,30],[-40,30],color = "r",lw = 1.5,ls = "--")
    ax[2].set_xlim(-40,30)
    ax[2].set_ylim(-40,30)
    ax[2].set_xlabel("predicted (trendcorr) dmy [kg]")
    ax[2].set_ylabel("observed (trendcorr) dmy [kg]")
    ax[2].set_title("predicted vs. observed plot")
    plt.savefig(os.path.join(path,"results","milk","mdl_res_stats_farm_" + \
                             str(farm)+ ".tif"))
    plt.close()
    
    # summer visualisation when heatstress is relevant
    years = data["year"].drop_duplicates().sort_values().reset_index(drop=1)
    for year in years:
        
        # model residuals
        subset = data.loc[(data["month"]>0) & (data["month"]<=13) & \
                          (data["year"] == year),:].reset_index(drop=1)
        if len(subset)>1000:
            ds = subset[["date","residual_lme","dmy"]].groupby(by="date").mean().reset_index()
            ds = ds.rename(columns = {"residual_lme" : "avg_res",
                                      "dmy":"avg_dmy"})
            sub = subset[["date","residual_lme","dmy"]].groupby(by="date").quantile(0.1).reset_index()
            sub = sub.rename(columns = {"residual_lme" : "q10_res",
                                      "dmy":"q10_dmy"})
            ds["q10_res"] = sub["q10_res"]
            ds["q10_dmy"] = sub["q10_dmy"]
            sub = subset[["date","residual_lme","dmy"]].groupby(by="date").quantile(0.9).reset_index()
            sub = sub.rename(columns = {"residual_lme" : "q90_res",
                                      "dmy":"q90_dmy"})
            ds["q90_res"] = sub["q90_res"]
            ds["q90_dmy"] = sub["q90_dmy"]
        
            fig, ax = plt.subplots(2,1,figsize = (16,12))
            ax[0].fill_between(ds["date"],
                            ds["q10_dmy"],ds["q90_dmy"],
                            alpha = 0.85, color = "mediumblue", lw= 1)
            ax[0].plot(ds["date"],ds["avg_dmy"], "lightcyan", lw = 1.5)
            ax[0].plot(ds["date"],ds["avg_dmy"].mean()*np.ones((len(ds),1)), "k--", lw = 1.5)
            ax[0].set_ylabel("daily milk yield [kg] mean+90%CI")
            ax[0].set_xlabel("date")
            thi = subset[["date","thi_avg"]].groupby("date").mean().reset_index()
            ax2 = ax[0].twinx()
            ax2.plot(thi["date"],thi["thi_avg"],'r',lw=1.5)
            ax2.fill_between(thi["date"],
                             68,80, color = "red",alpha = 0.2
                             )
            ax2.grid(visible = False)
            ax2.set_ylim(-40,80)
            ax2.set_xlim(subset["date"].min(),subset["date"].max())
            ax2.set_ylabel("average daily thi (red)")
            
            ax[1].fill_between(ds["date"],
                            ds["q10_res"],ds["q90_res"],
                            alpha = 0.85, color = "mediumblue", lw= 1)
            ax[1].plot(ds["date"],ds["avg_res"], "lightcyan", lw = 1.5)
            ax[1].plot(ds["date"],np.zeros((len(ds),1)), "k--", lw = 1.5)
            ax[1].set_ylabel("residual daily milk yield [kg] mean+90%CI")
            thi = subset[["date","thi_avg"]].groupby("date").mean().reset_index()
            ax2 = ax[1].twinx()
            ax2.plot(thi["date"],thi["thi_avg"],'r',lw=1.5)
            ax2.grid(visible = False)
            ax2.set_ylim(-40,80)
            ax2.set_ylabel("average daily thi (red)")
            ax[1].set_xlabel("date")
            ax2.fill_between(thi["date"],
                             68,80, color = "red",alpha = 0.2
                             )
            ax2.set_xlim(thi["date"].min(),thi["date"].max())
            plt.savefig(os.path.join(path,"results","milk","mdl_dmy_res_year_" + str(int(year)) + "_farm_" + \
                                     str(farm)+ ".tif"))
            plt.close()
            
            # episodes of very high THI - plot separately
            # first and last month with high THI
            thihigh = subset.loc[subset["thi_avg"]>68,"month"]
            
            subset = data.loc[(data["month"]>=thihigh.min()) & (data["month"]<=thihigh.max()) & \
                              (data["year"] == year),:].reset_index()
            if len(subset)> 500:
                ds = subset[["date","residual_lme","dmy"]].groupby(by="date").mean().reset_index()
                ds = ds.rename(columns = {"residual_lme" : "avg_res",
                                          "dmy":"avg_dmy"})
                sub = subset[["date","residual_lme","dmy"]].groupby(by="date").quantile(0.1).reset_index()
                sub = sub.rename(columns = {"residual_lme" : "q10_res",
                                          "dmy":"q10_dmy"})
                ds["q10_res"] = sub["q10_res"]
                ds["q10_dmy"] = sub["q10_dmy"]
                sub = subset[["date","residual_lme","dmy"]].groupby(by="date").quantile(0.9).reset_index()
                sub = sub.rename(columns = {"residual_lme" : "q90_res",
                                          "dmy":"q90_dmy"})
                ds["q90_res"] = sub["q90_res"]
                ds["q90_dmy"] = sub["q90_dmy"]
            
                fig, ax = plt.subplots(2,1,figsize = (16,12))
                ax[0].fill_between(ds["date"],
                                ds["q10_dmy"],ds["q90_dmy"],
                                alpha = 0.85, color = "teal", lw= 1)
                ax[0].plot(ds["date"],ds["avg_dmy"], "lightcyan", lw = 1.5)
                ax[0].plot(ds["date"],ds["avg_dmy"].mean()*np.ones((len(ds),1)), "k--", lw = 1.5)
                ax[0].set_ylabel("daily milk yield [kg] mean+90%CI")
                ax[0].set_xlabel("date")
                thi = subset[["date","thi_avg"]].groupby("date").mean().reset_index()
                ax2 = ax[0].twinx()
                ax2.plot(thi["date"],thi["thi_avg"],'r',lw=1.5)
                ax2.fill_between(thi["date"],
                                 68,80, color = "red",alpha = 0.2
                                 )
                ax2.grid(visible = False)
                ax2.set_ylim(-40,80)
                ax2.set_xlim(subset["date"].min(),subset["date"].max())
                ax2.set_ylabel("average daily thi (red)")
                
                ax[1].fill_between(ds["date"],
                                ds["q10_res"],ds["q90_res"],
                                alpha = 0.85, color = "teal", lw= 1)
                ax[1].plot(ds["date"],ds["avg_res"], "lightcyan", lw = 1.5)
                ax[1].plot(ds["date"],np.zeros((len(ds),1)), "k--", lw = 1.5)
                ax[1].set_ylabel("residual daily milk yield [kg] mean+90%CI")
                thi = subset[["date","thi_avg"]].groupby("date").mean().reset_index()
                ax2 = ax[1].twinx()
                ax2.plot(thi["date"],thi["thi_avg"],'r',lw=1.5)
                ax2.grid(visible = False)
                ax2.set_ylim(-40,80)
                ax2.set_ylabel("average daily thi (red)")
                ax[1].set_xlabel("date")
                ax2.fill_between(thi["date"],
                                 68,80, color = "red",alpha = 0.2
                                 )
                ax2.set_xlim(thi["date"].min(),thi["date"].max())
                plt.savefig(os.path.join(path,"results","milk","mdl_dmy_res_summer_year_" + str(int(year)) + "_farm_" + \
                                         str(farm)+ ".tif"))
                plt.close()
            
overview.to_csv(os.path.join(path,"results","milk","perturbation_freq_perfarm.txt"))



#%% TODO: per animal - individual differences per cow
"""
21684.0
6      21766.0
32     21798.0
39     21828.0
96     21817.0
130    21714.0
133    21718.0
134    21753.0
162    21826.0
173    21719.0
"""


# subject analysis - fit per cowlac
anid = 21719.0
re = mdf.random_effects[anid]



fig, ax = plt.subplots(2,2,figsize = (10,10))
cmaps = {"1":"Blues",
         "2":"Greens",
         "3":"Purples",
         "4":"Oranges"}
for p in data.parity.drop_duplicates().sort_values():
    print(p)
    anids = data.loc[data["parity"] == p,"animal_id"]
    norm = mpl.colors.Normalize(vmin=0, vmax=len(anids))
    this = np.linspace(data.thi_avg.min(),data.thi_avg.max())
    col = mpl.cm.get_cmap(cmaps[str(int(p))])
    tcol=0
    for anid in anids:
        print(anid)
        re = mdf.random_effects[anid]
        ax.plot(this, re["Group"]+this*re["thi_std"], lw = 1, c=col(norm(tcol)))
        tcol = tcol+1
# plot data and models per subject




# TODO: model with delayed effect of thi
# TODO: models for other THI effects
# TODO: models activity vs milk yield
# TODO: functions for plotting and model evaluation in separate script