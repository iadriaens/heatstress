# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:41:14 2024

@author: u0084712
"""


import os

path = os.path.join("C:/", "Users", "u0084712", "OneDrive - KU Leuven",
                    "projects", "ugent", "heatstress", "dataanalysis")
os.chdir(path)

#%% load packages

# from scipy.signal import savgol_filter as savgol
# import matplotlib
import matplotlib.pyplot as plt
# import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from dmy_functions import wood, itw, pert
from scipy.optimize import curve_fit
import seaborn as sns
import numpy as np
import pandas as pd
from random import sample
from itertools import combinations
from itertools import product

# %matplotlib qt


import warnings
warnings.filterwarnings("ignore", category = UserWarning)

#%% paths and constants

# farm selected
farms = [1, 2, 3, 4, 5, 6]
farm=1

# paths
path_data = os.path.join("C:/", "Users", "u0084712", "OneDrive - KU Leuven",
                         "projects", "ugent", "heatstress", "datapreprocessing",
                         "results")
dpath = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataanalysis",
                    "data")


startdates = {1: 2013,
              2: 2014,
              3: 2014,
              4: 2017,
              5: 2016,
              6: 2013}
enddates = {1: 2016,
            2: 2020,
            3: 2017,
            4: 2022,
            5: 2019,
            6: 2020}

#%% visualise and model


for farm in farms:
    # load data
    data = pd.read_csv(os.path.join(path,"results","data","milk_corrected_perturbations_farm_" + \
                             str(farm)+ ".txt"), index_col = 0)
    data["date"] = pd.to_datetime(data["date"], format='%Y-%m-%d')
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["season"] = "winter"
    data.loc[(data["month"] >= 3) & (data["month"] < 6), "season"] = "spring"
    data.loc[(data["month"] >= 6) & (data["month"] < 9), "season"] = "summer"
    data.loc[(data["month"] >= 9) & (data["month"] < 12), "season"] = "autumn"
    
    data = data.loc[(data["date"].dt.year >= startdates[farm]) &
                    (data["date"].dt.year <= enddates[farm]), :].drop_duplicates().reset_index(drop=1)
    data = data.loc[data["dim"] <306,:].reset_index(drop=1)

 
    # add ls class
    data.loc[(data["dim"]) < 22, "ls"] = "0-21"
    data.loc[(data["dim"] >= 22) &
             (data["dim"] < 61), "ls"] = "22-60"
    data.loc[(data["dim"] >= 61) &
             (data["dim"] < 121), "ls"] = "61-120"
    data.loc[(data["dim"] >= 121) &
             (data["dim"] < 201), "ls"] = "121-200"
    data.loc[(data["dim"] >= 201), "ls"] = ">200"
    data.ls = pd.Categorical(data.ls,
                             categories=["0-21", "22-60",
                                         "61-120", "121-200", ">200"],
                             ordered=True)
    
    # load weather
    wea = pd.read_csv(os.path.join(path_data, "data", "weather_prep_farm_" + str(farm) + ".txt"),
                      index_col = 0)
    wea["date"] = pd.to_datetime(wea["date"], format='%Y-%m-%d')

    # define extra features % (in hours) HS present day -1 > day -3
    wea["HS_3_1"] = np.nan
    wea.loc[3:,"HS_3_1"] = (wea["HS_tot"].iloc[0:-3].values + \
                            wea["HS_tot"].iloc[1:-2].values + \
                            wea["HS_tot"].iloc[2:-1].values) / (3*24)*100
   
    # define extra features % (in hours) recovery present day -1 > day -3
    wea["REC_3_1"] = np.nan
    wea.loc[3:,"REC_3_1"] = (wea["HS0"].iloc[0:-3].values + \
                            wea["HS0"].iloc[1:-2].values + \
                            wea["HS0"].iloc[2:-1].values) / (3*24)*100
    
    # define extra features % (in hours) recovery yesterday
    wea["REC_1"] = np.nan
    wea.loc[1:,"REC_1"] = (wea["HS0"].iloc[0:-1].values)/(24)*100
    
    # combine data and wea on date
    data = pd.merge(data,wea,on=["date"],how="inner").sort_values(by=["ID","date"]).reset_index(drop=1)
    

    # correct data for pargroup - dima
    # calculate average parity lactation curve
    parlac = (
        data[["dim","dmy","pargroup"]]
        .groupby(by = ["pargroup","dim"]).mean()
        ).reset_index().rename(columns={"dmy" : "paravg"})
    
    # correct dmy for parity-average
    data = pd.merge(data,parlac,how="left",on = ["pargroup","dim"])
    data["dmy_pc"] = data["dmy"] - data["paravg"]
    

    # plot data ifo dim
    # sns.set_style("whitegrid")
    # fig,ax = plt.subplots(2,1,figsize=(13,10))
    # sns.lineplot(data = data, x="dim", y="dmy",hue = "pargroup", 
    #              errorbar = "sd",estimator="mean", ax=ax[0])
    # sns.lineplot(data = data, x="dim", y="dmy_pc",hue = "pargroup", 
    #              errorbar = "sd",estimator="mean",ax = ax[1])  # parity-avergae corrected
    # ax[0].set_title("daily milk yield")
    # ax[0].set_ylabel("daily milk yield [kg]")
    # ax[0].set_xlabel("days in milk [d]")
    # ax[0].set_xlim(-0.5,305.5)
    # ax[1].set_title("lactation-corrected daily milk yield")
    # ax[1].set_ylabel("daily milk yield [kg]")
    # ax[1].set_xlabel("days in milk [d]")
    # ax[1].set_xlim(-0.5,305.5)

    # std dmy y variable
    data["dmy_s"] = (data["dmy_pc"] - data["dmy_pc"].min()) / \
                      (data["dmy_pc"].max()-data["dmy_pc"].min())
    
                  
    # ax,fig = plt.subplots()         
    # data["dmy_s"].hist(bins = 20)   

    # plot data ifo date
    # sns.set_style("whitegrid")
    # fig,ax = plt.subplots(1,1,figsize=(15,6))
    # sns.lineplot(data = data, x="date", y="dmy_s",hue = "ID", 
    #              ax=ax, linewidth=0.5) # parity-avergae corrected
    # sns.scatterplot(data=data.loc[data["HS_tot"]>0],x="date",
    #             color="r")
    # plt.legend([],[], frameon=False)    
    # test = data[["date","dmy_s","pargroup"]].groupby(by=["pargroup","date"]).mean().reset_index()
    # sns.lineplot(data=test,x="date",y="dmy_s",hue="pargroup")
    # plt.legend([],[], frameon=False)    

    # fig,ax = plt.subplots(1,1,figsize=(15,6))
    # data["THIavg"] = round(data["THIavg"])
    # sns.lineplot(data = data.loc[data["THIavg"]>30], x="THIavg", y="dmy_s",hue = "ID", 
    #              ax=ax, linewidth=0.5) # parity-avergae corrected
    # sns.lineplot(data = data.loc[data["THIavg"]>30], x="THIavg", y="dmy_s",
    #              hue = "ls", estimator="mean",
    #              ax=ax, linewidth=2.5) # parity-avergae corrected
    # sns.scatterplot(data=data.loc[data["HS_tot"]>0],x="date",
    #             color="r")
    # plt.legend([],[], frameon=False)    
    # test = data[["date","dmy_s","pargroup"]].groupby(by=["pargroup","date"]).mean().reset_index()
    # sns.lineplot(data=test,x="date",y="dmy_s",hue="pargroup")
    # plt.legend([],[], frameon=False)    
        

    # ax[0].set_title("perturbation-corrected daily milk yield")
    # ax[0].set_ylabel("daily milk yield [kg]")
    # ax[0].set_xlabel("days in milk [d]")
    # ax[0].set_xlim(-0.5,400.5)
    # ax[1].set_title("lactation-corrected daily milk yield")
    # ax[1].set_ylabel("daily milk yield [kg]")
    # ax[1].set_xlabel("days in milk [d]")
    # ax[1].set_xlim(-0.5,400.5)    
    
    data["THIs"] = (data["THIavg"] - data["THIavg"].min()) / \
                      (data["THIavg"].max()-data["THIavg"].min())
    data["HS0s"] = (data["HS0"] - data["HS0"].min()) / \
                      (data["HS0"].max()-data["HS0"].min()) 
    data["HS13s"] = (data["HS_3_1"] - data["HS_3_1"].min()) / \
                      (data["HS_3_1"].max()-data["HS_3_1"].min())
    data["RE13s"] = (data["REC_3_1"] - data["REC_3_1"].min()) / \
                      (data["REC_3_1"].max()-data["REC_3_1"].min())
    data["RE1s"] = (data["REC_1"] - data["REC_1"].min()) / \
                      (data["REC_1"].max()-data["REC_1"].min())
                      
    # try to catch nonlinearity of the effect "heat load"
    data["HSnl"] = data["HS2"] + 2*data["HS3"] + 4*data["HS4"]
    new = data[["date","HSnl"]].groupby("date").min().reset_index()
    new["HSnl13"] = 0
    new.loc[3:,"HSnl13"] = new["HSnl"].iloc[0:-3].values + \
                             new["HSnl"].iloc[1:-2].values + \
                             new["HSnl"].iloc[2:-1].values
    data = pd.merge(data,new[["date","HSnl13"]],how="left",on="date")
    data["HSnl13s"] = (data["HSnl13"] - data["HSnl13"].min()) / \
                      (data["HSnl13"].max()-data["HSnl13"].min())
                      
                      
    # new recovery parameter
    # if HS_tot = 0: REC = 0, if HS_tot>0 : REC = no hours HS0
    new = data[["date","HS0", "HS_tot"]].groupby("date").min().reset_index()
    new["REC"] = 0
    new.loc[new["HS_tot"]>0,"REC"] =new.loc[new["HS_tot"]>0,"HS0"]
    new["REC13"] = 0
    new.loc[3:,"REC13"] = new["REC"].iloc[0:-3].values + \
                             new["REC"].iloc[1:-2].values + \
                             new["REC"].iloc[2:-1].values
    new.loc[0:2,"REC13"]  =0                   
    data = pd.merge(data,new[["date","REC","REC13"]],how="left",on="date")
    data["RECs"] = (data["REC"] - data["REC"].min()) / \
                      (data["REC"].max()-data["REC"].min())
    data["REC13s"] = (data["REC13"] - data["REC13"].min()) / \
                      (data["REC13"].max()-data["REC13"].min())
                  
    # -------------------------------------------------------------------------
    eq = "dmy_s ~ THIs + HS13s + REC13s " + \
                  "+ C(ls) + C(year) + C(month) + C(year):C(month)" + \
                  " +THIs:C(ls) "
    
    md = smf.mixedlm(eq,
                     data=data,
                     groups=data["ID"],
                     re_formula="~THIs")

    mdf = md.fit(method=["lbfgs"],reml=False)

    print(mdf.summary())
    
    R = {"SSTO": (np.sqrt((data["dmy_s"] - data["dmy_s"].mean())**2)).sum(),
         "SSE": (np.sqrt(mdf.resid**2)).sum()
         }
    R["R2"] = round(1 - (R["SSE"] / R["SSTO"]), 3)
    print(R["R2"]) # 0.512
    print(mdf.aic)  # -276166
    print(
        "correlation random thi slope and intercept = " +
        str(round(mdf.cov_re.Group.THIs / (np.sqrt(mdf.cov_re.Group.Group)
            * (np.sqrt(mdf.cov_re.THIs.THIs))), 3))
    )  #-0.728
    
    wt = mdf.wald_test_terms(scalar=True)
    print(wt)
        
    
    
    # -------------------------------------------------------------------------
    """# THI x pargroup not significant
    # HSnl x pargroup not significant
    eq = "dmy_s ~ THIs + HSnl13s + RE1s" + \
         " + C(ls) + C(year) + C(season) + C(year):C(season)" + \
         " + HSnl13s:C(ls) + THIs:C(ls) + HSnl13s:C(pargroup)"  #-276222
    
    md = smf.mixedlm(eq,
                     data=data,
                     groups=data["ID"],
                     re_formula="~THIs")

    mdf = md.fit(method=["lbfgs"],reml=False)

    print(mdf.summary())
    
    R = {"SSTO": (np.sqrt((data["dmy_s"] - data["dmy_s"].mean())**2)).sum(),
         "SSE": (np.sqrt(mdf.resid**2)).sum()
         }
    R["R2"] = round(1 - (R["SSE"] / R["SSTO"]), 3)
    print(R["R2"]) # 0.512
    print(mdf.aic)  # -276222
    print(
        "correlation random thi slope and intercept = " +
        str(round(mdf.cov_re.Group.THIs / (np.sqrt(mdf.cov_re.Group.Group)
            * (np.sqrt(mdf.cov_re.THIs.THIs))), 3))
    )  #-0.729
    
    wt = mdf.wald_test_terms(scalar=True)
    print(wt)
    
    # -------------------------------------------------------------------------
    # THI x pargroup not significant
    # HSnl x pargroup not significant
    eq = "dmy_s ~ THIs + HSnl13s + REC1s" + \
         " + C(ls) + C(year) + C(season) + C(year):C(season)" + \
         " + THIs:C(ls) + HSnl13s:C(ls) "  #-276222
    
    md = smf.mixedlm(eq,
                     data=data,
                     groups=data["ID"],
                     re_formula="~THIs")

    mdf = md.fit(method=["lbfgs"],reml=False)

    print(mdf.summary())
    
    R = {"SSTO": (np.sqrt((data["dmy_s"] - data["dmy_s"].mean())**2)).sum(),
         "SSE": (np.sqrt(mdf.resid**2)).sum()
         }
    R["R2"] = round(1 - (R["SSE"] / R["SSTO"]), 3)
    print(R["R2"]) # 0.512
    print(mdf.aic)  # -276222
    print(
        "correlation random thi slope and intercept = " +
        str(round(mdf.cov_re.Group.THIs / (np.sqrt(mdf.cov_re.Group.Group)
            * (np.sqrt(mdf.cov_re.THIs.THIs))), 3))
    )  #-0.729
    
    wt = mdf.wald_test_terms(scalar=True)
    print(wt)
    """
    if farm == farms[0]:  # create table
        restable = pd.DataFrame(columns=["farm1", "farm2", "farm3", "farm4", "farm5", "farm6"],
                                index=["R2",
                                       "R2(m)",
                                       "R2(c)",
                                       wt.table.index.values[0],
                                       wt.table.index.values[1],
                                       wt.table.index.values[2],
                                       wt.table.index.values[3],
                                       wt.table.index.values[4],
                                       wt.table.index.values[5],
                                       wt.table.index.values[6],
                                       wt.table.index.values[7],
                                       wt.table.index.values[8],
                                       "random effects correlation",
                                       "residual error variance"])

    restable["farm"+str(farm)]["R2"] = R["R2"]
    restable["farm"+str(farm)][wt.table.index.values[0]] = wt.table.pvalue[0]
    restable["farm"+str(farm)][wt.table.index.values[1]] = wt.table.pvalue[1]
    restable["farm"+str(farm)][wt.table.index.values[2]] = wt.table.pvalue[2]
    restable["farm"+str(farm)][wt.table.index.values[3]] = wt.table.pvalue[3]
    restable["farm"+str(farm)][wt.table.index.values[4]] = wt.table.pvalue[4]
    restable["farm"+str(farm)][wt.table.index.values[5]] = wt.table.pvalue[5]
    restable["farm"+str(farm)][wt.table.index.values[6]] = wt.table.pvalue[6]
    restable["farm"+str(farm)][wt.table.index.values[7]] = wt.table.pvalue[7]
    restable["farm"+str(farm)][wt.table.index.values[8]] = wt.table.pvalue[8]

    restable["farm"+str(farm)]["random effects correlation"] = \
        round(mdf.cov_re.Group.THIs / (np.sqrt(mdf.cov_re.Group.Group)
              * (np.sqrt(mdf.cov_re.THIs.THIs))), 3)
    restable["farm"+str(farm)]["residual error variance"] = round(mdf.scale, 3)

    # fixed effects regression only ################"
    from statsmodels.formula.api import ols
    lmf = ols(formula=eq,
              data=data)
    lm = lmf.fit()
    print(lm.summary2())
    varlm = lm.scale  # variance error  sigma squared epsilon
    
    varlm = np.var(mdf.k_fe*mdf.bse_fe)
    
    
    R["SSE2"] = (np.sqrt(lm.resid**2)).sum()
    R["R2_fix"] = round(1 - (R["SSE2"] / R["SSTO"]), 3)
    print(R["R2_fix"]) # 0.512
    
    # marginal R² = proportion of variance
    # explained by the fixed effects
    # formula 2.4 of https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0213
    # R²(LMM_m) = varlm / (varlm + mdf.cov_re.Group.Group + mdf_re.thi_std.thi_std)
    R["fixed"] = varlm / (varlm + mdf.cov_re.Group.Group +
                          mdf.cov_re.THIs.THIs + mdf.scale)
    # conditional R² = proportion variance explained by fixed + random effects
    # formula 2.5 of https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0213
    R["conditional"] = (varlm + mdf.cov_re.Group.Group + mdf.cov_re.THIs.THIs) / \
        (varlm + mdf.cov_re.Group.Group + mdf.cov_re.THIs.THIs + mdf.scale)

    restable["farm"+str(farm)]["R2(m)"] = round(R["fixed"], 3)
    restable["farm"+str(farm)]["R2(c)"] = round(R["conditional"], 3)

    # fitted values
    data["fitted_lme"] = mdf.fittedvalues
    data["residual_lme"] = mdf.resid
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.scatterplot(data=data,x="dmy_s",y="fitted_lme", hue="ID")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-')
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.legend([],[], frameon=False)    

    re = mdf.random_effects
    re = pd.DataFrame.from_dict(re, orient="index").reset_index()
    re["rc_thi"] = (re["THIs"])/(data["THIavg"].max()-data["THIavg"].min())
    re["ic_thi"] = re["Group"] - \
        (re["THIs"]*data["THIavg"].min() /
         (data["THIavg"].max()-data["THIavg"].min()))
    re = re.sort_values(by="Group").reset_index(drop=1)
    n = len(re)
    cmap = plt.cm.PiYG(np.linspace(0, 1, n))
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    for cow in re.index.values:
        # print(cow)
        x = np.arange(25, 80)
        y = re.loc[cow, "ic_thi"]+re.loc[cow, "rc_thi"]*x
        plt.sca(ax[0])
        plt.plot(x, y, color=cmap[cow])
    
    ax[0].set_title("cow-individual random effects")
    ax[0].set_xlabel("thi")
    ax[0].set_ylabel("cow-individual random effects")
    sns.scatterplot(data=re, x="rc_thi", y="ic_thi",
                    ax=ax[1], color="orangered")
    ax[1].set_title("random effects correlation")
    ax[1].set_xlabel("random thi slope")
    ax[1].set_ylabel("random thi intercept")
    


