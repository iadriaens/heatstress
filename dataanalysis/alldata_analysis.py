# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:12:23 2024

@author: u0084712
"""

import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataanalysis")
os.chdir(path)

#%% load packages, change path, set farms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dmy_functions import qreg
from datetime import timedelta
# %matplotlib qt

path_data = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","datapreprocessing",
                    "results","data","new")
farms = [41,42,43,44,46,47,48,49,50,51,55,57,58,59,61,62,64,65,66,67,69]  
#

#%% load and combine data

for farm in farms:
    #--------------------------------------------------------------------------
    #-------------------------------- DATA PREP -------------------------------
    #--------------------------------------------------------------------------
    # read data - milk, activity, weather, heat stress events
    milk = pd.read_csv(os.path.join(path_data,"milk_preprocessed_" + str(farm) + ".txt"),
                       index_col = 0)
    milk["date"] = pd.to_datetime(milk["date"],format = "%Y-%m-%d")
    act = pd.read_csv(os.path.join(path_data,"act_preprocessed_" + str(farm) + ".txt"),
                       usecols=["farm_id","animal_id","parity","date","dim","median_corr"])
    act["date"] = pd.to_datetime(act["date"],format = "%Y-%m-%d")
    wea = pd.read_csv(os.path.join(path_data,"weather_daysum_farm_" + str(farm) + ".txt"),
                       index_col = 0)
    wea["date"] = pd.to_datetime(wea["date"],format = "%Y-%m-%d")
    hs = pd.read_csv(os.path.join(path_data,"weather_hs_farm_" + str(farm) + ".txt"),
                       index_col = 0)
    hs["start"] = pd.to_datetime(hs["start"],format = "%Y-%m-%d")
    hs["end"] = pd.to_datetime(hs["end"],format = "%Y-%m-%d")

    # -------------------------------------------------------------------------
    # model milk yield with adjusted quantile regression method
    cowlac = milk[["animal_id","parity"]].drop_duplicates().reset_index(drop=1)
            
    # adjusted quantile regression model
    milk["mod"] = 0
    for i in range(0,len(cowlac)):
        X = milk.loc[(milk["animal_id"] == cowlac["animal_id"][i]) & \
                        (milk["parity"] == cowlac["parity"][i]),"dim"]
        y = milk.loc[(milk["animal_id"] == cowlac["animal_id"][i]) & \
                        (milk["parity"] == cowlac["parity"][i]),"dmy"]    
        _,mod,_ = qreg(4,X,y,15,0.5,0.7,False)
        milk.loc[X.index,"mod"] = mod["mod"]
    del i,X,y,mod
    milk["res"] = milk["dmy"]-milk["mod"]
    
    #--------------------------------------------------------------------------
    # correct for heteroscedasticity in act by standardising via Q15-Q90 of surrounding 7 weeks
    
    # actday1 = act[["date","median_corr"]].groupby(by = 'date').quantile(0.05)
    # actday1["q05"] = actday1["median_corr"].rolling("49d",min_periods=1, center=True, closed='both').mean()
    # actday1["q95_notroll"] = act[["date","median_corr"]].groupby(by = 'date').quantile(0.95)["median_corr"].values
    # actday1["q95"] = actday1["q95_notroll"].rolling("49d",min_periods=1, center=True, closed='both').mean()

    # plt.plot(act["date"],act["median_corr"]) 
    # plt.plot(actday1["q05"])
    # plt.plot(actday1["q95"])
    
    # actday1 = actday1.reset_index()
    # act = act.merge(actday1[["date","q05","q95"]], on = "date")
    # act["median_corr"] = (act["median_corr"] - act["q05"]) / (act["q95"]-act["q05"])

    act = act.sort_values("date").reset_index(drop=1)
    test = act[["date","median_corr"]].set_index("date")
    test2 = test.rolling("49d",min_periods=1, center=True, closed='both').quantile(0.05).set_index(act.index)["median_corr"].rename("q15")
    test3 = test.rolling("49d",min_periods=1, center=True, closed='both').quantile(0.95).set_index(act.index)["median_corr"].rename("q85")
    # test2.plot()
    # test3.plot()
    act["median_corr"] = (act["median_corr"] - test2) / (test3-test2)
    del test2, test,test3
    # plt.plot(act["date"],act["median_corr"])
    # plt.plot(act["date"],test2)
    # plt.plot(act["date"],test3)
    # _,_ = plt.subplots()
    # plt.plot(act["date"],act["median_corr2"])
   
    # _,_ = plt.subplots()
    # sns.lineplot(data=act,x="date",y="median_corr2",errorbar="sd")
    
    
    #--------------------------------------------------------------------------
    # combine datasets based on date
    df = milk.merge(act,on=["farm_id","animal_id","parity","date","dim"],how="inner")    
    df = df.merge(wea,on="date",how="inner")
    df = df.loc[df["median_corr"].isna()==False,:]
    df = df.sort_values(["animal_id","date"]).reset_index(drop=1)
    del cowlac, act, milk, wea
    
    # select hs events that fall within dataset 'df'
    hs = hs.loc[(hs["start"]>df["date"].min())&(hs["end"]< df["date"].max()),:].reset_index(drop=1)
    hs["no"] = hs["no"] - (hs["no"].min()-1)
    df["no"] = df["no"]- (df["no"].min()-1)
    
    #--------------------------------------------------------------------------
    #----------------------------- explore herd IN event  ---------------------
    #--------------------------------------------------------------------------
    # subset = df.loc[(df["date"].dt.month >= 6) & (df["date"].dt.month <= 9),:].reset_index(drop=1)
    # subset["severity"] =subset["severity"].fillna(0)
    # fig,ax = plt.subplots()
    # sns.boxplot(data=subset,x="severity", y="res")
    # fig,ax = plt.subplots()
    # sns.boxplot(data=subset,x="severity", y="median_corr")
    # del subset,fig,ax
    
    #--------------------------------------------------------------------------
    #----------------------------- data per animal/event  ---------------------
    #--------------------------------------------------------------------------
    # select events
    events = df[["no","animal_id","parity","severity"]].drop_duplicates()
    events = events.dropna().reset_index(drop=1)
    if len(events)>0:
        
        events = events.merge(hs[["no","total_excess","start","end","mean_thi","heat_load"]],on="no",how="outer")
        events = events.dropna().reset_index(drop=1)
        events = events.sort_values(by=["no","animal_id"]).reset_index(drop=1)
        events["ref_act"] = np.nan
        events["ref_dmy"] = np.nan
        events["par"]=events["parity"]
        events.loc[events["par"]>3,"par"]=3
        
        
        # calculate individual parameters, first in a fixed time frame - 
        # reference period = last 10 days before event without heat stress
        rnd = events.sample(
            12)[["animal_id", "parity"]]  # random plots
        sns.set_style("whitegrid")
        for i in events.index.values:
            sub = df.loc[(df["animal_id"]==events["animal_id"][i]) & \
                         (df["parity"]==events["parity"][i]) & \
                         (df["no"].isna() == True) & \
                         (df["date"] < events["start"][i])]
            sub = sub[-10:]
            if len(sub)>=5:  # only calculate the reference when at least 5 days of data
                events.loc[i,"ref_act"] = sub["median_corr"].mean()
                events.loc[i,"ref_dmy"] = sub["res"].mean()
                events.loc[i,"refdim"] = sub["dim"].median()
            else:
                events.loc[i,"ref_act"] = np.nan
                events.loc[i,"ref_dmy"] = np.nan
                events.loc[i,"refdim"] = sub["dim"].median()
        
        # group ls classes based on ref dim
        events["ls"] = ''
        events.loc[(events["refdim"]) < 22, "ls"] = "0-21"
        events.loc[(events["refdim"] >= 22) &
                 (events["refdim"] < 61), "ls"] = "22-60"
        events.loc[(events["refdim"] >= 61) &
                 (events["refdim"] < 121), "ls"] = "61-120"
        events.loc[(events["refdim"] >= 121) &
                 (events["refdim"] < 201), "ls"] = "121-200"
        events.loc[(events["refdim"] >= 201), "ls"] = ">200"
        
        # # visualise reference values
        # fig,ax = plt.subplots(2,2,figsize=(12,10))
        # fig.suptitle("reference periods farm " + str(farm))
        # ax[0][0].axhline(0,ls= '--', color='red',lw=1)
        # sns.boxplot(data=events,x="severity", y="ref_act",hue="par",ax=ax[0][0],fliersize=1)
        # ax[0][1].axhline(0,ls= '--', color='red',lw=1)
        # sns.boxplot(data=events,x="severity", y="ref_act",hue="ls",
        #             hue_order = ["0-21","22-60","61-120","121-200",">200"],
        #             ax=ax[0][1],fliersize=1)
        # ax[1][0].axhline(0,ls= '--', color='red',lw=1)
        # sns.boxplot(data=events,x="severity", y="ref_dmy",hue="par",ax=ax[1][0],fliersize=1)
        # ax[1][1].axhline(0,ls= '--', color='red',lw=1)
        # sns.boxplot(data=events,x="severity", y="ref_dmy",hue="ls",
        #             hue_order = ["0-21","22-60","61-120","121-200",">200"],
        #             ax=ax[1][1],fliersize=1)
        # plt.savefig(os.path.join(path,"results","new","refs_farm_"+str(farm)+".png"))
        # plt.close()
        
        # del sub, i, fig, ax
        
        #----------------------------------------------------------------------
        #------------------------- calculate change during hs -----------------
        #----------------------------------------------------------------------
        # activity: days of heat stres (start->end) as compared to ref period
        events["act_avg"] = np.nan   # mean activity
        events["act_max"] = np.nan    # max activity
        events["act_dur"] = np.nan
        events["act_sum"] = np.nan
        events["dmy_exp"] = np.nan    # mean expected dmy (production level)
        events["dmy_avg"] = np.nan
        events["dmy_min"] = np.nan
        events["dmy_dur"] = np.nan
        events["dmy_sum"] = np.nan
       
        for i in events.index.values:
            # ref for act =  period of HS
            sub = df.loc[(df["animal_id"]==events["animal_id"][i]) & \
                         (df["date"] >= events["start"][i]) & \
                         (df["date"] <= events["end"][i])]
            events.loc[i,"act_avg"] = sub["median_corr"].mean()
            events.loc[i,"act_max"] = sub["median_corr"].max()
            events.loc[i,"act_sum"] = sub["median_corr"].sum()
            
            # during HS: avg milk yield expected
            events.loc[i,"dmy_exp"] = sub["mod"].mean()
            
            # for milk yield "period of interest" = hs period + 3 with first day not counted
            td = min(3,(events["end"][i]-events["start"][i]).days+1)
            
            sub = df.loc[(df["animal_id"]==events["animal_id"][i]) & \
                         (df["date"] > events["start"][i]) & \
                         (df["date"] <= events["end"][i]+timedelta(days=td))]     # td days after event
            events.loc[i,"dmy_avg"] = sub["res"].mean()   # average residual of expected lactation curve
            events.loc[i,"dmy_min"] = sub["res"].min()    # min residual of expected lacation curve
            events.loc[i,"dmy_sum"] = sub["res"].sum()    # total milk loss as compared to expected lactation curve
                
            # finally, calculate duration of deviation based on number of days after event 
            #      - act first time under reference value
            #      - milk yield is higher than reference
            sub = df.loc[(df["animal_id"]==events["animal_id"][i]) & \
                         (df["parity"]==events["parity"][i]) & \
                         (df["date"]>=events["end"][i])]
            try:
                events.loc[i,"act_dur"] = sub.loc[(sub["median_corr"]<events["ref_act"][i])].iloc[0].name - \
                                      sub.index.values[0]
            except:
                events.loc[i,"act_dur"] = len(sub)
            try:
                events.loc[i,"dmy_dur"] = sub.loc[(sub["res"]>= events["ref_dmy"][i])].iloc[0].name - \
                                      sub.index.values[0]
            except:
                events.loc[i,"act_dur"] = len(sub)
            events.loc[i,"hs_in_ref"] = min((sub.iloc[0:events.loc[i,"dmy_dur"].astype(int)]["no"].isna()==False).sum(),1)
            
            events.to_csv(os.path.join(path,"results","new","data_farm_"+str(farm) +'.txt'))
            
            if i in rnd.index.values:
                sub = df.loc[(df["animal_id"]==events["animal_id"][i]) & \
                             (df["parity"]==events["parity"][i])]
                fig,ax = plt.subplots(2,1,figsize=(16,6),sharex=True)
                plt.subplots_adjust(wspace=0, hspace=0)
                ax[1].grid("on",which="major",axis="both",color=[0.9,0.9,0.9],alpha=0.4)
                ax[0].grid("on",which="major",axis="both",color=[0.9,0.9,0.9],alpha=0.4)
                ax[0].plot(sub["dim"],sub["dmy"],'.-',ms=4,color="blue")
                ax[0].plot(sub["dim"],sub["mod"],color = 'teal',lw=2)
                ax[0].fill_between(sub["dim"],np.zeros(len(sub)),
                                (2+sub["dmy"].max())*np.ones(len(sub)),
                                where=((sub["no"].isna()==False) & (sub["severity"]==1)),
                                color= 'purple',alpha= 0.6)
                ax[0].fill_between(sub["dim"],np.zeros(len(sub)),
                                (2+sub["dmy"].max())*np.ones(len(sub)),
                                where=((sub["no"].isna()==False) & (sub["severity"]==2)),
                                color= 'orangered',alpha= 0.6)
                ax[0].fill_between(sub["dim"],np.zeros(len(sub)),
                                (2+sub["dmy"].max())*np.ones(len(sub)),
                                where=((sub["no"].isna()==False) & (sub["severity"]==3)),
                                color= 'firebrick',alpha= 0.6)
                ax[0].set_title("farm " + str(farm) + ", " + \
                                "cow " + str(sub.iloc[0].loc["animal_id"]) + \
                                " in parity " + str(sub.iloc[0].loc["parity"]) )
                ax[1].set_xlabel("days in milk (DIM) [days]")
                ax[0].set_ylabel("daily milk yield (DIM), [kg]")   
                ax[1].set_ylabel("median corrected activity, [steps]")         
                ax[1].plot(sub["dim"],sub["median_corr"],'.-',ms=4,color="steelblue")
                ax[1].plot(sub["dim"],np.zeros(len(sub)),":",color = 'teal',lw=2)
                ax[1].fill_between(sub["dim"],(sub["median_corr"].min()-2)*np.ones(len(sub)),
                                (2+sub["median_corr"].max())*np.ones(len(sub)),
                                where=((sub["no"].isna()==False) & (sub["severity"]==1)),
                                color= 'purple',alpha= 0.6)
                ax[1].fill_between(sub["dim"],(sub["median_corr"].min()-2)*np.ones(len(sub)),
                                (2+sub["median_corr"].max())*np.ones(len(sub)),
                                where=((sub["no"].isna()==False) & (sub["severity"]==2)),
                                color= 'orangered',alpha= 0.6)
                ax[1].fill_between(sub["dim"],(sub["median_corr"].min()-2)*np.ones(len(sub)),
                                (2+sub["median_corr"].max())*np.ones(len(sub)),
                                where=((sub["no"].isna()==False) & (sub["severity"]==3)),
                                color= 'firebrick',alpha= 0.6)
                ax[0].set_xlim(sub["dim"].min(),sub["dim"].max())
                ax2 = ax[0].twiny()
                ax2.grid(False)
                ax2.plot(sub["date"], sub["dmy"], linestyle="-", linewidth=0,
                          marker="s", markersize=0,
                          color="white")
                ax2.set_xlim([sub["date"].min(), sub["date"].max()])
                ax[1].plot([sub["dim"].min(),sub["dim"].max()],
                           [events["ref_act"][i],events["ref_act"][i]],
                           '--',lw=1,color="r")
                ax[1].set_ylim(sub["median_corr"].min()-0.5,0.5+sub["median_corr"].max())
                ax[0].set_ylim(sub["dmy"].min()-2,2+sub["dmy"].max())
                plt.savefig(os.path.join(path,"results","new","data_farm_"+str(farm)+\
                                         "_cow_" +str(events["animal_id"][i]) + "_par_" +\
                                         str(events["parity"][i]) + " .png"))
                    
                plt.close()


#%% data selection - event
"""
certain events are too light to have any effect, but were needed for 
calculation of the reference period. Now, we want to select the events to 
include in the analysis.

1. selection
2. relative change as compared to reference period
3. relation activity and milk yield- model development
4. investigation of covariates etc.



"""

import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.formula.api import ols
from openpyxl import load_workbook


def results_df(lm):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = lm.pvalues
    pvals[(pvals<0.001)] = 0.001   
    coeff = lm.params
    conf_lower = lm.conf_int()[0]
    conf_higher = lm.conf_int()[1]

    results = pd.DataFrame({"beta":coeff,
                            "pvals":pvals,
                            "ci_lb":conf_lower,
                            "ci_ub":conf_higher
                            })
    results = pd.concat([results, 
                         pd.DataFrame([round(lm.rsquared,4),np.nan,np.nan,np.nan],
                                       columns = ["R²"],
                                       index = results.columns).T])
    results = pd.concat([results, 
                         pd.DataFrame([lm.aic,np.nan,np.nan,np.nan],
                                       columns = ["AIC"],
                                       index = results.columns).T])
    results = pd.concat([results, 
                         pd.DataFrame([lm.scale,np.nan,np.nan,np.nan],
                                       columns = ["scale"],
                                       index = results.columns).T])
    results = pd.concat([results, 
                         pd.DataFrame([len(lm.model.endog),np.nan,np.nan,np.nan],
                                       columns = ["n_obs"],
                                       index = results.columns).T])
    return results



dict_results = {}

for farm in farms:

    events = pd.read_csv(os.path.join(path,"results","new","data_farm_" + \
                                      str(farm) +'.txt'), index_col=0)
    events.loc[events["ls"] == "0-21","ls"] = 0
    events.loc[events["ls"] == "22-60","ls"] = 1
    events.loc[events["ls"] == "61-120","ls"] = 2
    events.loc[events["ls"] == "121-200","ls"] = 3
    events.loc[events["ls"] == ">200","ls"] = 4
    
    events["dev_act"] = (events["act_max"]-(events["ref_act"]))  # average change act
    events["dev_dmy"] = (events["dmy_avg"]-(events["ref_dmy"]))  # average change dmy
    
    # fig,ax = plt.subplots()
    # sns.scatterplot(data=events,x="dev_act",y="dev_dmy",hue="ls",size="par")
    
    """
    model: outc var = dev_dmy
           pred var = dev_act,(severity)
           covr var = C(par), C(ls), C(ls):C(par), dev_act:C(par), dev_act:C(ls) 
    
    model fitting: 
    # parity not significant
    
    """
    # try: activity-powers
    
    # standardisation/scaling min-max
    events["y"] = (events["dev_dmy"]-events["dev_dmy"].min()) / \
                  (events["dev_dmy"].max()-events["dev_dmy"].min())
    events["x"] = (events["dev_act"]-events["dev_act"].min()) / \
                  (events["dev_act"].max()-events["dev_act"].min())    
    
    events = events.dropna().reset_index(drop=1)
    
    #--------------------------------------------------------------------------
    # simple linear model, deviation of milk yield predicted by deviation of act
    # model = "model0"
    # lmf = ols(formula="y ~ x + np.power(x, 2) + C(ls) + C(severity) + x:C(severity)",
    #            data=events)
    # lm = lmf.fit()
    # print(lm.summary2())
    
    #--------------------------------------------------------------------------
    # additionally take into account the dmy level as compared to herd
    # relative standardised change as x and y variable
    events["rel_dmy"] = events["ref_dmy"]/events["dmy_exp"]*100
    events["rel_dmy_class"] = 2
    events.loc[events["rel_dmy"]<events["rel_dmy"].quantile(0.33),"rel_dmy_class"] = 3
    events.loc[events["rel_dmy"]>events["rel_dmy"].quantile(0.66),"rel_dmy_class"] = 1
    
    #--------------------------------------------------------------------------
    # fit models
    model = "model6"
    lmf = ols(formula="y ~ x + np.power(x, 2) " + \
                         "+ C(ls) + C(severity) " + \
                         "+ x:C(severity) + np.power(x,2):C(severity)",    
              data=events)
    lm = lmf.fit()
    print(lm.summary2())
    print("rsquared = " + str(round(lm.rsquared,4)))
    
    #--------------------------------------------------------------------------
    # create summary table
    results = results_df(lm)
    
    # read and write results file with model number, R², R²adj, AIC, nobs
    gof = pd.read_csv(os.path.join(path, "results","new","model_gof.txt"),index_col=0)
    # model, rsquared, rsqadj, aic, farm, nobs, formula
    gof_new = pd.DataFrame([model,round(lm.rsquared,4),round(lm.rsquared_adj,4),lm.aic,farm,lm.nobs,lm.model.formula],
                           index = gof.columns).T
    gof = pd.concat([gof,gof_new]).reset_index(drop=1)  
    gof.to_csv(os.path.join(path, "results","new","model_gof.txt"))   
    gof = gof.sort_values(by = ["farm","rsquared"],ascending=[True,False])                  
    
    # add standardisation 
    results = pd.concat([results, 
                         pd.DataFrame([np.nan,np.nan,events["dev_dmy"].min(),events["dev_dmy"].max()],
                                       columns = ["standardisation delta dmy"],
                                       index = results.columns).T])
    results = pd.concat([results, 
                         pd.DataFrame([np.nan,np.nan,events["dev_act"].min(),events["dev_act"].max()],
                                       columns = ["standardisation delta act"],
                                       index = results.columns).T])
    results = pd.concat([results,
                         pd.DataFrame([np.nan,np.nan,events["rel_dmy"].quantile(0.33),events["rel_dmy"].quantile(0.66)],
                                       columns = ["quantile ref dmy"],
                                       index = results.columns).T])
    
    
    dict_results["farm_" + str(farm)] = results
    
    
#---------------------------- SAVE RESULTS ------------------------------------    
# file and filename
fn = model+".xlsx"
path_out = os.path.join(path,"results","new",fn)
with pd.ExcelWriter(path_out, engine="xlsxwriter") as writer:
    pd.DataFrame([lm.model.formula]).to_excel(writer,
                                              sheet_name = "formula",
                                              index=False,
                                              header=False)
    
    for farm in dict_results.keys():
        b = farm
        dict_results[farm].to_excel(writer, 
                         sheet_name = b,
                         index = True)

    
    #--------------------------------------------------------------------------
    # SEVERITY 3 ONLY
    # events2 = events.loc[events["severity"]==1].copy()
    # events2["y"] = (events2["dev_dmy"]-events2["dev_dmy"].min()) / \
    #               (events2["dev_dmy"].max()-events2["dev_dmy"].min())
    # events2["x"] = (events2["dev_act"]-events2["dev_act"].min()) / \
    #               (events2["dev_act"].max()-events2["dev_act"].min())              
    # events2 = events2.dropna().reset_index(drop=1)
    # events2["rel_dmy_class"] = 2
    # events2.loc[events2["rel_dmy"]<events2["rel_dmy"].quantile(0.33),"rel_dmy_class"] = 3
    # events2.loc[events2["rel_dmy"]>events2["rel_dmy"].quantile(0.66),"rel_dmy_class"] = 1
    
    # # rel_dmy_class NOT significant, not in interactions either when event is severe!
    # lmf = ols(formula="y ~ x + np.power(x, 2) + C(ls) + C(rel_dmy_class) + x:C(rel_dmy_class) + np.power(x,2):C(rel_dmy_class)",
    #            data=events2)
    # lm = lmf.fit()
    # print(lm.summary2())
    
    
    
    # lmf = ols(formula="y ~ x + np.power(x, 2) + C(ls) + C(severity) + C(rel_dmy_class) + x:C(rel_dmy_class) + np.power(x,2):C(rel_dmy_class)",
    #           data=events)
    # lm = lmf.fit()
    # print(lm.summary2())
    # fig = sm.graphics.plot_fit(lm, "x", ms=2)
    
    # # rel_dmy_class NOT significant, not in interactions either when event is severe!
    # lmf = ols(formula="y ~ x + np.power(x, 2) + C(ls) + x:C(ls) + np.power(x,2):C(ls) + C(rel_dmy_class) + x:C(rel_dmy_class) + np.power(x,2):C(rel_dmy_class)",
    #            data=events)
    # lm = lmf.fit()
    # print(lm.summary2())
    #--------------------------------------------------------------------------
    # # standardisation/scaling mean-std  == doesnt make a difference
    # events["y"] = (events["dev_dmy"]-events["dev_dmy"].mean()) / \
    #               (events["dev_dmy"].std())
    # events["x"] = (events["dev_act"]-events["dev_act"].mean()) / \
    #               (events["dev_act"].std())              
    # events = events.dropna().reset_index(drop=1)
    #--------------------------------------------------------------------------
    # test with mixed model show there is no consistency in the relation 
    # between increase in act and decrease in dmy that is not already captured 
    # by the marginal model
    #--------------------------------------------------------------------------
    # # test with taking activity level into account has no significant effect
    # events["ref_act_s"] = events["ref_act"] = (events["ref_act"]-events["ref_act"].min()) / \
    #               (events["ref_act"].max()-events["ref_act"].min())
    # lmf = ols(formula="y ~ x + np.power(x, 2) + C(ls) + C(severity) + x:C(severity) + ref_act_s",
    #           data=events)
    # lm = lmf.fit()
    # print(lm.summary2())
    #--------------------------------------------------------------------------
    

#todo: models max act
    # events["dev_act"] = (events["act_max"]-(events["ref_act"]))  # average change act

    
    
#%% bin
    # # merge and combine data with df
    # data = df.merge(hs[["no","start","severity"]], 
    #                 left_on = "date",right_on = "start",
    #                 how="outer").drop(columns="start")
    # data = data.merge(hs[["no","end","severity"]], 
    #                 left_on = "date",right_on = "end",
    #                 how="outer").drop(columns="end")
    # data["no"] = data[["no_x","no_y"]].max(axis=1) 
    # data["severity"] = data[["severity_x","severity_y"]].max(axis=1) 
    # data = (
    #        data.drop(columns=["no_x","no_y","severity_x","severity_y"])
    #        .sort_values(by=["animal_id","date"])
    #        .reset_index(drop=1)
    #        )
    # data["no"] = data["no"].interpolate()
    # data["severity"] = data["severity"].interpolate()
    # data.loc[(data["no"]/np.floor(data["no"]))!=1,["no","severity"]] = np.nan    
    # data["date"] = pd.to_datetime(data["date"],format = "%Y-%m-%d")
    # del milk, act, wea, hs
    
    
    model = "model1"
    lmf = ols(formula="y ~ x + np.power(x, 2) + C(ls) + C(severity) + C(rel_dmy_class) + x:C(rel_dmy_class) + np.power(x,2):C(rel_dmy_class)",
              data=events)
    lm = lmf.fit()
    # print(lm.summary2())
    # fig = sm.graphics.plot_fit(lm, "x", ms=2)
    
    
    """
    output:
        - quantiles of DMY
        - lm:
            ° variables
            ° coeff + se
            ° pval
            ° ci
            
            ° scale
            ° R²
            ° rel_dmy quantiles (0.33,0.66)
            ° standardisation and 'translation' into words deviation vs kg milk
 
    """
    
    dict_std = {"dev_dmy_min" : events["dev_dmy"].min(),
                "dev_dmy_max" : events["dev_dmy"].max(),
                "dev_act_min" : events["dev_act"].min(),
                "dev_act_max" : events["dev_act"].max(),
                "rel_res_q33" : events["rel_dmy"].quantile(0.33),
                "rel_res_q66" : events["rel_dmy"].quantile(0.66)}
    
    # backcalculate relation with kg / steps
    pval = round(lm.pvalues,3)
    pval.loc[pval<0.001]=0.001
    result = lm.params.reset_index()
    result.columns = ["var","beta"]
    result["pval"] = pval.values
    result = pd.concat([result,])
    
    # for LS0,severity1,rel
    test = events[["ls","severity","rel_dmy_class"]].drop_duplicates()
    
    x = np.linspace(0,1,100)
    
    # model is of the form ax² + bx + c, with a,b ~ rel_dmy_class, c ~ ls, severity, rel_dmy_class
    
    a1 = results["beta"][12]                         # rel_dmy_class = 1
    a2 = results["beta"][12] + result["beta"][13]    # rel_dmy_class = 2
    a3 = results["beta"][12] + result["beta"][13]    # rel_dmy_class = 3
    
    b1 = result["beta"][9]                          # rel_dmy_class = 1
    b2 = result["beta"][9] + result["beta"][10]     # rel_dmy_class = 2
    b3 = result["beta"][9] + result["beta"][11]     # rel_dmy_class = 3
    
    # rel_dmy_class = 1
    y1 = dict_std['dev_dmy_min'] + 0.5*(dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) + \
        b1 * (dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) * x + \
        a1 * (dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) * x**2
    # rel_dmy_class = 2
    y2 = dict_std['dev_dmy_min'] + 0.5*(dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) + \
        b2 * (dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) * x + \
        a2 * (dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) * x**2
    # rel_dmy_class = 3
    y3 = dict_std['dev_dmy_min'] + 0.5*(dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) + \
        b3 * (dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) * x + \
        a3 * (dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) * x**2
        
    fig,ax = plt.subplots()
    xt = x*(dict_std['dev_act_max']-dict_std['dev_act_min']) + dict_std['dev_act_min']
    ax.plot(xt,y1,color="red",label="dmy severity 1 (no pert)")
    ax.plot(xt,y2,color="blue",label="dmy severity 2 (small pert)")
    ax.plot(xt,y3,color="green",label="dmy severity 3 (large pert)")
    plt.legend()
    ax.plot(xt,np.zeros(len(xt)),"--",lw=0.5,color="grey")
    ax.set_xlim(min(xt),max(xt))
    ax.set_title("farm " + str(farm) + ", n = " + str(len(events))+", R² = " + str(round(lm.rsquared,3)))
    ax.set_ylabel("\u0394" + "DMY [kg]")
    ax.set_xlabel("\u0394" + "ACT [steps]")
    plt.savefig(os.path.join(path,"results","new","model1_farm_"+str(farm) + \
                              " .png"))
        
    plt.close()
    del a1,a2,a3,b1,b2,b3,xt,ax,fig,y1,y2,y3,x,test,pval,dict_std
    
    
    
    
    
    # model is of the form ax² + bx + c, with a,b ~ rel_dmy_class, c ~ ls, severity, rel_dmy_class
    
    a1 = results["beta"][10]                         # rel_dmy_class = 1
    a2 = results["beta"][10] + result["beta"][11]    # rel_dmy_class = 2
    a3 = results["beta"][10] + result["beta"][12]    # rel_dmy_class = 3
    
    b1 = result["beta"][7]                          # rel_dmy_class = 1
    b2 = result["beta"][7] + result["beta"][8]     # rel_dmy_class = 2
    b3 = result["beta"][7] + result["beta"][9]     # rel_dmy_class = 3
    
    
    
    # rel_dmy_class = 1
    y1 = dict_std['dev_dmy_min'] + 0.5*(dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) + \
        b1 * (dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) * x + \
        a1 * (dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) * x**2
    # rel_dmy_class = 2
    y2 = dict_std['dev_dmy_min'] + 0.5*(dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) + \
        b2 * (dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) * x + \
        a2 * (dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) * x**2
    # rel_dmy_class = 3
    y3 = dict_std['dev_dmy_min'] + 0.5*(dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) + \
        b3 * (dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) * x + \
        a3 * (dict_std['dev_dmy_max']-dict_std['dev_dmy_min']) * x**2