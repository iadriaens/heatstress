# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:13:02 2024

@author: u0084712

-------------------------------------------------------------------------------


Investigate 
1) activity shifts in day-rythm / circadian rythm upon heat shifts


-------------------------------------------------------------------------------

- load activity and weather data
- select days with avg THI > 68 + period before

- check activity day pattern / is there a pattern?
- check shift in pattern upon heatwave

- assume / describe causality



"""
import numpy as np
import ruptures as rpt
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.signal import savgol_filter

path = os.path.join("C:/", "Users", "u0084712", "OneDrive - KU Leuven",
                    "projects", "ugent", "heatstress", "datapreprocessing")
os.chdir(path)


# %% load packages, set filepaths, constants and settings

# import statsmodels.api as sm
# import statsmodels.formula.api as smf

# path data
path_data = os.path.join("C:/", "Users", "u0084712", "OneDrive - KU Leuven",
                         "projects", "ugent", "heatstress", "data")

# farm selected
farms = [6, 30, 31, 33, 34, 35, 38, 39, 40, 43, 44, 45, 46, 48]

# %matplotlib qt
sns.set_style("whitegrid")


# %% read and select data

for farm in farms:
    # read data
    data = pd.read_csv(os.path.join(path_data, "act_" + str(farm) + ".txt"),
                       usecols=["farm_id", "animal_id", "parity", 
                                "measured_on", "activity_total",
                                "rumination_time"])
    data["measured_on"] = pd.to_datetime(data["measured_on"],format = "%Y-%m-%d %H:%M:%S")
    data["date"] = pd.to_datetime(data["measured_on"].dt.date,format="%Y-%m-%d")
    data["time"] = data["measured_on"].dt.hour
    data["week"] = data["measured_on"].dt.isocalendar().week
    data["year"] = data["measured_on"].dt.isocalendar().year
    data["time"] = data["time"] - data["time"]%4
    
    
    # correct for hourly vs 2-hourly measurements
    test = (
            data[["farm_id","animal_id","date","time","activity_total","rumination_time"]]
            .groupby(by = ["farm_id","animal_id","date","time"]).sum()
            ).reset_index()
    data = (
            data.drop(columns={"activity_total","rumination_time"})
            .merge(test,on=["farm_id","animal_id","date","time"])
            )
    data=data.loc[data["measured_on"].dt.hour ==data["time"],: ].reset_index(drop=1)
    del test
    
    # load thi data
    wea = pd.read_csv(os.path.join(path,"results","data", "weather_farm_" 
                                    + str(farm) + ".txt"), 
                      usecols = ['date', 'thi'])
    wea["date"] = pd.to_datetime(wea["date"],format='%Y-%m-%d')
    data["date"] = pd.to_datetime(data["date"],format='%Y-%m-%d')
    wea = wea.loc[(wea["date"] >= data["date"].min()) & \
                  (wea["date"] <= data["date"].max()),:]
 
    # merge data and summary with wea
    data = ((
            data.merge(wea,on = ["date"])).sort_values(by=["animal_id","date"])
            .reset_index(drop=1))
    
    # summary of the activity and rumination time and thi values per week of the year
    summary = (
               data[["year","week","time","activity_total","rumination_time","thi"]]
               .groupby(by = ["year","week","time"]).mean()
               ).sort_values(by = ["year","week","time"]).reset_index()
    summary["x"] = summary.index.values
    summary["act_sm"] = summary[["activity_total"]].apply(lambda x: savgol_filter(x, 9*6,2)) # smooth 4.5 weeks each side
    summary["rum_sm"] = summary[["rumination_time"]].apply(lambda x: savgol_filter(x, 9*6,2)) 
    summary["act"] = summary["activity_total"]-summary["act_sm"]
    summary["rum"] = summary["rumination_time"]-summary["rum_sm"]
    summary = summary.merge(data[["year","week","date"]].drop_duplicates(subset=["year","week"]),how="inner",on=["year","week"])

    # # plot ifo thi
    # _,ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(12,8))
    # sns.lineplot(data=summary, x="x", y="activity_total",ax=ax[0],color="lightseagreen",lw=1)
    # sns.lineplot(data=summary, x="x", y="act_sm",ax=ax[0],color="m",lw=2.5)
    # sns.lineplot(data=summary, x="x",y="rumination_time",ax=ax[1],color="lightseagreen",lw=1)
    # sns.lineplot(data=summary, x="x", y="rum_sm",ax=ax[1],color="m",lw=2.5)
    # sns.lineplot(data=summary, x="x", y="thi",ax=ax[0],color="r",lw=1.5)
    # sns.lineplot(data=summary, x="x", y="thi",ax=ax[1],color="r",lw=1.5)
    # ax[1].set_xlim(0,max(summary["x"]))
    # ax[1].set_xticks(np.linspace(0,max(summary["x"]),round(max(summary["x"])/(52*6))))
    # labels = [item.get_text() for item in ax[1].get_xticklabels()]
    # lc =list(map(int,labels))
    # labels = summary.loc[lc,"date"].astype(str)
    # ax[1].set_xticklabels(labels)
    # plt.savefig(os.path.join(path,"results","activity","activity_hour_ts_" + str(farm) + ".tif"))
    # plt.close()

    
    # # summary ifo thi - per day
    # sum2 = (
    #         data[["date","time","activity_total","rumination_time","thi"]]
    #         .groupby(by = ["date","time"]).mean()
    #         ).sort_values(by = ["date","time"]).reset_index()
    # sum2["THI"]=round(sum2["thi"])
    # sum2["x"] = sum2.index.values
    # sum2["act_sm"] = sum2[["activity_total"]].apply(lambda x: savgol_filter(x, 60*6,2)) # smooth 4.5 weeks
    # sum2["rum_sm"] = sum2[["rumination_time"]].apply(lambda x: savgol_filter(x, 60*6,2)) 
    # sum2["act"] = sum2["activity_total"]-sum2["act_sm"]
    # sum2["rum"] = sum2["rumination_time"]-sum2["rum_sm"]
    
    # _,ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(12,8))
    # sns.lineplot(data=sum2, x="x", y="activity_total",ax=ax[0],color="lightseagreen",lw=1)
    # sns.lineplot(data=sum2, x="x", y="act_sm",ax=ax[0],color="m",lw=2.5)
    # sns.lineplot(data=sum2, x="x", y="rumination_time",ax=ax[1],color="lightseagreen",lw=1)
    # sns.lineplot(data=sum2, x="x", y="rum_sm",ax=ax[1],color="m",lw=2.5)
    # sns.lineplot(data=sum2, x="x", y="thi",ax=ax[0],color="r",lw=1.5)
    # sns.lineplot(data=sum2, x="x", y="thi",ax=ax[1],color="r",lw=1.5)
    # plt.savefig(os.path.join(path,"results","activity","activity_hour_ts_" + str(farm) + ".tif"))
    # plt.close()
    
    
    # merge 
    
    
    # summary2 = summary.iloc[0:6000,:].groupby(by=["THI","time"]).agg({"act":["mean","std","count"]}).reset_index()
    # summary2.columns = summary2.columns.droplevel()
    # summary2.columns = ["THI","time","mean","std","count"]   
    
    # # plot the surplus activity on top of normal trend in the data
    # _,ax = plt.subplots(nrows=1,ncols=1,sharex=True,figsize=(14,8))
    # g=sns.scatterplot(data=sum2.sort_values(by="time"),x="thi",y="act",hue="time", 
    #                 palette=["blue","darkslateblue","coral","firebrick","purple","navy"],
    #                 legend="full")
    # handles, labels  =  g.get_legend_handles_labels()
    # g.legend(handles,["00-04","04-08","08-12","12-16","16-20","20-00"])
    # sns.lineplot(data=sum2,x="THI",y="act",hue="time",estimator="mean",errorbar=None, linewidth=3, 
    #              palette=["blue","darkslateblue","coral","firebrick","purple","navy"],
    #              legend=False)
    # ax.plot([68, 68],[-45,45],color="r",lw=2)
    # ax.plot([60, 60],[-45,45],color="r",lw=2,ls="--")
    # ax.set_xlim(35,79)
    # ax.set_ylim(-45,45)
    # ax.set_title("farm = " + str(farm))
    # plt.savefig(os.path.join(path,"results","activity","activity_perhour_thi_" + str(farm) + ".tif"))
    # plt.close()
    
    
    # _,ax = plt.subplots(nrows=1,ncols=1,sharex=True,figsize=(14,8))
    # sns.scatterplot(data=summary,x="thi",y="rum",hue="time", palette=["blue","darkslateblue","coral","firebrick","purple","navy"])
    # sns.lineplot(data=summary,x="THI",y="rum",hue="time",estimator="mean",errorbar="sd", linewidth=3, palette=["blue","darkslateblue","coral","firebrick","purple","navy"])
    # ax.plot([68, 68],[-45,45],color="r",lw=3)
    # ax.plot([60, 60],[-45,45],color="r",lw=2,ls="--")

    # ax.set_xlim(35,79)
    # ax.set_ylim(-45,45)
    # ax.legend(labels=["00-04","04-08","08-12","12-16","16-20","20-00"])
    
    """
    voor italie: gedrag aanpassen
    dus vergelijken act ifv thi in mei/juni vs in augustus?
    """
    
    
    # load weather in higher granularity
    wea = pd.read_csv(os.path.join(path_data,"weather_"+str(farm)+".txt"),
                      usecols = ["datetime","temp","rel_humidity"])
    
    wea["datetime"] = pd.to_datetime(wea["datetime"],format = "%Y-%m-%d %H:%M:%S")
    wea["date"] = wea["datetime"].dt.date
    wea["time"] = wea["datetime"].dt.hour    
    wea["time"] = wea["time"] - wea["time"]%4
    wea["thi"] = 1.8 * wea["temp"] + 32 - \
                    ((0.55 - 0.0055 * wea["rel_humidity"]) * \
                     (1.8 * wea["temp"] - 26))
    # drop hours for which temp or rhum are nan
    idx = wea[["temp","rel_humidity"]].dropna().index
    wea = wea.loc[idx].reset_index(drop=1)
    del idx       

    # groupby per 4 hours thi
    wea = (
           wea[["date","time","thi"]]
           .groupby(by = ["date","time"]).mean()
           ).sort_values(by = ["date","time"]).reset_index()
    wea["date"] = pd.to_datetime(wea["date"],format="%Y-%m-%d")
    
    # merge with data (inner merge) based on date and time
    data = data.drop(columns=["thi"])
    df = data.merge(wea,on=["date","time"])
    df = df.dropna().reset_index(drop=1)
    

    # summary ifo thi - per 4 hours
    sum2 = (
            df[["date","time","activity_total","rumination_time","thi"]]
            .groupby(by = ["date","time"]).mean()
            ).sort_values(by = ["date","time"]).reset_index()
    sum2["THI"]=round(sum2["thi"])
    # set classes to ensure sufficient data are available
    sum2.loc[(sum2["THI"]>70)&(sum2["THI"]<=72),"THI"] = 71 # ensure enough data
    sum2.loc[(sum2["THI"]>72)&(sum2["THI"]<=76),"THI"] = 74 # ensure enough data
    sum2.loc[(sum2["THI"]>76),"THI"] = 78

    sum2["x"] = sum2.index.values
    sum2["act_sm"] = sum2[["activity_total"]].apply(lambda x: savgol_filter(x, 60*6,2)) # smooth 30 days before and after
    sum2["rum_sm"] = sum2[["rumination_time"]].apply(lambda x: savgol_filter(x, 60*6,2)) 
    sum2["act"] = sum2["activity_total"]-sum2["act_sm"]
    sum2["rum"] = sum2["rumination_time"]-sum2["rum_sm"]
    
    sum2.loc[sum2["activity_total"]<=sum2["activity_total"].quantile(0.01),"act"]=np.nan
    sum2.loc[sum2["activity_total"]>=sum2["activity_total"].quantile(0.99),"act"]=np.nan
    sum2.loc[sum2["rumination_time"]<=sum2["rumination_time"].quantile(0.01),"rum"]=np.nan
    sum2.loc[sum2["rumination_time"]>=sum2["rumination_time"].quantile(0.99),"rum"]=np.nan

    
    _,ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(12,8))
    sns.lineplot(data=sum2, x="x", y="activity_total",ax=ax[0],color="lightseagreen",lw=1)
    sns.lineplot(data=sum2, x="x", y="act_sm",ax=ax[0],color="#008080",lw=1.8)
    sns.lineplot(data=sum2, x="x", y="rumination_time",ax=ax[1],color="lightseagreen",lw=1)
    sns.lineplot(data=sum2, x="x", y="rum_sm",ax=ax[1],color="#008080",lw=1.8)
    sns.lineplot(data=sum2, x="x", y="thi",ax=ax[0],color="r",lw=0.6)
    sns.lineplot(data=sum2, x="x", y="thi",ax=ax[1],color="r",lw=0.6)
    ax[0].plot([sum2["x"].min(), sum2["x"].max()], [68, 68],color="k",ls="--",lw=0.8)
    ax[1].plot([sum2["x"].min(), sum2["x"].max()], [68, 68],color="k",ls="--",lw=0.8)
    ax[0].set_xlim(sum2["x"].min(),sum2["x"].max())
    ax[0].set_title("farm = " + str(farm))
    ax[1].set_xticks(np.linspace(0,max(sum2["x"]),round(max(sum2["x"])/(52*6*7))))
    labels = [item.get_text() for item in ax[1].get_xticklabels()]
    lc =list(map(int,labels))
    labels = sum2.loc[lc,"date"].astype(str)
    ax[1].set_xticklabels(labels)
    ax[1].set_xlabel("date")
    plt.savefig(os.path.join(path,"results","activity","activity_hour_ts2_" + str(farm) + ".tif"))
    plt.close()
    
    # plot the surplus activity on top of normal trend in the data
    _,ax = plt.subplots(nrows=1,ncols=1,sharex=True,figsize=(14,8))
    g=sns.scatterplot(data=sum2.sort_values(by="time"),x="thi",y="act",hue="time", 
                    palette=["blue","darkslateblue","coral","firebrick","purple","navy"],
                    legend="full", s = 6,alpha=0.5)
    handles, labels  =  g.get_legend_handles_labels()
    g.legend(handles,["00-04","04-08","08-12","12-16","16-20","20-00"])
    sns.lineplot(data=sum2,x="THI",y="act",hue="time",estimator="mean",errorbar="ci", linewidth=2, 
                  palette=["blue","darkslateblue","coral","firebrick","purple","navy"],
                  legend=False)
    ax.plot([68, 68],[-45,45],color="r",lw=2)
    ax.plot([60, 60],[-45,45],color="r",lw=2,ls="--")
    ax.set_xlim(35,78)
    ax.set_ylim(-45,45)
    ax.set_title("farm = " + str(farm))
    ax.set_xlabel("thi per 4h")
    plt.savefig(os.path.join(path,"results","activity","activity_perhour_thi2_" + str(farm) + ".tif"))
    plt.close()
    
    
    
    # plot the difference in rumination on top of normal trend in the data
    _,ax = plt.subplots(nrows=1,ncols=1,sharex=True,figsize=(14,8))
    g=sns.scatterplot(data=sum2.loc[sum2["rumination_time"]!=0].sort_values(by="time"),x="thi",y="rum",hue="time", 
                    palette=["blue","darkslateblue","coral","firebrick","purple","navy"],
                    legend="full", s = 6,alpha=0.5)
    ax.plot([20, 80],[sum2["rum"].quantile(0.99),sum2["rum"].quantile(0.99)],color="r",lw=1,ls="--")
    ax.plot([20, 80],[sum2["rum"].quantile(0.01),sum2["rum"].quantile(0.01)],color="r",lw=1,ls="--")
    handles, labels  =  g.get_legend_handles_labels()
    g.legend(handles,["00-04","04-08","08-12","12-16","16-20","20-00"])
    sns.lineplot(data=sum2.loc[sum2["rumination_time"]!=0],x="THI",y="rum",hue="time",estimator="mean",errorbar="ci", linewidth=2, 
                 palette=["blue","darkslateblue","coral","firebrick","purple","navy"],
                 legend=False)
    ax.plot([68, 68],[-40,30],color="r",lw=2)
    ax.plot([60, 60],[-40,30],color="r",lw=2,ls="--")
    ax.set_xlim(25,78)
    ax.set_ylim(-40,30)
    ax.set_title("RUMINATION TIME, farm = " + str(farm))
    ax.set_xlabel("thi per 4h")
    plt.savefig(os.path.join(path,"results","activity","rumination_perhour_thi2_" + str(farm) + ".tif"))
    plt.close()
    
    

    #--------------------------------------------------------------------------
    # selection step 1: act = nan
    # selection step 2: dim < 400
    # selection step 3: act < 
    
    
