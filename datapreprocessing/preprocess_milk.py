# -*- coding: utf-8 -*-
"""
Created on Fri May 26 08:33:46 2023

@author: u0084712

-------------------------------------------------------------------------------
DAILY MILK YIELD DATA
-------------------------------------------------------------------------------
GOAL:
== overall: link between individual heat stress susceptibility and behaviour
-- this script: preprocess and select daily yield data

-------------------------------------------------------------------------------
STEPS:
1. data exploration

2. data selection

3. data preprocessing

4. final selection

"""

import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","datapreprocessing")
os.chdir(path)


#%% import packages

import pandas as pd
import numpy as np
import seaborn as sns
# import statsmodels
import matplotlib.pyplot as plt
from datetime import date


#%% plot settings

# %matplotlib qt
sns.set_style("whitegrid")

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


del SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE

#%% set paths and constants

path_data = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","data","new")

# farm selected
# farms = [30,31,33,34,35,38,39,40,43,44,45,46,48] 
farms = [41,42,43,44,46,47,48,49,50,51,54,55,57,58,59,61,62,64,65,66,67,68,69] 


#%% read and preprocess data

for farm in farms:
    # read data
    data = pd.read_csv(os.path.join(path_data, "milk_" + str(farm) + ".txt"),
                       index_col=0, 
                       usecols=['milking_id', 'milking_oid', 'farm_id', 'animal_id', 'lactation_id',
                               'parity', 'ended_at', 'mi', 'dim', 'tmy'])
    data["ended_at"] = pd.to_datetime(data["ended_at"],format='%Y-%m-%d %H:%M:%S')
    data = data.sort_values(by = ["animal_id","ended_at"]).reset_index()
    print(data["ended_at"].min())
    
    # correct floating number accuracy errors of days in milk parameter: if >23:50
    idx = data.loc[(data["ended_at"].dt.hour == 23) & \
                   (data["ended_at"].dt.minute > 40) & \
                   (data["dim"] % 1 == 0),"dim"].index    # index where dim = wrong
    dim = (np.floor(data.loc[idx.values+1,"dim"])-1 + (23/24)).values + \
          (data.loc[idx.values,"ended_at"].dt.minute / 3600).values + \
          (data.loc[idx,"ended_at"].dt.minute / 86400).values
    data.loc[idx,"dim"] = dim
    del dim
    
    # milking interval to hours -- lely sets all MI > 48h to 48h
    # data["mi"] = data["mi"]/(3600)
    
    # correct data ~ MI: data with MI >= 48 hours & dim < 1 : first milking
    data.loc[(data["mi"] >= 48) & (data["dim"] < 1), "mi"] = np.nan
    data = data.drop((data.loc[(data["mi"] >= 48) & (data["tmy"].isna())]).index).reset_index(drop=1)
    
    # ----------------------------- calculate DMY based on MI------------------
    data["hour"] = data["ended_at"].dt.hour + data["ended_at"].dt.minute / 60
    
    # fraction of milk produced in current day    
    data["fraction1"] = 1
    idx = data.loc[(data["hour"] - data["mi"] < 0)].index  # if mi not in same day
    data.loc[idx,"fraction1"] = data.loc[idx,"hour"] / data.loc[idx,"mi"]
    if (0 in idx.values):
        idx = idx[1:]
    
    # fraction of milk given in next day
    idx_prev = idx-1   # milkings to add on 
    data["tmy_nextday"] = 0
    data.loc[idx_prev.values,"tmy_nextday"] = data.loc[idx,"tmy"].values
    data["fraction2"] = 0
    data.loc[idx_prev.values,"fraction2"] = 1-(data.loc[idx.values,"fraction1"].values)
    
    # total milk = fraction1*tmy + fraction2 * tmy_nextday
    data["tmy_corrected"] = data["fraction1"] * data["tmy"] + \
                            data["fraction2"] * data["tmy_nextday"]
    data["date"] = pd.to_datetime(data["ended_at"].dt.date,format = "%Y-%m-%d")
    data["ddim"] = np.floor(data["dim"])
    
    # daily data
    milk = (
            data[["farm_id","animal_id","parity","date","ddim","tmy_corrected"]]
            .groupby(by = ["farm_id","animal_id","parity","date","ddim"]).sum()
            ).reset_index()
    milk = milk.rename(columns={"ddim": "dim",
                        "tmy_corrected" : "dmy"})
    
    #------------------------ data visualisation if dim > 650 -----------------
    
    """
        The visualisations of long lactation curves show that sometimes the 
        correction of parity/dim and calving date fails to set the right dim/dates
        to the correct data. This is really rare, and cannot be solved without
        1) individually checking and correcting each lactation curve
        2) proposing a general approach, with the risk of introducing more 
           errors. 
        For now, we chose to "leave" the data as-is. The rare cases that an error
        remains after the selection phase, will probably not affect analysis
        results.
    """
    
    # subset = milk.loc[milk["dim"]>600,["animal_id"]].drop_duplicates()
    
    # # plot + save random 10 with extended lactation (seems OK)
    # for cow in subset.sample(5)["animal_id"].values:
    #     print(cow)
    #     fig,ax = plt.subplots(1,1,figsize = (15,6))
    #     sns.lineplot(data=milk.loc[milk["animal_id"]==cow,:],x = "date",y = "dmy",
    #                  hue = "parity",linewidth = 1.5,marker = "o",markersize=4,
    #                  palette = sns.color_palette("bright",n_colors = len(milk.loc[milk["animal_id"]==cow,"parity"].drop_duplicates())))
    #     ax.set_title("farm = " + str(farm) + ", cow = " + str(cow) + ", extended lactation > 600 days", size=14)
    #     ax.set_ylabel("daily milk yield [kg]")
    #     plt.savefig(os.path.join(path,"results","milk","farm_ " + str(farm) + "_extended_lactations_cow" + str(cow) + ".tif"))
    #     plt.close()
    
    # del ax, cow, fig, idx, idx_prev, subset
    
    #---------------------------- data selection steps ------------------------
    
    # select data with DIM <= 600
    milk = milk.loc[milk["dim"]<=400,:]
    milk = milk.sort_values(by = ["animal_id","parity","date","dim"]).reset_index(drop=1)
    
    # calculate gap
    milk["gap"] = (milk["dim"].iloc[0:-1] - milk["dim"].iloc[1:].reset_index(drop=1))
    idx = (milk[["animal_id","parity"]].drop_duplicates(inplace = False, keep = "last")).index.values
    milk.loc[idx,"gap"] = -1
    del idx
    
    # selection of lactations based on start-end dim + gapsize
    cowlac = (
              milk[["animal_id","parity","dim","gap"]].groupby(by = ["animal_id","parity"])
              .agg({"dim":["min","max"],
                    "gap":["min"]})).reset_index()
    cowlac.columns = cowlac.columns.droplevel()
    cowlac.columns = ["animal_id","parity","dim_min","dim_max","gap"]
    print("farm " + str(farm) + " has " + str(len(cowlac)) + " unique lactations")
    cowlac = cowlac.loc[(cowlac["dim_min"]<=5) & \
                        (cowlac["dim_max"]>=65) & \
                        (cowlac["gap"]>=-5),:]
    print("      from these, " + str(len(cowlac)) + " remain")
    
    # select in milk
    milk = pd.merge(milk[["farm_id","animal_id","parity","date","dim","dmy"]],
                    cowlac[["animal_id","parity"]], 
                    how="inner", on = ["animal_id","parity"]).reset_index(drop=1)
    # save milk 
    milk.to_csv(os.path.join(path,"results","data","new","milk_preprocessed_" + str(farm) + ".txt"))
    
    # --------------------------- weather data --------------------------------
    weather = pd.read_csv(os.path.join(path_data, "weather_" + str(farm) + ".txt"),
                       index_col=0)
    weather["datetime"] = pd.to_datetime(weather["datetime"], format = "%Y-%m-%d %H:%M:%S")
    weather["date"] = pd.to_datetime(weather["datetime"].dt.date, format = "%Y-%m-%d")
    weather = weather[["date","temp","rel_humidity"]].groupby(by = ["date"]).mean().reset_index()
    
    # calculate THI = (1.8 × Tmean + 32) − [(0.55 − 0.0055 × RHmean) × (1.8 × Tmean − 26)]
    weather["thi"] = 1.8 * weather["temp"] + 32 - \
                    ((0.55 - 0.0055 * weather["rel_humidity"]) * \
                     (1.8 * weather["temp"] - 26))
        
    # aggregate to weekly data for plotting
    weather["week_ref"] = weather["date"] - \
                          pd.to_timedelta(weather["date"].dt.weekday, unit = "d")
    weekweather = weather[["week_ref",
                           "temp",
                           "rel_humidity",
                           "thi"]].groupby(by = "week_ref").agg({"temp":["mean","max"], 
                                                                 "rel_humidity":["mean","max"],
                                                                 "thi" :["mean","max"]}).reset_index()
    weekweather.columns = weekweather.columns.droplevel()
    weekweather.columns = ["week_ref","temp_mean","temp_max",
                                      "RH_mean","RH_max",
                                      "thi_mean","thi_max"]
    
    #-------------------------- visualisation per date ------------------------
    
    milk["week_ref"] = milk["date"] - pd.to_timedelta(milk["date"].dt.weekday, unit = "d")
    milk = milk.sort_values(by = "week_ref").reset_index(drop=1)
    
    fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize=(15,8))    
    sns.boxplot(data=milk, x="week_ref", y="dmy", 
                 fliersize=0, whis = 0.8)
    ax.set_ylim([0,90])
    # convert all xtick labels to selected format from ms timestamp
    xticks = ax.xaxis.get_ticklabels()
    xticks = ax.get_xticks()
    n_weeks = [1,11,22,33,44]
    prev_year = milk["date"].dt.year.min()
    xlabels = []
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        print(n,label)
        label.set_rotation(90)
        xlabels.append(label.get_text()[0:10])
        weekofyear = date(int(label.get_text()[0:4]),
                          int(label.get_text()[5:7]),
                          int(label.get_text()[8:10])).isocalendar().week
        currentyear = date(int(label.get_text()[0:4]),
                          int(label.get_text()[5:7]),
                          int(label.get_text()[8:10])).isocalendar().year
        if weekofyear not in n_weeks:
            label.set_visible(False)
        else:
            print(label.get_text()[0:10])
        if currentyear != prev_year:
            ax.plot([n,n], [0,1500], linestyle = 'dotted',color = "lightseagreen",linewidth = 2)
            print(currentyear)
            prev_year = currentyear
        # if week % every_nth != 0:
        #     label.set_visible(False)      
    ax.set_xticklabels(xlabels)
    ax.set_title("Daily milk production farm " + str(farm))
    ax.set_xlabel("date")
    ax.set_ylabel("milk")
    del n,label

    plot_weather = pd.merge(weekweather,milk["week_ref"].drop_duplicates(),
                            on = "week_ref", how = "inner")
    
    # add to plot : mean THI and max daily temperature
    ax2 = ax.twinx()
    ax2.plot(plot_weather.index.values, plot_weather["temp_max"],
            linewidth = 1.5, color = "r") 
    ax2.plot(plot_weather.index.values, plot_weather["thi_mean"],
            linewidth = 1.5, color = "blue")  
    ax2.set_ylim([-160,85])
    ax2.set_ylabel("THI and max daily temperature")  
    ax2.grid(False)      
    
    plt.savefig(os.path.join(path,"results","milk","milk_thi_farm_" + str(farm) + ".tif"))    
    del ax, ax2, currentyear, fig, n_weeks, prev_year
    
    # ---------------- summarize and plot stats activity ------------------
    def q10(x):
        return x.quantile(0.1)
    def q90(x):
        return x.quantile(0.9)
    sumstat = (milk[["date","dmy"]].groupby(by = "date").agg({"dmy" : ["count","mean","std","median",q10,q90]})).reset_index()
    sumstat.columns = sumstat.columns.droplevel()
    sumstat.columns = ["date","count","mean",
                                      "std","median",
                                      "q10","q90"]
    sumstat["date"] = pd.to_datetime(sumstat["date"], format = "%Y-%m-%d")
    
    # plot stats ----  %matplotlib qt
    fig,ax = plt.subplots(nrows = 2, ncols = 1, figsize=(15,10))
    
    ax[0].fill_between(sumstat["date"],sumstat["q10"],sumstat["q90"], 
                     color = "palevioletred", alpha = 0.5)
    ax[0].plot(sumstat["date"],sumstat["median"], linewidth = 2, color = "crimson")
    
    ax[1].plot(sumstat["date"],sumstat["count"], linewidth = 2, color = "darkmagenta")
    # ax[1].grid(False)     
    ax[1].set_ylim([0,sumstat["count"].max()+20])
    ax[1].set_xlabel("date")
    ax[0].set_xlabel("date")
    ax[1].set_ylabel("count - number of animals")
    ax[0].set_title("farm = "+str(farm)+ ", summary stats of milk production data")
    ax[1].set_title("number of measurements, i.e., cows per day")
    ax[0].set_ylabel("median  - IQR of milk production data")
    
    # save plot
    plt.savefig(os.path.join(path,"results","milk","milk_stats_" + str(farm) + ".tif"))
    plt.close()
    
    del sumstat, ax, fig, xlabels, xticks
    
    #-------------------------- visualisation per dim/parity ------------------
    milk["pargroup"] = (
        (pd.concat([milk["parity"],pd.DataFrame(3*np.ones((len(milk),1)))], axis = 1))
        .min(axis = 1)
        )
    sumstat = (
        (milk[["pargroup","dim","dmy"]].groupby(by = ["pargroup","dim"])
         .agg({"dmy" : ["count","mean","std"]})).reset_index()
        )
    sumstat.columns = sumstat.columns.droplevel()
    sumstat.columns = ["pargroup","dim","count","mean","std"]

    fig,ax = plt.subplots(3,1,figsize = (16,11))
    for parity in sumstat["pargroup"].drop_duplicates().astype(int):
        print(parity)
        subset = sumstat.loc[sumstat["pargroup"]==parity,:]
        ax[parity-1].fill_between(subset["dim"],
                                  subset["mean"]-2*subset["std"],
                                  subset["mean"]+2*subset["std"],
                                  color = "palevioletred", alpha = 0.5)
        ax[parity-1].plot(subset["dim"],
                          subset["mean"],
                          linewidth = 2, color = "crimson")
        ax[parity-1].set_xlim(0,subset["dim"].max())
        ax[parity-1].set_ylim(0,(subset["mean"]+2.2*subset["std"]).max())
        if parity == 1:
            ax[parity-1].set_title("farm  " + str(farm) + ", milk production, parity = " + \
                                   str(round(parity)), fontsize = 14)
        else:
            ax[parity-1].set_title("parity = " + str(parity))
        if parity == 3:
            ax[parity-1].set_xlabel("dim [d]")
        ax[parity-1].set_ylabel("daily milk yield, mean+2*std, [kg]")
        ax2 = ax[parity-1].twinx()
        ax2.plot(subset["dim"],
                 subset["count"],
                 linewidth = 2, color = "blue")
        ax2.grid(False)
        ax2.set_ylabel("number of animals")
    plt.savefig(os.path.join(path,"results","milk","milk_stats_dim_" + str(farm) + ".tif"))
    plt.close()
