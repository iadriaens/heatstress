# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:38:33 2023

@author: u0084712
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
                    "projects","ugent","heatstress","data")

# farm selected
farms = [39]# [30,31,33,34,35,38,39,40,43,44,45,46,48] 


#%% read and preprocess data

for farm in farms:
    # read data
    data = pd.read_csv(os.path.join(path_data, "scc_" + str(farm) + ".txt"),
                       index_col=0)
    data["measured_on"] = pd.to_datetime(data["measured_on"],format='%Y-%m-%d')
    data["last_insemination_date"] = pd.to_datetime(data["last_insemination_date"],
                                                    format='%Y-%m-%d')
    data = data.sort_values(by = ["animal_id","measured_on"]).reset_index()
    print(data["measured_on"].min())
    data["lnscc"] = np.log(data["scc"])
    
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
    
    
    #--------------------------- visualisations -------------------------------
    data = data.sort_values(by="measured_on")
    # %matplotlib qt
    fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize=(15,8))    
    sns.boxplot(data=data, x="measured_on", y="scc", 
                 fliersize=0.1, whis = 0.8)
    plt.yscale("log")
    
    # convert all xtick labels to selected format from ms timestamp
    xticks = ax.xaxis.get_ticklabels()
    xticks = ax.get_xticks()
    n_weeks = [1,11,22,33,44]
    prev_year = data["measured_on"].dt.year.min()
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
            # do nothing // label.set_visible(False)
            print("week of year")
        else:
            print(label.get_text()[0:10])
        if currentyear != prev_year:
            ax.plot([n,n], [0.1,data["scc"].max()], linestyle = 'dotted',color = "lightseagreen",linewidth = 2)
            print(currentyear)
            prev_year = currentyear
        # if week % every_nth != 0:
        #     label.set_visible(False)      
    ax.set_xticklabels(xlabels)
    ax.set_title("SCC data, farm " + str(farm))
    ax.set_xlabel("date")
    ax.set_ylabel("scc")
    ax.set_ylim(0.1,data["scc"].max())
    del n,label
    
    # find unique dates of SCC and weather on that date
    dates = pd.DataFrame(data["measured_on"].drop_duplicates().reset_index(drop=1))
    dates["thi"] = np.nan

    for d in dates["measured_on"]:
        weather["diff"] = (weather["date"] - d).dt.days
        dates.loc[dates["measured_on"] == d,"thi"] = \
            (weather.loc[(weather["diff"]>=-5) & 
                         (weather["diff"]<=0),"thi"]).mean()
        
    
    # plot THI
    ax2 = ax.twinx()
    ax2.plot(xticks, dates["thi"],
            linewidth = 1.5, color = "blue") 
    ax2.set_ylim([-160,85])
    ax2.set_ylabel("avg THI 5 days before sampling")  
    ax2.grid(False)  
    
    plt.savefig(os.path.join(path,"results","scc","scc_thi_farm_" + str(farm) + ".tif"))    
    plt.close()
    
    # add thi to scc
    scc = pd.merge(data,dates, how = "outer",on = "measured_on")
    scc.to_csv(os.path.join(path,"results","scc","scc_preprocessed_" + str(farm) + ".txt"))

