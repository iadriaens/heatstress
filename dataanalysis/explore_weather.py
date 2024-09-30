# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 09:00:50 2024

@author: u0084712

-------------------------------------------------------------------------------

Goal: further explore and quantify weather features:
        - distributions
        - correlations
        
Verify how other domains "define" the weather for modelling.


"""

import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataanalysis")
os.chdir(path)


#%% import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import geopandas as gpd
# from shapely.geometry import Point
import seaborn as sns
import matplotlib

# %matplotlib qt


#%% paths and constants

path_data = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataanalysis","data")
path_results = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataanalysis","results","thi")

# farm selected
farms = [30,31,33,34,35,38,39,40,43,44,45,46,48] 


#%% load data 

data = pd.DataFrame([])
for farm in farms:
    
    df = pd.read_csv(os.path.join(path_data,"weatherfeatures_" + str(farm)+".txt"), index_col=0)
    data = (pd.concat([data,df])).reset_index(drop=1)

del df, farm


#%% data exploration: distributions and general visualisations ALL together - data

# create dataset
df = data.iloc[:,1:]
df = df.groupby(by=["date"]).mean()
df = df.drop(columns = ["no_meas"])

# variable for categories sep, thi, year, rec
# wf = {"sep" : ["temp_min", "temp_max", "rh_min","rh_max","temp_hrs_low",
#                            "temp_hrs_high","temp_hrs","temp_hrs_mod","temp_difference"],
#       "thi" : ["thi_avg","thi_max","thi_hrs_high","thi_hrs_mild","thi_hrs_mod",
#                "thi_hrs_sev","perc_thi_5d_prior","perc_thi_2d_prior","thi_high",
#                "no_days_highTHI","no_days_high_THI"],
#       "year":["no_days_year_prior"],
#       "rec" : ["hrs_rec_succ_prev","hrs_rec_succ_next"]}

# add variable combining all years per day
dfy = (df.groupby(by = "day").mean()).reset_index()
dfy = dfy.drop(columns = ["year","month","week"])


# #TODO
# # change / recaluclate for variables 0/1
# df["thi_high"] = (df["thi_avg"]>=68).astype(int)
# df["no_days_highTHI"] = pd.DataFrame(df["thi_high"]).eq(0).cumsum().groupby('thi_high').cumcount()
# data["no_days_year_prior"] = data[["farm_id","year","thi_high"]].groupby(by = ["farm_id","year"]).cumsum()

#%% save-close var
scvar = 0

#%% data exploration: distributions and general visualisations ALL together - figures

# plot histograms / boxplots all and per year
sns.set_style("whitegrid")

#------------------------------------------------------------------------------
# temp_min and temp_max --> necessary to put it in long instead of wide format
dfy["pos"] = 0
dfyw = (
        pd.concat([dfy[["pos","temp_min"]].rename(columns={"temp_min" : "temp"}),
                  (pd.concat([dfy["pos"]+1,dfy["temp_max"]],axis=1)).rename(columns={"temp_max" : "temp"})])
        ).reset_index(drop=1)


# plot over all years together
_, ax = plt.subplots(1,2, figsize = (18,5), width_ratios=[2,6]) 
sns.swarmplot(data=dfyw,x="pos", y="temp",ax=ax[0], size = 2,hue = "pos", 
              palette = sns.set_palette(["#0000CC","#D80D0D"]),legend=False)
sns.boxplot(data=dfyw,x="pos",y="temp",ax=ax[0], saturation = 1, 
            palette =sns.set_palette(["#6666FF","#FF9999"]))  # palette red blue
ax[0].set_xticklabels(["min T°","max T°"])
ax[0].set_title("average minimum and maximum temperature\n -all years together")
ax[0].set_ylabel("temperature [C°]")
ax[0].set_xlabel("")

# per year plots = df in long format
df["pos"] = 0
dfw = (
        pd.concat([df[["pos","temp_min","year"]].rename(columns={"temp_min" : "temp"}),
                  (pd.concat([df["pos"]+1,df[["temp_max","year"]]],axis=1)).rename(columns={"temp_max" : "temp"})])
        ).reset_index(drop=1)
dfw["year"] = dfw["year"].astype(int)


# plot
sns.boxplot(data=dfw,x="year",y="temp",ax=ax[1], saturation = 1, hue="pos",
            palette =sns.set_palette(["#6666FF","#FF9999"]),
            flierprops={"marker":"o", "markersize":2,"color" :"k"},
            boxprops = {"edgecolor":"black"},
            medianprops={"color": "red", "linewidth": 1})  # palette red blue
ax[1].set_title("average minimum and maximum temperature\n -per year")
ax[1].set_ylabel("temperature [C°]")
ax[1].legend(["min T°","_","_","_","_","_","_","_","max T°"],loc="best")
ax[0].set_ylim(ax[1].get_ylim())
# save and close
if scvar==1:
    plt.savefig(os.path.join(path_results,"temperature_all_farms_boxplots.tif"))    
    plt.close()


#------------------------------------------------------------------------------
# rel humidity
# rh_min and rh_max --> necessary to put it in long instead of wide format
dfy["pos"] = 0
dfyw = (
        pd.concat([dfy[["pos","rh_min"]].rename(columns={"rh_min" : "RH"}),
                  (pd.concat([dfy["pos"]+1,dfy["rh_max"]],axis=1)).rename(columns={"rh_max" : "RH"})])
        ).reset_index(drop=1)

# per year plots = df in long format
df["pos"] = 0
dfw = (
        pd.concat([df[["pos","rh_min","year"]].rename(columns={"rh_min" : "RH"}),
                  (pd.concat([df["pos"]+1,df[["rh_max","year"]]],axis=1)).rename(columns={"rh_max" : "RH"})])
        ).reset_index(drop=1)
dfw["year"] = dfw["year"].astype(int)

# plot over all years together
fig, ax = plt.subplots(1,2, figsize = (18,5), width_ratios=[2,6]) 
sns.swarmplot(data=dfyw,x="pos", y="RH",ax=ax[0], size = 2,hue = "pos", 
              palette = sns.set_palette(["#FF8000","#009999"]),legend=False)
sns.boxplot(data=dfyw,x="pos",y="RH",ax=ax[0], saturation = 1, 
            palette =sns.set_palette(["#FFB266","#CCFFFF"]))  # palette red blue
ax[0].set_xticklabels(["min RH","max RH"])
ax[0].set_title("average minimum and maximum RH\n -all years together")
ax[0].set_ylabel("RH [%]")
ax[0].set_xlabel("")

# plot
sns.boxplot(data=dfw,x="year",y="RH",ax=ax[1], saturation = 1, hue="pos",
            palette =sns.set_palette(["#FFB266","#00CCCC"]),
            flierprops={"marker":"o", "markersize":2,"color" :"k"},
            boxprops = {"edgecolor":"black"},
            medianprops={"color": "red", "linewidth": 1})  # palette red blue
ax[1].set_title("average minimum and maximum RH\n -per year")
ax[1].set_ylabel("RH [%]")
ax[1].legend(["min RH","_","_","_","_","_","_","_","max RH"],loc="lower left")
ax[0].set_ylim(ax[1].get_ylim())
# save and close
if scvar==1:
    plt.savefig(os.path.join(path_results,"relhum_all_farms_boxplots.tif"))    
    plt.close()
        
#------------------------------------------------------------------------------
# thi variables

# thi avg and thi max --> necessary to put it in long instead of wide format
dfy["pos"] = 0
dfyw = (
        pd.concat([dfy[["pos","thi_avg"]].rename(columns={"thi_avg" : "thi"}),
                  (pd.concat([dfy["pos"]+1,dfy["thi_max"]],axis=1)).rename(columns={"thi_max" : "thi"})])
        ).reset_index(drop=1)

# per year plots = df in long format
df["pos"] = 0
dfw = (
        pd.concat([df[["pos","thi_avg","year"]].rename(columns={"thi_avg" : "thi"}),
                  (pd.concat([df["pos"]+1,df[["thi_max","year"]]],axis=1)).rename(columns={"thi_max" : "thi"})])
        ).reset_index(drop=1)
dfw["year"] = dfw["year"].astype(int)

# plot over all years together
_,ax = plt.subplots(1,2, figsize = (18,5), width_ratios=[2,6]) 
sns.swarmplot(data=dfyw,x="pos", y="thi",ax=ax[0], size = 2,hue = "pos", 
              palette = sns.set_palette(["#003366","#99004C"]),legend=False)
sns.boxplot(data=dfyw,x="pos",y="thi",ax=ax[0], saturation = 1, 
            palette =sns.set_palette(["#0066CC","#AC317B"]))  # palette red blue
ax[0].set_xticklabels(["avg THI","max THI"])
ax[0].set_title("average and maximum THI\n -all years together")
ax[0].set_ylabel("THI")
ax[0].set_xlabel("")

# plot
sns.boxplot(data=dfw,x="year",y="thi",ax=ax[1], saturation = 1, hue="pos",
            palette =sns.set_palette(["#0066CC","#AC317B"]),
            flierprops={"marker":"o", "markersize":2,"color" :"k"},
            boxprops = {"edgecolor":"black"},
            medianprops={"color": "red", "linewidth": 1})  # palette red blue
ax[1].set_title("average and maximum THI\n -per year")
ax[1].set_ylabel("THI")
ax[1].legend(["avg THI","_","_","_","_","_","_","_","max THI"],loc="lower left")
ax[0].set_ylim(ax[1].get_ylim())
ax[1].axhline(y=68, color = "r",lw=3)
# save and close
if scvar==1:
    plt.savefig(os.path.join(path_results,"thi_all_farms_boxplots.tif"))    
    plt.close()

#♥-----------------------------------------------------------------------------
# percentage of hours THI was above 68 in the past 5 days

_,ax = plt.subplots(len(df.year.drop_duplicates())+1,1, figsize = (10,25), 
                    height_ratios=[2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    sharex=True) 
plt.subplots_adjust(hspace = 0)
sns.lineplot(data = dfy,y="perc_thi_5d_prior",x="day",ax=ax[0], lw = 2, color='r')
sns.lineplot(y=[0,0],x=[0,366],ax=ax[0], lw = 0.5, color='k')
ax[0].set_xlim(-1,366.5)
ax[0].set_ylim(-1,75)
ax[0].set_xticks([0,60,120,180,240,300])
ax[0].set_xticklabels(["jan","mar","may","jul","sep","nov"])
ax[0].set_ylabel("all years",rotation="horizontal")
ax[0].set_ylim([-1,75])
ax[0].set_yticks([0,50,75])
T=0
for y in df.year.drop_duplicates():
    T=T+1
    if y%2 == 1:
        ax[T].set_yticks([])
        ax[T].set_yticklabels([])
        ax[T].axvline(x=366,color = '#99004C',lw=2)
        ax[T] = ax[T].twinx()
    else:
        ax[T].axvline(x=0,color = '#99004C')

    sns.lineplot(data = df.loc[df["year"]==y],y="perc_thi_5d_prior",x="day",ax=ax[T], lw = 1, color='b')
    ax[T].set_ylabel(round(y), rotation='horizontal')
    ax[T].set_ylim([-1,75])
    ax[T].set_yticks([0, 50,75])
    
ax[T].set_xlabel("day of the year")    
ax[0].set_title('percentage of hours THI was >68 in preceding 5 days')
# save and close
if scvar==1:
    plt.savefig(os.path.join(path_results,"perc_thi_high_5dprev_all_farms.tif"))    
    plt.close()    

#♥-----------------------------------------------------------------------------
# percentage of hours THI was above 68 in the past 2 days

_,ax = plt.subplots(len(df.year.drop_duplicates())+1,1, figsize = (10,25), 
                    height_ratios=[2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    sharex=True) 
plt.subplots_adjust(hspace = 0)
sns.lineplot(data = dfy,y="perc_thi_2d_prior",x="day",ax=ax[0], lw = 2, color='r')
sns.lineplot(y=[0,0],x=[0,366],ax=ax[0], lw = 0.5, color='k')
ax[0].set_xlim(-1,366.5)
ax[0].set_ylim(-1,75)
ax[0].set_xticks([0,60,120,180,240,300])
ax[0].set_xticklabels(["jan","mar","may","jul","sep","nov"])
ax[0].set_ylabel("all years",rotation="horizontal")
ax[0].set_ylim([-1,100])
ax[0].set_yticks([0,50,100])
T=0
for y in df.year.drop_duplicates():
    T=T+1
    if y%2 == 1:
        ax[T].set_yticks([])
        ax[T].set_yticklabels([])
        ax[T].axvline(x=366,color = '#99004C',lw=2)
        ax[T] = ax[T].twinx()
    else:
        ax[T].axvline(x=0,color = '#99004C')

    sns.lineplot(data = df.loc[df["year"]==y],y="perc_thi_2d_prior",x="day",ax=ax[T], lw = 1, color='#00994C')
    ax[T].set_ylabel(round(y), rotation='horizontal')
    ax[T].set_ylim([-1,100])
    ax[T].set_yticks([0, 50,100])
    
ax[T].set_xlabel("day of the year")    
ax[0].set_title('percentage of hours THI was >68 in preceding 2 days')
# save and close
if scvar==1:
    plt.savefig(os.path.join(path_results,"perc_thi_high_2dprev_all_farms.tif"))    
    plt.close()

#♥-----------------------------------------------------------------------------
# number of hours recovery (<18°) bewteen noon and noon previous and current day 

_,ax = plt.subplots(len(df.year.drop_duplicates())+1,1, figsize = (10,25), 
                    height_ratios=[2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    sharex=True) 
plt.subplots_adjust(hspace = 0)
sns.lineplot(data = dfy,y="hrs_rec_succ_prev",x="day",ax=ax[0], lw = 2, color='r')
sns.lineplot(y=[0,0],x=[0,366],ax=ax[0], lw = 0.5, color='k')
ax[0].set_xlim(-1,366.5)
ax[0].set_ylim(-1,25)
ax[0].set_xticks([0,60,120,180,240,300])
ax[0].set_xticklabels(["jan","mar","may","jul","sep","nov"])
ax[0].set_ylabel("all years",rotation="horizontal")
ax[0].set_ylim([-1,25])
ax[0].set_yticks([0,12,24])
T=0
for y in df.year.drop_duplicates():
    T=T+1
    if y%2 == 1:
        ax[T].set_yticks([])
        ax[T].set_yticklabels([])
        ax[T].axvline(x=366,color = '#99004C',lw=2)
        ax[T] = ax[T].twinx()
    else:
        ax[T].axvline(x=0,color = '#99004C')

    sns.lineplot(data = df.loc[df["year"]==y],y="hrs_rec_succ_prev",x="day",ax=ax[T], lw = 1, color='#990099')
    ax[T].set_ylabel(round(y), rotation='horizontal')
    ax[T].set_ylim([-1,25])
    ax[T].set_yticks([0, 12,24])
    
ax[T].set_xlabel("day of the year")    
ax[0].set_title('number of hours recovery noon previous - noon current day')
# save and close
if scvar==1:
    plt.savefig(os.path.join(path_results,"hours_recovery_prev_curr_all_farms.tif"))    
    plt.close()

#------------------------------------------------------------------------------
# temp min max over the year all years together
cmap = matplotlib.cm.get_cmap('brg')

_,ax = plt.subplots(1,1,figsize = (12,4))
for d in dfy["day"]:
    c = ((dfy.loc[dfy["day"]==d,"temp_min"]+dfy.loc[dfy["day"]==d,"temp_max"])/2-\
        dfy["temp_min"].min())/(dfy["temp_max"].max()-dfy["temp_min"].min())*0.6
    ax.plot([d,d],[dfy.loc[dfy["day"]==d,"temp_min"],dfy.loc[dfy["day"]==d,"temp_max"]],
                   color = cmap(c), lw = 1.5)
ax.set_xticks([0,31,60,91,121,152,183,213,244,275,305,336,366])
ax.set_xticklabels(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec",""])
ax.set_title("temperature range over the year")
ax.set_ylabel("T [C°]")
ax.set_xlim(-1,366)
# save and close
if scvar==1:
    plt.savefig(os.path.join(path_results,"temperature_range_allyears_all_farms.tif"))    
    plt.close()

# temp min max over the years per year

_,ax = plt.subplots(len(df.year.drop_duplicates())+1,1, figsize = (12,35), 
                    height_ratios=[2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    sharex=True) 
plt.subplots_adjust(hspace = 0)
for d in dfy["day"]:
    c = ((dfy.loc[dfy["day"]==d,"temp_min"]+dfy.loc[dfy["day"]==d,"temp_max"])/2-\
        dfy["temp_min"].min())/(dfy["temp_max"].max()-dfy["temp_min"].min())*0.6
    ax[0].plot([d,d],[dfy.loc[dfy["day"]==d,"temp_min"],dfy.loc[dfy["day"]==d,"temp_max"]],
                   color = cmap(c), lw = 1.5)
sns.lineplot(y=[0,0],x=[0,366],ax=ax[0], lw = 0.5, color='k')
ax[0].set_xlim(-1,366.5)
ax[0].set_ylim(-1,25)
ax[0].set_xticks([0,60,120,180,240,300])
ax[0].set_xticklabels(["jan","mar","may","jul","sep","nov"])
ax[0].set_ylabel("all years",rotation="horizontal")
ax[0].set_ylim([-1,25])
ax[0].set_yticks([0,12,24])
T=0
for y in df.year.drop_duplicates():
    T=T+1
    if y%2 == 1:
        ax[T].set_yticks([])
        ax[T].set_yticklabels([])
        ax[T].axvline(x=366,color = '#99004C',lw=2)
        ax[T] = ax[T].twinx()
    else:
        ax[T].axvline(x=0,color = '#99004C')
    
    for d in df.loc[df["year"]==y,"day"]:
        c = ((df.loc[(df["day"]==d)&(df["year"]==y),"temp_min"]+\
              df.loc[(df["day"]==d)&(df["year"]==y),"temp_max"])/2-\
              df.loc[df["year"]==y,"temp_min"].min())/\
              (df.loc[df["year"]==y,"temp_max"].max()-\
              df.loc[df["year"]==y,"temp_min"].min())*0.6
        ax[T].plot([d,d],[df.loc[(df["day"]==d)&(df["year"]==y),"temp_min"].values,
                          df.loc[(df["day"]==d)&(df["year"]==y),"temp_max"].values],
                    lw = 1, color=cmap(c))
    ax[T].set_ylabel(round(y), rotation='horizontal')
    # ax[T].set_ylim([-1,25])
    ax[T].set_yticks([0, 15,30])
    
ax[T].set_xlabel("day of the year")    
ax[0].set_title('min and max temperature over the year')
# save and close
if scvar==1:
    plt.savefig(os.path.join(path_results,"temperature_range_all_farms.tif"))    
    plt.close()

#------------------------------------------------------------------------------
# RH min max over the year all years together
cmap = matplotlib.cm.get_cmap('cool')

_,ax = plt.subplots(1,1,figsize = (12,4))
for d in dfy["day"]:
    c = 1-(dfy.loc[dfy["day"]==d,"rh_min"])/100
    ax.plot([d,d],[dfy.loc[dfy["day"]==d,"rh_min"],dfy.loc[dfy["day"]==d,"rh_max"]],
                   color = cmap(c), lw = 1.5)
ax.set_xticks([0,31,60,91,121,152,183,213,244,275,305,336,366])
ax.set_xticklabels(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec",""])
ax.set_title("relative humidity range over the year")
ax.set_ylabel("RH [%]")
ax.set_xlim(-1,366)
# save and close
if scvar==1:
    plt.savefig(os.path.join(path_results,"relhum_range_allyears_all_farms.tif"))    
    plt.close()

# RH min max over the years per year

_,ax = plt.subplots(len(df.year.drop_duplicates())+1,1, figsize = (12,35), 
                    height_ratios=[2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    sharex=True) 
plt.subplots_adjust(hspace = 0)
for d in dfy["day"]:
    c = 1-(dfy.loc[(dfy["day"]==d),"rh_min"])/100
    ax[0].plot([d,d],[dfy.loc[dfy["day"]==d,"rh_min"],dfy.loc[dfy["day"]==d,"rh_max"]],
                   color = cmap(c), lw = 1.5)
sns.lineplot(y=[0,0],x=[0,366],ax=ax[0], lw = 0.5, color='k')
ax[0].set_xlim(-1,366.5)
ax[0].set_ylim(-1,101)
ax[0].set_xticks([0,60,120,180,240,300])
ax[0].set_xticklabels(["jan","mar","may","jul","sep","nov"])
ax[0].set_ylabel("all years",rotation="horizontal")
ax[0].set_yticks([33,66,100])
T=0
for y in df.year.drop_duplicates():
    T=T+1
    if y%2 == 1:
        ax[T].set_yticks([])
        ax[T].set_yticklabels([])
        ax[T].axvline(x=366,color = '#99004C',lw=2)
        ax[T] = ax[T].twinx()
    else:
        ax[T].axvline(x=0,color = '#99004C')
    
    for d in df.loc[df["year"]==y,"day"]:
        c = 1-(df.loc[(df["day"]==d)&(df["year"] == y),"rh_min"])/100
        ax[T].plot([d,d],[df.loc[(df["day"]==d)&(df["year"]==y),"rh_min"].values,
                          df.loc[(df["day"]==d)&(df["year"]==y),"rh_max"].values],
                    lw = 1, color=cmap(c))
    ax[T].set_ylabel(round(y), rotation='horizontal')
    # ax[T].set_ylim([-1,25])
    ax[T].set_yticks([33,66,100])
    
ax[T].set_xlabel("day of the year")    
ax[0].set_title('min and max RH over the year')
# save and close
if scvar==1:
    plt.savefig(os.path.join(path_results,"relhum_range_all_farms.tif"))    
    plt.close()

#-----------------------------------------------------------------------------
# THI avg max over the year all years together
cmap = matplotlib.cm.get_cmap('afmhot')

_,ax = plt.subplots(1,1,figsize = (12,4))
for d in dfy["day"]:
    c = 0.6*((dfy.loc[dfy["day"]==d,"thi_avg"])-\
        dfy["thi_avg"].min())/(dfy["thi_max"].max()-dfy["thi_avg"].min())
    ax.plot([d,d],[dfy.loc[dfy["day"]==d,"thi_avg"],dfy.loc[dfy["day"]==d,"thi_max"]],
                   color = cmap(c), lw = 1.5)
ax.set_xticks([0,31,60,91,121,152,183,213,244,275,305,336,366])
ax.set_xticklabels(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec",""])
ax.set_title("thi avg-max range over the year")
ax.set_ylabel("THI")
ax.set_xlim(-1,366)
# save and close
if scvar==1:
    plt.savefig(os.path.join(path_results,"thiavgmax_range_allyears_all_farms.tif"))    
    plt.close()

# RH min max over the years per year

_,ax = plt.subplots(len(df.year.drop_duplicates())+1,1, figsize = (12,35), 
                    height_ratios=[2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    sharex=True) 
plt.subplots_adjust(hspace = 0)
for d in dfy["day"]:
    c = (((dfy.loc[dfy["day"]==d,"thi_avg"])-\
        dfy["thi_avg"].min())/(dfy["thi_max"].max()-dfy["thi_avg"].min()))*0.6
    ax[0].plot([d,d],[dfy.loc[dfy["day"]==d,"thi_avg"],dfy.loc[dfy["day"]==d,"thi_max"]],
                   color = cmap(c), lw = 1.5)
# sns.lineplot(y=[0,0],x=[0,366],ax=ax[0], lw = 0.5, color='k')
ax[0].set_xlim(-1,366.5)
ax[0].set_ylim(35,85)
ax[0].set_xticks([0,60,120,180,240,300])
ax[0].set_xticklabels(["jan","mar","may","jul","sep","nov"])
ax[0].set_ylabel("all years",rotation="horizontal")
ax[0].set_yticks([68])
T=0
for y in df.year.drop_duplicates():
    T=T+1
    if y%2 == 1:
        ax[T].set_yticks([])
        ax[T].set_yticklabels([])
        ax[T].axvline(x=366,color = '#99004C',lw=2)
        ax[T] = ax[T].twinx()
    else:
        ax[T].axvline(x=0,color = '#99004C')
    
    for d in df.loc[df["year"]==y,"day"]:
        c = ((df.loc[(df["day"]==d) & (df["year"] == y),"thi_avg"]-\
            df["thi_avg"].min())/(df["thi_max"].max()-df["thi_avg"].min()))*0.6
        ax[T].plot([d,d],[df.loc[(df["day"]==d)&(df["year"]==y),"thi_avg"].values,
                          df.loc[(df["day"]==d)&(df["year"]==y),"thi_max"].values],
                    lw = 1, color=cmap(c))
    ax[T].set_ylabel(round(y), rotation='horizontal')
    # ax[T].set_ylim([-1,25])
    ax[T].set_yticks([68])
    
ax[T].set_xlabel("day of the year")    
ax[0].set_title('avg and max THI over the year')
# save and close
if scvar==1:
    plt.savefig(os.path.join(path_results,"thiavgmax_range_all_farms.tif"))    
    plt.close()






#%% correlations

wfnames = ["min T°","max T°","min RH","max RH","avg thi","max thi", 
           "no. of h thi > 68","no. of h thi [68,72[", "no. of h thi [72,80[",
           "no. of h thi >=80", "no. of h temp >=25", "no. of h temp <=18",
           "% of h thi > 68 in past 5d","% of h thi >68 in past 2d",
           "recovery (<18°) h\nin prev. 12-12:00","recovery (<18°) h\nin next 12-12:00",
           "is thi high","no. of successive d high thi\n= duration insult",
           "cumulative d with high thi","no. of days in 5d prior\nwith high thi"]
sns.set(font_scale=0.8)
sns.set_style("darkgrid")
_,ax = plt.subplots(1,1,figsize = (15,15))
sns.heatmap(data=round(dfy.iloc[:,1:-1].corr(),2),ax=ax, 
            annot=True,lw=0.2,cmap="PiYG",
            mask = np.triu(np.ones_like(dfy.iloc[:,1:-1].corr(), dtype=bool))==True,
            xticklabels=wfnames,yticklabels=wfnames)
# save and close
if scvar==1:
    plt.savefig(os.path.join(path_results,"correlations_allvar.tif"))    
    plt.close()


_,ax = plt.subplots(1,1,figsize = (12,10))
sns.heatmap(data=round(dfy.iloc[:,1:-1].corr(),2),ax=ax, 
            annot=True,lw=0.2,linecolor = "k",cmap="PiYG",
            mask = ((abs(round(dfy.iloc[:,1:-1].corr(),2)) < 0.8) | \
                    (round(dfy.iloc[:,1:-1].corr(),2)==1) | \
                    np.triu(np.ones_like(dfy.iloc[:,1:-1].corr(), dtype=bool))==True),
            xticklabels=wfnames,yticklabels=wfnames)
# save and close
if scvar==1:
    plt.savefig(os.path.join(path_results,"correlations_highonly.tif"))    
    plt.close()


for y in df["year"].drop_duplicates():
    _,ax = plt.subplots(1,1,figsize = (12,12))
    sns.heatmap(data=round(df.loc[df["year"]==y].iloc[:,1:-1].corr(),2),ax=ax, 
                annot=True,lw=0.2,cmap="PiYG",
                mask = ((abs(round(df.loc[df["year"]==y].iloc[:,1:-1].corr(),2)) < 0.8) | \
                        (round(df.loc[df["year"]==y].iloc[:,1:-1].corr(),2)==1) | \
                        np.triu(np.ones_like(df.loc[df["year"]==y].iloc[:,1:-1].corr(), dtype=bool))==True),
                xticklabels=wfnames,yticklabels=wfnames)
    ax.set_title(str(round(y)))
    # save and close
    if scvar==1:
        plt.savefig(os.path.join(path_results,"correlations_allvar_"+str(round(y))+".tif"))    
        plt.close()
    
#------------------------------------------------------------------------------
# scatterplots // pairplots
# add season to dfy
dfy["season"] = "winter"
dfy.loc[dfy["day"] > 59,"season"] = "spring"
dfy.loc[dfy["day"] > 151,"season"] = "summer"
dfy.loc[dfy["day"] > 243,"season"] = "autumn"
dfy.loc[dfy["day"] > 334,"season"] = "winter"
sns.set(font_scale=1.2)


# temp and relative humidity
# _,ax = plt.subplots(1,1,figsize = (18,15))
g = sns.pairplot(dfy.iloc[:,[1,2,3,4,5,6,-1]], 
                 diag_kind="kde",
                 hue = "season",
                 palette = {"winter" : "#3333FF","spring" : "#00CC66","summer" : "#CC0000","autumn":"#F97306"},
                 corner = False,
                 # x_vars  = ["temp_min","temp_max","rh_min","rh_max","thi_avg","thi_max"],
                 # y_vars = ["temp_min","temp_max","rh_min","rh_max","thi_avg","thi_max"]
                 )
labels = {"temp_min":"min T°",
          "temp_max":"max T°",
          "rh_min":"min RH",
          "rh_max":"max RH",
          "thi_avg":"avg THI",
          "thi_max":"max THI"}
# set labels
for i in range(6):
    for j in range(6):
        xlabel = g.axes[i][j].get_xlabel()
        ylabel = g.axes[i][j].get_ylabel()
        if xlabel in labels.keys():
            g.axes[i][j].set_xlabel(labels[xlabel])
        if ylabel in labels.keys():
            g.axes[i][j].set_ylabel(labels[ylabel])

# set lines and white triangle
for i in range(6):
    for j in range(6):
        x = g.axes[i][j].get_xlim()
        y = g.axes[i][j].get_ylim()
        if (i==j)or(i>j):
            if (i==4) or (i==5):
                g.axes[i][j].axhline(y=68,ls="--",color="r")
                g.axes[i][j].fill_between(x,68,y[1],color="r",alpha=0.3)
            if (j==4) or (j==5):
                g.axes[i][j].axvline(x=68,ls="--",color="r")
                g.axes[i][j].fill_betweenx(y,68,x[1],color="r",alpha=0.3)
        if j>i:
            x = g.axes[i][j].get_xlim()
            y = g.axes[i][j].get_ylim()
            g.axes[i][j].fill_between(x,y[0],y[1],color = 'w')
        g.axes[i][j].set_xlim(x)
        g.axes[i][j].set_ylim(y)
if scvar==1:
    plt.savefig(os.path.join(path_results,"scatter_temp_rh_thi.tif"))    
    plt.close()

#------------------------------------------------------------------------------


#%% table with per year description of number of days with weather

dfc = df[["thi_high","thi_avg","thi_max","thi_hrs_high","thi_hrs_mild","thi_hrs_mod",
          "thi_hrs_sev","temp_hrs_high","perc_thi_5d_prior","perc_thi_2d_prior",
          "hrs_rec_succ_prev","no_days_highTHI","no_days_5d_prior","year","day"]]
dfc = dfc.groupby(by="year").sum()

dfc = (
        dfc
        .groupby(by = ["year"])
        .agg({"thi_high":["count"],         # number of days high thi
              "thi_max":["count"],          # number of days max thi = 68
              "thi_hrs_high":["mean",""],
              "thi_ishigh":["sum"],
              "thi_mild":["sum"],
              "thi_mod":["sum"],
              "thi_sev":["sum"],
              "temp_ishigh":["sum"],
              "temp_islow":["sum"]
              })
        ).reset_index()
    

# TODO: describe in words and numbers (document)


#%% weather per hour / heat load and heat recovery in past 24,48h, 3d,4d,5d,6d,7d




# TODO: prepare monday meeting