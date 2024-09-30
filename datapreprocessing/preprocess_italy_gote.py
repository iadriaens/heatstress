# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:17:47 2023

@author: u0084712

--- 

Italian data = farm 50

====================  DMY  =====================
STEP1: load data
STEP2: preprocess and select


====================  ACT  =====================



====================  THI  ======================




"""


import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","datapreprocessing")
os.chdir(path)


#%% load packages

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
import ruptures as rpt
import numpy as np


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

# path to data
path_data = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","data")

# farm selected
farms = [1,2,3,4,5,6]


#%% ACTIVITY: read data, preprocess and select

for farm in farms:
    # read data
    data = pd.read_csv(os.path.join(path_data, "act_" + str(farm) + ".txt"),
                       index_col=0)
    data["measured_on"] = pd.to_datetime(data["measured_on"],format='%Y-%m-%d %H:%M:%S')
    print(data["measured_on"].min())
    data["farm_id"] = farm
    data["date"] = data["measured_on"].dt.date
    data = data.sort_values(by = ["animal_id","measured_on"]).reset_index(drop=1)
    
    # delete data that has erroneous  dates
    data = data.loc[data["date"] >= date(2005,1,1),:]
    print("farm = " + str(farm) + ", startdate = " + str(data["measured_on"].min()))
    data["date"] =  pd.to_datetime(data["measured_on"].dt.date, format = "%Y-%m-%d")
    
    # unique cows / cowids
    cows = data["animal_id"].drop_duplicates().reset_index(drop=1)
    cowlac = data[["animal_id","parity"]].drop_duplicates().sort_values(by=["parity","animal_id"]).reset_index(drop=1)
    #print(cowlac)
    
    # delete parity 0 (data before first calving) and data with NaN values
    data = data.loc[data["activity_total"].isna()==False,:]
    data = data.drop(data.loc[(data["parity"]==0) | (data["parity"].isna()),:].index).reset_index(drop=1)
    cowlac = data[["animal_id","parity"]].drop_duplicates().sort_values(by=["parity","animal_id"]).reset_index(drop=1)
    #print(cowlac)
    
    # aggregate data for activity per day
    data = data.sort_values(by = ["animal_id","parity","measured_on"]).reset_index(drop=1)
    # activity =  data[["animal_id","date","activity_total"]].groupby(by = ["animal_id","date"]).sum()
    # activity = activity.rename(columns = {"activity_total" : "activity"})
    
    """
    # remove data with summed activity > 1800 or smaller than 180
    activity = activity.loc[(activity["activity"]<1800)&(activity["activity"]>180),:].reset_index(drop=1)
    """
    
    # summarize and remove where too little data
    activity_count = data[["animal_id","date","activity_total"]].fillna(-1).groupby(by = ["animal_id","date"]).count()
    activity_sum = data[["animal_id","date","activity_total"]].fillna(-1).groupby(by = ["animal_id","date"]).sum()
    activity_count["sum"] = activity_sum["activity_total"]
    activity_count = activity_count.sort_values(by = "activity_total")
    activity_count = activity_count.rename(columns = {"activity_total" : "count"})
    del activity_sum
    
    # select data along day-cow criteria
    activity_count = activity_count.reset_index()
    activity_count = activity_count.loc[(activity_count["sum"] >180) & \
                                        (activity_count["sum"] < 1800) & \
                                        (activity_count["count"] == 12),:].reset_index(drop=1)
    #merge data and activity_count to select data with completeness
    data = pd.merge(data,activity_count[["animal_id","date"]],how = "inner").reset_index(drop=1)
        
    # dates where activity has too few data points different from nan (rumination only)
    day_count = data[["date","activity_total"]].groupby(by = ["date"]).count().reset_index()
    day_count["date"] = pd.to_datetime(day_count["date"],format='%Y-%m-%d')
    day_count = day_count.loc[day_count["activity_total"] > 0,:]
    
    # if too few data points (less than 180 days) > delete farm from list
    if (day_count["date"].max() - day_count["date"].min()).days < 180:
        print("farm "  + str(farm) + " has less than 180d valid days of activity data and is removed")
    
    else:
        
        # get daily activity
        activity =  (
            data[["farm_id","animal_id","lactation_id","parity","date","activity_total"]]
                .groupby(by = ["farm_id","animal_id","lactation_id","parity","date"]).sum()
                .reset_index()
                )
        activity = activity.rename(columns = {"activity_total" : "activity"})
        
        # add start and enddate to cowlac (animal_id, parity, date_min, date_max)
        cowlac = activity[["animal_id","parity","date"]].groupby(by = ["parity","animal_id"]).agg({"date":["min","max"]}).reset_index()
        cowlac.columns = cowlac.columns.droplevel()
        cowlac.columns = ["parity","animal_id","date_min","date_max"]
        
        # calculate end - begin days
        cowlac["no_days"] = (cowlac["date_max"] - cowlac["date_min"]).dt.days
        cowlac = cowlac.loc[cowlac["no_days"] > 180,:].sort_values(by = ["animal_id","parity"]).reset_index(drop=1)
        
        # drop data of cows not in cowlac, both in activity and in data
        activity = (pd.merge(activity,cowlac, 
                        how = 'inner', on = ["parity", "animal_id"])).reset_index(drop=1)
        
        # sort and calculate gapsize
        activity = activity.sort_values(by = ["animal_id","parity","date"]).reset_index(drop=1)
        activity["gap"] = (activity["date"].iloc[0:-1] - activity["date"].iloc[1:].reset_index(drop=1)).dt.days 
        idx = (activity[["animal_id","parity"]].drop_duplicates(inplace = False, keep = "last")).index.values
        activity.loc[idx,"gap"] = -1
        
        # select maximum gap per cowlac = 9 days
        sel = activity[["animal_id","parity","gap"]].groupby(by = ["animal_id","parity"]).min().reset_index()
        sel = sel.loc[sel["gap"] > -10,:]
        activity = pd.merge(activity,sel[["animal_id","parity"]], how = "inner", on = ["animal_id","parity"]).reset_index(drop=1)
        cowlac = pd.merge(cowlac,sel, how = "inner", on = ["animal_id","parity"]).reset_index(drop=1)
        
        # sort activity
        activity = activity.sort_values(by = ["animal_id","parity","date"]).reset_index(drop=1)
        
        # del obsolete variables
        del activity_count, cows, idx, sel        
        
        # =====================================================================
        # --------------------- remove estrus spikes --------------------------
        from scipy.signal import savgol_filter as savgol

        # select individual curves for plotting + quantify peaks
        randanimals = cowlac.sample(10)[["animal_id","parity"]].index.values  # random plots
        sns.set_style("whitegrid")
        for i in range(0,len(cowlac)):
            
            # preprocess individual activity
            df = activity.loc[(activity["animal_id"] == cowlac["animal_id"][i]) & \
                              (activity["parity"] == cowlac["parity"][i]),["date","activity"]]
            
            df["act_sm"] = savgol(df["activity"],7,1)
            df["act_res"] = df["activity"] - df["act_sm"]
            # median absolute deviation in 9 days rolling window
            def mad(x):
                return np.median(np.fabs(x - np.median(x)))
            
            df["mad"] = df["act_res"].rolling(window = 9,center = True).apply(mad,raw=True)
            # threshold = at least 20% above mean activity level of this cow
            m = (df["activity"].mean()*0.20)/4
            df.loc[(df["mad"] < m) | (df["mad"].isna()), "mad"] = m
            df["mad_thres"] = df["act_sm"] + 4 * df["mad"]
            
            # set values to smoothed values when above threshold
            df["act_new"] = df["activity"]
            df.loc[df["activity"]>df["mad_thres"],"act_new"] = df["act_sm"]
            
            # add df to act
            activity.loc[df.index.values,"act_new"] = df["act_new"]
        
            # plot if in randomly selected 
            if i in randanimals:
                print(i)
                fig, ax = plt.subplots(2,1,figsize = (15,8), sharex = True)
                ax[0].grid(True)
                ax[0].plot(df["date"],df["activity"], linestyle = "-",linewidth = 1,
                           marker= "s", markersize = 2.3,
                           color = "teal" )
                ax[0].set_xlim([df["date"].min(),df["date"].max()])
                
                # best smoother = 1st order Savitsky-Golay filter window 7d
                ax[0].plot(df["date"],df["act_sm"], linestyle = "-",linewidth = 1.2,
                           color = "blue")
                ax[0].set_title("farm = " + str(farm) + ", activity of cow " + \
                                str(cowlac["animal_id"][i]) +" in parity " + str(round(cowlac["parity"][i])))
                ax[1].set_xlabel("date") 
                ax[0].set_ylabel("activity")
                ax[0].legend(["activity","savgol(7d,p1)"])
                
                # plot threshold and "estrus" related spikes
                ax[0].set_ylim(ax[0].get_ylim()[0],ax[0].get_ylim()[1])
                ax[0].fill_between(df["date"],ax[0].get_ylim()[0],df["mad_thres"],
                           linewidth = 0.1, color = "olive", alpha=0.2)
                
                ax[0].plot(df.loc[df["activity"]>df["mad_thres"],"date"],
                           df.loc[df["activity"]>df["mad_thres"],"activity"],
                           marker  = "x", markersize = 8, markeredgewidth = 2,
                           linewidth = 0, color = "red")
                
                # plot corrected values
                ax[1].plot(df["date"],df["act_new"], linestyle = "-",linewidth = 1,
                           marker= "s", markersize = 2.3,
                           color = "teal")
                ax[1].plot(df.loc[df["activity"]>df["mad_thres"],"date"],
                           df.loc[df["activity"]>df["mad_thres"],"act_new"],
                           marker  = "s", markersize = 3,
                           linewidth = 0, color = "red")
                ax[1].set_ylabel("estrus corrected activity")
                ax[1].legend(["activity","corrected"])
                ax[1].set_ylim(ax[0].get_ylim()[0],ax[0].get_ylim()[1])
                
                plt.savefig(os.path.join(path,"results","activity",
                   "act_indcurve_farm_" + str(farm) + "_cow_" + str(cowlac["animal_id"][i]) + ".tif"))
                plt.close()
        
        # =====================================================================
    
        # plot activity profiles in function of date with boxplot
        #%matplotlib qt
        activity["date"] = pd.to_datetime(activity["date"],format='%Y-%m-%d')
        activity = activity.sort_values(by = "date").reset_index(drop=1)
        activity["week_ref"] = activity["date"] - pd.to_timedelta(activity["date"].dt.weekday, unit = "d")
        
        # # create empty values for dates in week_ref that do not exist
        # test = pd.DataFrame([], columns = ["week_ref"])
        # test["week_ref"] = pd.date_range(start = activity["week_ref"].min(), 
        #               end = activity["week_ref"].max(),freq="W") 
        
        # activity sort
        activity = activity.sort_values(by = "week_ref").reset_index(drop=1)
        
        
        fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize=(15,8))
        
        # ----------------------------- plot activity -------------------------
        sns.boxplot(data=activity, x="week_ref", y="act_new", 
                     fliersize=0, whis = 0.8)
        ax.set_ylim([0,1500])
        # convert all xtick labels to selected format from ms timestamp
        xticks = ax.xaxis.get_ticklabels()
        xticks = ax.get_xticks()
        
        # ax.set_xticklabels([pd.to_datetime(data["week_ref"].strftime('%Y-%m-%d\n %H:%M:%S') for tm in xticks],
        #                    rotation=50)
    
        n_weeks = [1,11,22,33,44]
        prev_year = activity["date"].dt.year.min()
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
        ax.set_title("Daily activity farm " + str(farm))
        ax.set_xlabel("date")
        ax.set_ylabel("activity")
        del n,label
        
        #------------------------------- plot THI ---------------------------------
        weather = pd.read_csv(os.path.join(path_data, "weather_" + str(farm) + ".txt"),
                           index_col=0)
        weather["datetime"] = pd.to_datetime(weather["datetime"], format = "%Y-%m-%d %H:%M:%S")
        weather["date"] = pd.to_datetime(weather["datetime"].dt.date, format = "%Y-%m-%d")
        
        # weather["week_ref"] = weather["date"] - pd.to_timedelta(weather["date"].dt.weekday, unit = "d")
        
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
    
        plot_weather = pd.merge(weekweather,activity["week_ref"].drop_duplicates(),on = "week_ref", how = "inner")
        
        # add to plot : mean THI and max daily temperature
        ax2 = ax.twinx()
        ax2.plot(plot_weather.index.values, plot_weather["temp_max"],
                linewidth = 1.5, color = "r") 
        ax2.plot(plot_weather.index.values, plot_weather["thi_mean"],
                linewidth = 1.5, color = "blue")  
        ax2.set_ylim([-120,85])
        ax2.set_ylabel("THI and max daily temperature")  
        ax2.grid(False)      
        
        plt.savefig(os.path.join(path,"results","activity","act_thi_farm_" + str(farm) + ".tif"))    
        del ax, ax2, currentyear, fig, n_weeks, prev_year
        
        # -------------------------------- combine and save -------------------
        
        activity2 = activity[["farm_id","animal_id","parity","date","week_ref","activity","act_new"]]
        # activity2.to_csv(os.path.join(path,"results","act_preprocessed_" + str(farm) + ".txt"))

        # ---------------- summarize and plot stats activity ------------------
        def q10(x):
            return x.quantile(0.1)
        def q90(x):
            return x.quantile(0.9)
        sumstat = (activity[["date","act_new"]].groupby(by = "date").agg({"act_new" : ["count","mean","std","median",q10,q90]})).reset_index()
        sumstat.columns = sumstat.columns.droplevel()
        sumstat.columns = ["date","count","mean",
                                          "std","median",
                                          "q10","q90"]
        sumstat["date"] = pd.to_datetime(sumstat["date"], format = "%Y-%m-%d")
        
        # --------------------- correct differences in data stats without removing information
        # set number of breakpoints per farm
        no_breaks = {"1": 0,  # 34
                     "2": 1,  # 38
                     "3": 0,  # 39
                     "4": 2,  # 43
                     "5": 2,  # 44
                     "6": 0,  # IT
                     }        
        n_bkps = no_breaks[str(farm)]
        
        # plot stats ----  %matplotlib qt
        fig,ax = plt.subplots(nrows = 2, ncols = 1, figsize=(15,10))
        
        ax[0].fill_between(sumstat["date"],sumstat["q10"],sumstat["q90"], 
                         color = "mediumseagreen", alpha = 0.5)
        ax[0].plot(sumstat["date"],sumstat["median"], linewidth = 2, color = "teal")
        
        ax[1].plot(sumstat["date"],sumstat["count"], linewidth = 2, color = "indianred")
        # ax[1].grid(False)     
        ax[1].set_ylim([0,sumstat["count"].max()+20])
        ax[1].set_xlabel("date")
        ax[0].set_xlabel("date")
        ax[1].set_ylabel("count - number of animals")
        ax[0].set_title("farm = "+str(farm)+ ", summary stats of activity data, corrected for estrus")
        ax[1].set_title("number of measurements, i.e., cows per day")
        ax[0].set_ylabel("median  - IQR of activity")
        
        # prepare correction
        sumstat["cormean"] = np.nan
        sumstat["corstd"] = np.nan
        
        if n_bkps == 0:
            # correct to mean + std of 
            print(0,len(sumstat)-1)
            sumstat["cormean"] = round(sumstat["median"].mean(),2)
            sumstat["corstd"] = round(sumstat["median"].std(),2)
            
            ax[0].plot([sumstat["date"].iloc[0],sumstat["date"].iloc[-1]],
                       [sumstat["cormean"].iloc[0],sumstat["cormean"].iloc[0]],
                       color = "crimson",linewidth = 2, linestyle = '-')
            
        else:
            # find breakpoints - set breakpoints per farm based on visual assessment
            signal = sumstat[["q10","median","q90"]].to_numpy()
            algo = rpt.Dynp(model="l2").fit(signal)
            result = algo.predict(n_bkps=n_bkps)  # find correct number of breaks
            
            #rpt.display(signal, result)
            print(result)
            brkpts = sumstat.iloc[result[:-1]]["date"]
            resultrange = [0]+result
            
            # plot breakpoints
            for i in range(0,len(brkpts)):
                print(brkpts.iloc[i])
                ax[0].set_ylim([ax[0].get_ylim()[0], ax[0].get_ylim()[1]])
                ax[1].set_ylim([ax[1].get_ylim()[0], ax[1].get_ylim()[1]])
                ax[0].plot([brkpts.iloc[i],brkpts.iloc[i]],
                           [ax[0].get_ylim()[0], ax[0].get_ylim()[1]],
                           color = 'royalblue',linestyle = '--',linewidth = 2)
                ax[1].plot([brkpts.iloc[i],brkpts.iloc[i]],
                           [ax[1].get_ylim()[0], ax[1].get_ylim()[1]],
                           color = 'royalblue',linestyle = '--',linewidth = 2)
            
            # find and plot correction parameters
            for i in range(1,len(resultrange)):
                print(resultrange[i-1],resultrange[i])
                cormean = sumstat.iloc[resultrange[i-1]:resultrange[i]-1]["median"].mean()
                corstd = sumstat.iloc[resultrange[i-1]:resultrange[i]-1]["median"].std()
                
                print(round(cormean,2), round(corstd,2))
                ax[0].plot([sumstat.iloc[resultrange[i-1]]["date"],
                            sumstat.iloc[resultrange[i]-1]["date"]],
                           [cormean,cormean],
                           color = "crimson",linewidth = 2, linestyle = '-')
                
                # add to sumstat
                sumstat.loc[resultrange[i-1]:resultrange[i]-1,"cormean"] = cormean
                sumstat.loc[resultrange[i-1]:resultrange[i]-1,"corstd"] = corstd
              
        # save plot
        plt.savefig(os.path.join(path,"results","activity","act_stats_breaks_aftercorr" + str(farm) + ".tif"))
        
        # implement statistical correction
        activity2 = pd.merge(activity2,sumstat[["date","cormean","corstd"]],
                             on ="date")
        activity2["act_corr"] = (activity2["act_new"] - activity2["cormean"]) / \
                                 activity2["corstd"]
                                
        activity2 = activity2[["farm_id","animal_id","parity","date","week_ref","activity","act_new","act_corr"]]
                             
        # plot corrected data
        sumstat2 = (activity2[["date","act_corr"]].groupby(by = "date").agg({"act_corr" : ["median",q10,q90]})).reset_index()
        sumstat2.columns = sumstat2.columns.droplevel()
        sumstat2.columns = ["date","median","q10","q90"]
        sumstat2["date"] = pd.to_datetime(sumstat2["date"], format = "%Y-%m-%d")
        
        # plot stats corrected ----  %matplotlib qt
        fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize=(15,6))
        
        ax.fill_between(sumstat2["date"],sumstat2["q10"],sumstat2["q90"], 
                         color = "lightskyblue", alpha = 0.5)
        ax.plot(sumstat2["date"],sumstat2["median"], linewidth = 2, color = "steelblue")
        ax.set_xlabel("date")
        ax.set_title("farm = "+ str(farm)+ ", summary stats of corrected activity data")
        ax.set_ylabel("median  - IQR of activity")

        # save plot
        plt.savefig(os.path.join(path,"results","activity","act_stats_corrdata_" + str(farm) + ".tif"))
        plt.close()
        # -------------------------------- combine and save -------------------
        
        weather.to_csv(os.path.join(path, "results","data","weather_farm_" + str(farm) + ".txt"))        
        activity2.to_csv(os.path.join(path,"results","data","act_preprocessed_" + str(farm) + ".txt"))


del activity, activity2,algo,ax,brkpts,cormean,corstd,cowlac,data,day_count
del df, farm, fig, i, m, n_bkps, no_breaks, plot_weather,randanimals, result
del resultrange, signal, sumstat, sumstat2, weather, weekofyear, weekweather
del xlabels, xticks


#%% Summary table selection
#==============================================================================
#==============================================================================
#==============================================================================

startdates = {1 : 2011,
              2 : 2014,
              3 : 2014,
              4 : 2017,
              5 : 2016,
              6 : 2013}
enddates = {1 : 2019,
            2 : 2020,
            3 : 2017,
            4 : 2022,
            5 : 2019,
            6 : 2020}



#%% MILK YIELD read and preprocess data

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
    # data["mi"] = data["mi"]/(3600) # not needed for CowBase
    
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
    
    # print and store average 305 day milk yield for summary table
    my305d = milk.loc[milk["dim"]<306,["animal_id","parity","dim","dmy","date"]]
    aniids = my305d.loc[(my305d["date"].dt.year >= startdates[farm]) & \
                  (my305d["date"].dt.year <= enddates[farm]),["animal_id","parity"]].drop_duplicates().reset_index(drop=1)
    my305d = my305d.merge(aniids,how="inner",on=["animal_id","parity"]).reset_index(drop=1)
    test = my305d[["animal_id","dmy","parity"]].groupby(by=["animal_id","parity"]).sum()
    test2 = my305d[["animal_id","dmy","parity"]].groupby(by=["animal_id","parity"]).count()
    test3 = test.loc[test2["dmy"]>290]  # for calculation of 305d my
    test4 = (test.loc[test2["dmy"]>50]).reset_index() # for the modelling - selection
    
    print("farm " + str(farm) + ": average 305d MY = " + str(round(test3["dmy"].mean())) + " +/- " + str(round(test3["dmy"].std())) +" kg")
    print("farm " + str(farm) + " has " + str(len(test4)) + " cowlacs in the dataset")
    print("farm " + str(farm) + " has " + str(len(test4.animal_id.drop_duplicates())) + " cows in the dataset")
    
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
    
    subset = milk.loc[milk["dim"]>600,["animal_id"]].drop_duplicates()
    
    # plot + save random 10 with extended lactation (seems OK)
    for cow in subset.sample(5)["animal_id"].values:
        print(cow)
        fig,ax = plt.subplots(1,1,figsize = (15,6))
        sns.lineplot(data=milk.loc[milk["animal_id"]==cow,:],x = "date",y = "dmy",
                     hue = "parity",linewidth = 1.5,marker = "o",markersize=4,
                     palette = sns.color_palette("bright",n_colors = len(milk.loc[milk["animal_id"]==cow,"parity"].drop_duplicates())))
        ax.set_title("farm = " + str(farm) + ", cow = " + str(cow) + ", extended lactation > 600 days", size=14)
        ax.set_ylabel("daily milk yield [kg]")
        plt.savefig(os.path.join(path,"results","milk","farm_ " + str(farm) + "_extended_lactations_cow" + str(cow) + ".tif"))
        plt.close()
    
    del ax, cow, fig, idx, idx_prev, subset
    
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
    milk.to_csv(os.path.join(path,"results","data","milk_preprocessed_" + str(farm) + ".txt"))
    
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



#%% THI

import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataanalysis")
os.chdir(path)


#%% import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib qt


#%% paths and constants

path_data = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","data")

# farm selected
farms = [1,2,3,4,5,6]

#%% explore weather data and thi features for all farms

# read data and preprocess to get measurements per hour per farm
weather = pd.DataFrame([])
for farm in farms:
    # read data and preprocess for obtaining data per hour
    df = pd.read_csv(os.path.join(path_data,"weather_" + str(farm) + ".txt"),
                     index_col = 0)
    df["datetime"] = pd.to_datetime(df["datetime"], format = "%Y-%m-%d %H:%M:%S")
    df["farm_id"] = farm
    df["hourly"] = df["datetime"].round("H")
                   
    # calculate average weather variables per hour
    wea = (
           df[["farm_id","hourly", "temp","rel_humidity"]]
           .groupby(by=["farm_id","hourly"]).mean()
           ).reset_index()
    wea.columns = ["farm_id","hourly","temp","rel_humidity"]

    # calculate THI = (1.8 × Tmean + 32) − [(0.55 − 0.0055 × RHmean) × (1.8 × Tmean − 26)]
    wea["thi"] = 1.8 * wea["temp"] + 32 - \
                    ((0.55 - 0.0055 * wea["rel_humidity"]) * \
                     (1.8 * wea["temp"] - 26))
                        
    weather = pd.concat([weather,wea])
    del df, wea
    del farm

# calculate THI features per day for each farm
#   - hours high >=68 THI of that day
#   - hours cooled down < 21° at night
#   - successive days with high THI
#   - % hours insulted in x (fixed) days before today
#   - % hours insulted in successive days with high THI
weather["date"] = pd.to_datetime(weather["hourly"].dt.date, format = "%Y-%m-%d")
weather["temp_ishigh"] = 0   # to count hours high temp
weather.loc[weather["temp"] >= 25,"temp_ishigh"] = 1
weather["temp_islow"] = 0   # to count hours low temp
weather.loc[weather["temp"] <= 18,"temp_islow"] = 1
weather["thi_ishigh"] = 0
weather.loc[weather["thi"]>=68,"thi_ishigh"] = 1
weather["thi_mild"] = 0   # to count high 
weather.loc[(weather["thi"] >= 68) & (weather["thi"] < 72),"thi_mild"] = 1
weather["thi_mod"] = 0   # 
weather.loc[(weather["thi"] >= 72) & (weather["thi"] < 80),"thi_mod"] = 1
weather["thi_sev"] = 0   # 
weather.loc[(weather["thi"] >= 80),"thi_sev"] = 1

# weather["date"].head(35)

# add a day 12:00 to 12:00 variable for recovery
weather["halfday"] = weather["hourly"] - pd.to_timedelta(12, unit = "h")
weather["halfdate"] = weather["halfday"].dt.date

# add a day 12:00 to 12:00 variable for recovery next day
weather["halfdaynext"] = weather["hourly"] + pd.to_timedelta(12, unit = "h")
weather["halfdatenext"] = weather["halfdaynext"].dt.date

# hours recovery from 12 (noon) to 12 (noon) in prevous and current day
test = weather[["farm_id","halfdate","temp_islow"]].groupby(by = ["farm_id","halfdate"]).sum().reset_index()
test.columns = ["farm_id","date","hrs_rec_succ_prev"]
test["date"] = pd.to_datetime(test["date"],format = "%Y-%m-%d")
test = test.loc[(test["date"].dt.year > 2004),:].reset_index(drop=1)

# hours recovery from 12 (noon) to 12 (noon) in current and next day
test2 = weather[["farm_id","halfdatenext","temp_islow"]].groupby(by = ["farm_id","halfdatenext"]).sum().reset_index()
test2.columns = ["farm_id","date","hrs_rec_succ_next"]
test2["date"] = pd.to_datetime(test2["date"],format = "%Y-%m-%d")
test2.loc[test2["date"] == pd.to_datetime("2005-01-01",format = "%Y-%m-%d"),"hrs_rec_succ_next"] = 24
test2 = test2.loc[test2["date"] < weather["date"].max(),:]



# prepare dataframe per day with the different THI derivates
data = (
        weather[['farm_id', 'date', 'temp', 'rel_humidity', 'thi', 
                 'temp_ishigh','temp_islow','thi_mild','thi_mod','thi_sev','thi_ishigh']]
        .groupby(by = ["farm_id","date"])
        .agg({"temp":["count","min","max"],
              "rel_humidity":["min","max"],
              "thi":["mean","max"],
              "thi_ishigh":["sum"],
              "thi_mild":["sum"],
              "thi_mod":["sum"],
              "thi_sev":["sum"],
              "temp_ishigh":["sum"],
              "temp_islow":["sum"]
              })
        ).reset_index()
data.columns = data.columns.droplevel()
data.columns = ["farm_id","date","no_meas","temp_min","temp_max","rh_min","rh_max",
                "thi_avg","thi_max","thi_hrs_high","thi_hrs_mild","thi_hrs_mod",
                "thi_hrs_sev","temp_hrs_high","temp_hrs_low"]
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["week"] = np.floor(data["date"].dt.dayofyear/7)
data["day"] = data["date"].dt.dayofyear
data = data.sort_values(by = ["farm_id","date"]).reset_index(drop=1)

# add the % of hours with high THI in 5 days prior
data["perc_thi_5d_prior"] = np.nan
data.loc[4:,"perc_thi_5d_prior"] = (data["thi_hrs_high"].iloc[0:-4].values + \
                                     data["thi_hrs_high"].iloc[1:-3].values + \
                                     data["thi_hrs_high"].iloc[2:-2].values + \
                                     data["thi_hrs_high"].iloc[3:-1].values + \
                                     data["thi_hrs_high"].iloc[4:].values) / (5*24)*100

# add the % of hours with high THI in 2 days prior
data["perc_thi_2d_prior"] = np.nan
data.loc[2:,"perc_thi_2d_prior"] = (data["thi_hrs_high"].iloc[0:-2].values + \
                                    data["thi_hrs_high"].iloc[1:-1].values) / (2*24)*100 

# add recovery capacity previous day
data["date"] = pd.to_datetime(data["date"],format = "%Y-%m-%d")
data = pd.merge(data,test, on = ["farm_id","date"])

# add revovery capacity next day
data = pd.merge(data,test2, on = ["farm_id","date"])


# add 0/1 whether thi is on average above 68
data["thi_high"] = 0
data.loc[data["thi_avg"]>=68,"thi_high"] = 1

# add number of days thi was successively high including today
data["no_days_highTHI"] = pd.DataFrame(data["thi_high"]).eq(0).cumsum().groupby('thi_high').cumcount()

# add days high THI since start of year
data["no_days_year_prior"] = data[["farm_id","year","thi_high"]].groupby(by = ["farm_id","year"]).cumsum()

# add number of days that thi was high in 5 days prior
data["no_days_5d_prior"] = 0
data.loc[4:,"no_days_5d_prior"] = data["thi_high"].iloc[0:-4].values + \
                                  data["thi_high"].iloc[1:-3].values + \
                                  data["thi_high"].iloc[2:-2].values + \
                                  data["thi_high"].iloc[3:-1].values + \
                                  data["thi_high"].iloc[4:].values
                                  
# add difference in temperature 
data["temp_difference"] = data["temp_max"] - data["temp_min"]                                  
                                  
# correct for when a new farm starts: no_days_5d_prior, perc_thi_2d_prior, perc_thi_5d_prior
data.loc[data["date"].dt.dayofyear < 6,["no_days_5d_prior","perc_thi_2d_prior","perc_thi_5d_prior"]] = 0

# correct for when less than 20 measurements per day => set to nan only 9 times in whole dataset)
data.loc[(data["no_meas"] < 20),:] = np.nan

# make summary of weather information over all farms per date
wea_all = data.groupby(by=["date"]).mean()
sumwea = (wea_all.describe().T).drop(
      index = ["year","month","day","week","thi_high","farm_id","no_meas"],
      columns = ["count","25%","50%","75%"])

del test, test2

# save weather features
data.to_csv(os.path.join(path,"data","weatherfeatures.txt"))

for farm in farms: 
    subset = data.loc[data["farm_id"] == farm,:].reset_index(drop=1)
    subset.to_csv(os.path.join(path,"data","weatherfeatures_" + str(farm) + ".txt"))
    