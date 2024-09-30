# -*- coding: utf-8 -*-
"""
Created on Fri May 26 08:50:37 2023

@author: Ines Adriaens
-------------------------------------------------------------------------------
DAILY ACTIVITY DATA
-------------------------------------------------------------------------------
GOAL:
== overall: link between individual heat stress susceptibility and behaviour
-- this script: preprocess and select activity data

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
import seaborn as sns
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
import matplotlib.pyplot as plt
from datetime import date
import ruptures as rpt
import numpy as np
from scipy.signal import savgol_filter

def q10(x):
    return x.quantile(0.1)
def q90(x):
    return x.quantile(0.9)


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
                    "projects","ugent","heatstress","data","new")

# farm selected
# farms = [30,31,33,34,35,38,39,40,43,44,45,46,48] 
farms = [41,42,43,44,46,47,48,49,50,51,54,55,57,58,59,61,62,64,65,66,67,68,69] 



#%% read data

for farm in farms:
    # read data
    data = pd.read_csv(os.path.join(path_data, "act_" + str(farm) + ".txt"),
                       usecols = ["farm_id","animal_id","parity","measured_on",
                                  "dim","activity_total"])
    data["measured_on"] = pd.to_datetime(data["measured_on"],format='%Y-%m-%d %H:%M:%S')
    # print(data["measured_on"].min())
    data["farm_id"] = farm
    data["date"] = data["measured_on"].dt.date
    
    # delete data that has erroneous dates
    data = data.loc[data["date"] >= date(2005,1,1),:]
    print("farm = " + str(farm) + ", startdate = " + str(data["measured_on"].min()))
    
    # unique cows / cowids
    cows = data["animal_id"].drop_duplicates().reset_index(drop=1)
    cowlac = data[["animal_id","parity"]].drop_duplicates().sort_values(by=["parity","animal_id"]).reset_index(drop=1)
    #print(cowlac)
    
    # delete parity 0 (data before first calving) and data with NaN values
    data = data.loc[data["activity_total"].isna()==False,:]
    data = data.drop(data.loc[data["parity"]==0,:].index).reset_index(drop=1)
    cowlac = data[["animal_id","parity"]].drop_duplicates().sort_values(by=["parity","animal_id"]).reset_index(drop=1)
    #print(cowlac)
    
    # aggregate data for activity per day
    data = data.sort_values(by = ["animal_id","parity","measured_on"]).reset_index(drop=1)
    activity =  data[["animal_id","date","activity_total"]].groupby(by = ["animal_id","date"]).sum()
    activity = activity.rename(columns = {"activity_total" : "activity"})
    
    # summarize and remove where too little data
    activity_count = data[["animal_id","date","activity_total"]].dropna().groupby(by = ["animal_id","date"]).count()
    activity_sum = data[["animal_id","date","activity_total"]].dropna().groupby(by = ["animal_id","date"]).sum()
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
    day_count = day_count.sort_values(by = ["activity_total"]).reset_index(drop=1)
    day_count["date"] = pd.to_datetime(day_count["date"],format='%Y-%m-%d')
    data["date"] = pd.to_datetime(data["date"],format='%Y-%m-%d')
    day_count = day_count.loc[day_count["activity_total"] > 100,:].reset_index(drop=1)
    data = data.merge(day_count["date"],how="inner",on="date").sort_values(by=["animal_id","measured_on"]).reset_index(drop=1)
    
    del cows, cowlac, activity, activity_count
    
    # test how high activity when in estrus
    data["year"] = data["date"].dt.year
    data["week"] = data["date"].dt.weekofyear
    data["time"] = data["measured_on"].dt.hour
    data["t4"] = data["time"] - data["time"]%4
    
    # combine data in 4hourly instead of 2 hourly measurements
    test = (
            data[["farm_id","animal_id","date","t4","activity_total"]]
            .groupby(by = ["farm_id","animal_id","date","t4"]).sum()
            ).reset_index()
    data = (
            data.drop(columns={"activity_total"})
            .merge(test,on=["farm_id","animal_id","date","t4"])
            )
    data=data.loc[data["measured_on"].dt.hour == data["t4"],: ].reset_index(drop=1)
    del test
    
    # test = data.loc[data["year"]==2011,:]
    # sns.lineplot(data=test,x="measured_on",y="activity_total",hue="animal_id",estimator=None)
    
    # if too few data points (less than 180 days) > delete farm from list
    if (day_count["date"].max() - day_count["date"].min()).days < 180:
        print("farm "  + str(farm) + " has less than 180d valid days of activity data and is removed")
    
    else:
        
        """
        - data per hour -> corrected for herd average + normalise 
            ==> trend in stats over time
        - data per day = aggregation of data per hour
        """
        # summary and smoothed per time window
        # _,ax = plt.subplots(1,1,figsize=(15,8))
        # dfa = pd.DataFrame([],columns=['farm_id', 'animal_id', 'parity', 'dim', 'date', 'measured_on', 'year',
        #        'week', 't4', 'activity_total', 'min', 'max'])
        # for h in [0,4,8,12,16,20]:
            
        #     # summary per time window of 4 hours, summarized per week
        #     summary = (
        #            data.loc[data["t4"]==h,["year","week","activity_total"]]
        #            .groupby(by = ["year","week"]).agg({"activity_total" : ["min","max","mean"]})
        #            ).sort_values(by = ["year","week"]).reset_index()
        #     summary.columns = summary.columns.droplevel()
        #     summary.columns = ["year","week","min","max","mean"]
        #     summary["act_sm"] = summary[["mean"]].apply(lambda x: savgol_filter(x,9,2)) # smooth 4.5 weeks each side    
        #     summary = summary.reset_index()
        #     summary["min"] = 0
        #     summary["max"] = 2*summary["mean"]
            
        #     # decision for correction = standardisation according to [0,2*avg per week]
        #     df = (
        #           data.loc[data["t4"]==h,["farm_id","animal_id","parity","dim","date",
        #                                   "measured_on","year","week","t4","activity_total"]]
        #           .reset_index(drop=1)
        #           .merge(summary[["year","week","min","max"]], on=["year","week"])
        #           )
        #     dfa = pd.concat([df,dfa])
            
        # del h,df,summary
        
        # # correct ts for "min" and "max"
        # dfa["act"] = (dfa["activity_total"]-dfa["min"])/(dfa["max"]-dfa["min"])
        # dfa = dfa.sort_values(by=["animal_id","measured_on"]).reset_index(drop=1)  
        # dfa = dfa.drop(columns={"min","max"})


        # test = dfa.loc[dfa["year"]==2015,:]
        # _,ax = plt.subplots(2,1,figsize=(15,8))
        # sns.lineplot(data=test,x="measured_on",y="activity_total",hue="animal_id",estimator=None, ax=ax[0])
        # sns.lineplot(data=test,x="measured_on",y="act",hue="animal_id",estimator=None, ax=ax[1])
        # sns.lineplot(data=test,x="measured_on",y="act",estimator="median", ax=ax[1])

        
            # plt.plot(summary["index"],summary["mean"],"cornflowerblue",lw=0.6) 
            # plt.plot(summary["index"],summary["act_sm"],"b",lw=2)
            # plt.plot(summary["index"],2*summary["act_sm"],"r",lw=2)
            
            
            # plt.hist(data.loc[data["t4"]==0,"activity_total"],bins=20)
            # plt.plot([data.loc[data["t4"]==0,"activity_total"].mean(),data.loc[data["t4"]==0,"activity_total"].mean()],[0,165000],"r--")
            # plt.plot([0,0],[0,165000],"r--")
            # plt.plot([2*data.loc[data["t4"]==0,"activity_total"].mean(),2*data.loc[data["t4"]==0,"activity_total"].mean()],[0,165000],"r--")

            # ax.set_title("activity average per week - 4 hours, rolling 2nd order SG filter 9w")
            # ax.set_xlabel("data")
            # ax.set_ylabel("weekly activity, per 4 hours")        

        
        # # per week average activity at hour level
        # summary = (
        #        data.loc[data["t4"]==0,["year","week","t4","activity_total"]]
        #        .groupby(by = ["year","week","t4"]).agg({"activity_total" : ["min","max","mean"]})
        #        ).sort_values(by = ["year","week","t4"]).reset_index()
        # summary.columns = summary.columns.droplevel()
        # summary.columns = ["year","week","t4","min","max","mean"]
        # summary["act_sm"] = summary[["mean"]].apply(lambda x: savgol_filter(x, 9*6,2)) # smooth 4.5 weeks each side       
        # summary=summary.reset_index()
        
        # # plot correction
        # _,ax = plt.subplots(1,1,figsize=(15,8))
        # plt.plot(summary["index"],summary["mean"],"cornflowerblue",lw=0.6) 
        # plt.plot(summary["index"],summary["act_sm"],"b",lw=2)
        # plt.plot(summary.loc[summary["t4"]==0,"index"],summary.loc[summary["t4"]==0,"mean"],"r",lw=1.2)
        # plt.plot(summary.loc[summary["t4"]==8,"index"],summary.loc[summary["t4"]==8,"mean"],"r",lw=1.2)
        # plt.plot(summary["index"],summary["max"],"r",lw=1.2)
        # ax.set_title("activity average per week - 4 hours, rolling 2nd order SG filter 9w")
        # ax.set_xlabel("data")
        # ax.set_ylabel("weekly activity, per 4 hours")        
        
        # # subtract weekly smoothed (independent of time)
        # data2 = summary[["year","week","t4","act_sm","index"]].merge(data,on=["year","week","t4"],how="inner")
        # data2["act_corr"] = data2["activity_total"] - data2["act_sm"]
        
        # test = data2.loc[data2["year"] == 2015,:]
        # _,ax = plt.subplots(1,1,figsize=(15,8))
        # sns.lineplot(data=test,x="measured_on",y="activity_total",hue="animal_id",estimator=None)
        # sns.lineplot(data=test,x="measured_on",y="act_sm",estimator="mean")

    
    
    
        # get daily activity
        activity =  (
            data[["farm_id","animal_id","parity","date","activity_total"]]
                .groupby(by = ["farm_id","animal_id","parity","date"]).sum()
                .reset_index()
                )
        activity = activity.rename(columns = {"activity_total" : "activity"})
        
        # add start and enddate to cowlac (animal_id, parity, date_min, date_max)
        cowlac = activity[["animal_id","parity","date"]].groupby(by = ["parity","animal_id"]).agg({"date":["min","max"]}).reset_index()
        cowlac.columns = cowlac.columns.droplevel()
        cowlac.columns = ["parity","animal_id","date_min","date_max"]
        
        # calculate end - begin days and select at least 60 days data
        cowlac["no_days"] = (cowlac["date_max"] - cowlac["date_min"]).dt.days
        cowlac = cowlac.loc[cowlac["no_days"] > 60,:].sort_values(by = ["animal_id","parity"]).reset_index(drop=1)
        
        # drop data of cows not in cowlac, both in activity and in data
        activity = (pd.merge(activity,cowlac, 
                        how = 'inner', on = ["parity", "animal_id"])).reset_index(drop=1)
        
        # sort and calculate gapsize
        activity = activity.sort_values(by = ["animal_id","parity","date"]).reset_index(drop=1)
        activity["gap"] = (activity["date"].iloc[0:-1] - activity["date"].iloc[1:].reset_index(drop=1)).dt.days 
        idx = (activity[["animal_id","parity"]].drop_duplicates(inplace = False, keep = "last")).index.values
        activity.loc[idx,"gap"] = -1
        del idx
        
        # select maximum gap per cowlac = 9 days
        sel = activity[["animal_id","parity","gap"]].groupby(by = ["animal_id","parity"]).min().reset_index()
        sel = sel.loc[sel["gap"] > -10,:]
        activity = pd.merge(activity,sel[["animal_id","parity"]], how = "inner", on = ["animal_id","parity"]).reset_index(drop=1)
        cowlac = pd.merge(cowlac,sel, how = "inner", on = ["animal_id","parity"]).reset_index(drop=1)
        
        # sort activity
        activity = activity.sort_values(by = ["animal_id","parity","date"]).reset_index(drop=1)
        
        # del obsolete variables
        del sel        
        
        # =====================================================================
        # --------------------- remove estrus spikes --------------------------
        # from scipy.signal import savgol_filter as savgol

        # select individual curves for plotting + quantify peaks
        randanimals = cowlac.sample(1)[["animal_id","parity"]].index.values  # random plots
        sns.set_style("whitegrid")
        for i in range(0,len(cowlac)):
            
            # preprocess individual activity
            df = activity.loc[(activity["animal_id"] == cowlac["animal_id"][i]) & \
                              (activity["parity"] == cowlac["parity"][i]),["date","activity"]]
            
            # df["act_sm"] = savgol(df["activity"],7,1)
            a = lowess(df["activity"],np.arange(0,len(df)),frac= 7./len(df), it=1)  #7 seven day window frac of data used based on len df
            df["act_sm"] = a[:,1]
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
                   "act_indcurve_LOWESS_farm_" + str(farm) + "_cow_" + str(cowlac["animal_id"][i]) + ".tif"))
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
        weather = pd.read_csv(os.path.join(path_data,"newweather_" + str(farm)+".txt"),
                           index_col=0)
        weather["time"] = pd.to_datetime(weather["time"], format = "%Y-%m-%d %H:%M:%S")
        weather["date"] = pd.to_datetime(weather["time"].dt.date, format = "%Y-%m-%d")
        
        weather = weather.loc[(weather["date"] >= activity["date"].min()) & \
                              (weather["date"] <= activity["date"].max()),:].reset_index(drop=1)

        weather["week_ref"] = weather["date"] - pd.to_timedelta(weather["date"].dt.weekday, unit = "d")
        
        weather = weather[["date","temp","rhum"]].groupby(by = ["date"]).mean().reset_index()
        
        
        # calculate THI = (1.8 × Tmean + 32) − [(0.55 − 0.0055 × RHmean) × (1.8 × Tmean − 26)]
        weather["thi"] = 1.8 * weather["temp"] + 32 - \
                        ((0.55 - 0.0055 * weather["rhum"]) * \
                          (1.8 * weather["temp"] - 26))
        
        # aggregate to weekly data for plotting
        weather["week_ref"] = weather["date"] - \
                              pd.to_timedelta(weather["date"].dt.weekday, unit = "d")
        weekweather = weather[["week_ref",
                               "temp",
                               "rhum",
                               "thi"]].groupby(by = "week_ref").agg({"temp":["mean","max"], 
                                                                     "rhum":["mean","max"],
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
        
        plt.savefig(os.path.join(path,"results","activity","act_thi_farm_new_" + str(farm) + ".tif"))    
        del ax, ax2, currentyear, fig, prev_year
        
        # result = activity per day, not corrected yet for differences in stats
        act = activity[["farm_id","animal_id","parity","date","activity","act_new"]]
        
        
        # merge with data and act per 4 hours
        sub = data.loc[data["time"]==0,["animal_id","date","activity_total"]]
        sub.columns=["animal_id","date","0004"]
        act = act.merge(sub, on= ["animal_id","date"],how='outer')
        sub = data.loc[data["time"]==4,["animal_id","date","activity_total"]]
        sub.columns=["animal_id","date","0408"]
        act = act.merge(sub, on= ["animal_id","date"],how='outer')
        sub = data.loc[data["time"]==8,["animal_id","date","activity_total"]]
        sub.columns=["animal_id","date","0812"]
        act = act.merge(sub, on= ["animal_id","date"],how='outer')
        sub = data.loc[data["time"]==12,["animal_id","date","activity_total"]]
        sub.columns=["animal_id","date","1216"]
        act = act.merge(sub, on= ["animal_id","date"],how='outer')
        sub = data.loc[data["time"]==16,["animal_id","date","activity_total"]]
        sub.columns=["animal_id","date","1620"]
        act = act.merge(sub, on= ["animal_id","date"],how='outer')
        sub = data.loc[data["time"]==20,["animal_id","date","activity_total"]]
        sub.columns=["animal_id","date","2000"]
        act = act.merge(sub, on= ["animal_id","date"],how='outer')
        
        # add dim
        dim = data[["animal_id","date","dim"]].drop_duplicates(["animal_id","date"]).reset_index(drop=1)
        act=act.merge(dim, on= ["animal_id","date"],how='outer')
        del dim,sub,i,df
        
        # calculate fractios
        act['f0004'] = act["0004"]/act["activity"]
        act['f0408'] = act["0408"]/act["activity"]
        act['f0812'] = act["0812"]/act["activity"]
        act['f1216'] = act["1216"]/act["activity"]
        act['f1620'] = act["1620"]/act["activity"]
        act['f2000'] = act["2000"]/act["activity"]
        
        #  calculate corrections for estrus-corrected act
        act['c0004'] = act["f0004"]*act["act_new"]
        act['c0408'] = act["f0408"]*act["act_new"]
        act['c0812'] = act["f0812"]*act["act_new"]
        act['c1216'] = act["f1216"]*act["act_new"]
        act['c1620'] = act["f1620"]*act["act_new"]
        act['c2000'] = act["f2000"]*act["act_new"]
        
        # week for correction
        act["week"] = act["date"].dt.isocalendar().week
        act["year"] = act["date"].dt.year
        
        # save estrus corrected activity, no correction for stats yet
        act.to_csv(os.path.join(path_data,"activity_estruscorr_farm_" + str(farm) + ".txt"))
        
"""
act contains per day, new and per 4 hours corrected with fraction equal to original contributions
"""
del m, day_count, cowlac, n_weeks, a, weekofyear, weekweather, xlabels, xticks
del plot_weather,randanimals
del ax, ax2, currentyear, fig, prev_year

#%% correct stats per hour with method 1 = with rolling per hour

# chosen correction for the differing stats is a min-max correction, 
# in which the correction basis is a rolling 7 day 
for farm in farms:
  
    # load farm data 
    act = pd.read_csv(os.path.join(path_data,"activity_estruscorr_farm_" + str(farm) + ".txt"),
                      index_col=0)
    act = act.loc[act["farm_id"].isna()==False,:].sort_values(by=["animal_id","date"]).reset_index(drop=1)
    
    days = act["date"].drop_duplicates().sort_values().reset_index(drop=1).reset_index()
    act=act.merge(days,on="date",how="outer")
    
    # dfa = act[["index","c0004"]].groupby('index').rolling(30,min_periods=5,center=True,on="index").median()
    # add = act[["index","c0004"]].rolling(30,min_periods=5,center=True,on="index").quantile(0.95)
    
    
    
    
    def q99(x):
        return x.quantile(0.99)
    # calculate rolling weekly averages per set of 4 hours
    dfa = act.groupby(by = ["year","week"]).agg({"c0004":["median",q99],
                                                 "c0408":["median",q99],
                                                 "c0812":["median",q99],
                                                 "c1216":["median",q99],
                                                 "c1620":["median",q99],
                                                 "c2000":["median",q99]}).reset_index()
    dfa.columns = dfa.columns.droplevel(1)
    dfa.columns = ["year","week","c0004_med","c0004_q99","c0408_med","c0408_q99",
                   "c0812_med","c0812_q99","c1216_med","c1216_q99","c1620_med","c1620_q99","c2000_med","c2000_q99"]
    dfa["sm0004"] = dfa[["c0004"]].rolling(6,min_periods=1,center=True).median() # smooth 4.5 weeks each side    
    dfa["sm0408"] = dfa[["c0408"]].rolling(6,min_periods=1,center=True).median()
    dfa["sm0812"] = dfa[["c0812"]].rolling(6,min_periods=1,center=True).median()
    dfa["sm1216"] = dfa[["c1216"]].rolling(6,min_periods=1,center=True).median()
    dfa["sm1620"] = dfa[["c1620"]].rolling(6,min_periods=1,center=True).median()
    dfa["sm2000"] = dfa[["c2000"]].rolling(6,min_periods=1,center=True).median()
    dfa["q0004"] = dfa[["c0004"]].rolling(6,min_periods=3,center=True).quantile(0.99) # smooth 4.5 weeks each side    
    dfa["q0408"] = dfa[["c0408"]].rolling(6,min_periods=3,center=True).quantile(0.99)
    dfa["q0812"] = dfa[["c0812"]].rolling(6,min_periods=3,center=True).quantile(0.99)
    dfa["q1216"] = dfa[["c1216"]].rolling(6,min_periods=3,center=True).quantile(0.99)
    dfa["q1620"] = dfa[["c1620"]].rolling(6,min_periods=3,center=True).quantile(0.99)
    dfa["q2000"] = dfa[["c2000"]].rolling(6,min_periods=3,center=True).quantile(0.99)

    dfa["sm0004"] = dfa[["c0004"]].apply(lambda x: savgol_filter(x,11,1)) # smooth 4.5 weeks each side    
    dfa["sm0408"] = dfa[["c0408"]].apply(lambda x: savgol_filter(x,9,1))
    dfa["sm0812"] = dfa[["c0812"]].apply(lambda x: savgol_filter(x,9,1))
    dfa["sm1216"] = dfa[["c1216"]].apply(lambda x: savgol_filter(x,9,1))
    dfa["sm1620"] = dfa[["c1620"]].apply(lambda x: savgol_filter(x,9,1))
    dfa["sm2000"] = dfa[["c2000"]].apply(lambda x: savgol_filter(x,9,1))
    

    # plot correction factors
    _,ax = plt.subplots(1,1,figsize=(15,8))
    (dfa.iloc[:,2::2]).plot(ax=ax)
    (dfa.iloc[:,3::2]).plot(ls="--", lw=1, color="k",ax=ax)
    (dfa.iloc[:,14:]).plot(ls="--", lw=1, color="r",ax=ax)
    
    _,ax = plt.subplots(1,1,figsize=(8,8))
    act["c1216"].hist(bins=50,ax=ax)
    
    
    act["c1216"].quantile(0.99)
    #mediaan en 99th percentile standardisation

        
    #     # decision for correction = standardisation according to [0,2*avg per week]
    #     df = (
    #           data.loc[data["t4"]==h,["farm_id","animal_id","parity","dim","date",
    #                                   "measured_on","year","week","t4","activity_total"]]
    #           .reset_index(drop=1)
    #           .merge(summary[["year","week","min","max"]], on=["year","week"])
    #           )
    #     dfa = pd.concat([df,dfa])
        
    # del h,df,summary
    
    # # correct ts for "min" and "max"
    # dfa["act"] = (dfa["activity_total"]-dfa["min"])/(dfa["max"]-dfa["min"])
    # dfa = dfa.sort_values(by=["animal_id","measured_on"]).reset_index(drop=1)  
    # dfa = dfa.drop(columns={"min","max"})


    # test = dfa.loc[dfa["year"]==2015,:]
    # _,ax = plt.subplots(2,1,figsize=(15,8))
    # sns.lineplot(data=test,x="measured_on",y="activity_total",hue="animal_id",estimator=None, ax=ax[0])
    # sns.lineplot(data=test,x="measured_on",y="act",hue="animal_id",estimator=None, ax=ax[1])
    # sns.lineplot(data=test,x="measured_on",y="act",estimator="median", ax=ax[1])

    


"""
- data per hour -> corrected for herd average + normalise 
    ==> trend in stats over time
- data per day = aggregation of data per hour
"""
# summary and smoothed per time window
# _,ax = plt.subplots(1,1,figsize=(15,8))
# dfa = pd.DataFrame([],columns=['farm_id', 'animal_id', 'parity', 'dim', 'date', 'measured_on', 'year',
#        'week', 't4', 'activity_total', 'min', 'max'])
# for h in [0,4,8,12,16,20]:
    
#     # summary per time window of 4 hours, summarized per week
#     summary = (
#            data.loc[data["t4"]==h,["year","week","activity_total"]]
#            .groupby(by = ["year","week"]).agg({"activity_total" : ["min","max","mean"]})
#            ).sort_values(by = ["year","week"]).reset_index()
#     summary.columns = summary.columns.droplevel()
#     summary.columns = ["year","week","min","max","mean"]
#     summary["act_sm"] = summary[["mean"]].apply(lambda x: savgol_filter(x,9,2)) # smooth 4.5 weeks each side    
#     summary = summary.reset_index()
#     summary["min"] = 0
#     summary["max"] = 2*summary["mean"]
    
#     # decision for correction = standardisation according to [0,2*avg per week]
#     df = (
#           data.loc[data["t4"]==h,["farm_id","animal_id","parity","dim","date",
#                                   "measured_on","year","week","t4","activity_total"]]
#           .reset_index(drop=1)
#           .merge(summary[["year","week","min","max"]], on=["year","week"])
#           )
#     dfa = pd.concat([df,dfa])
    
# del h,df,summary

# # correct ts for "min" and "max"
# dfa["act"] = (dfa["activity_total"]-dfa["min"])/(dfa["max"]-dfa["min"])
# dfa = dfa.sort_values(by=["animal_id","measured_on"]).reset_index(drop=1)  
# dfa = dfa.drop(columns={"min","max"})


# test = dfa.loc[dfa["year"]==2015,:]
# _,ax = plt.subplots(2,1,figsize=(15,8))
# sns.lineplot(data=test,x="measured_on",y="activity_total",hue="animal_id",estimator=None, ax=ax[0])
# sns.lineplot(data=test,x="measured_on",y="act",hue="animal_id",estimator=None, ax=ax[1])
# sns.lineplot(data=test,x="measured_on",y="act",estimator="median", ax=ax[1])

#%% exclude data with unclear breakpoints / mixed number of 
farms = [41,42,43,44,46,47,48,49,50,51,54,55,57,58,59,61,62,64,65,66,67,69] 
sel = pd.read_csv(os.path.join(path_data,"act_data_selection_breakpoints.txt"))
sel["start"] = pd.to_datetime(sel["start"], format="%d/%m/%Y")
sel["end"] = pd.to_datetime(sel["end"], format="%d/%m/%Y")

for farm in farms:    
    
    # load farm data 
    act = pd.read_csv(os.path.join(path_data,"activity_estruscorr_farm_" + str(farm) + ".txt"),
                      index_col=0)
    act = act.loc[act["farm_id"].isna()==False,:].sort_values(by=["animal_id","date"]).reset_index(drop=1)
    act["date"] = pd.to_datetime(act["date"], format = "%Y-%m-%d")
    
    # prepare plotting
    fig,ax = plt.subplots(nrows = 2, ncols = 1, figsize=(18,16),sharex=False)
    
    # delete farm data when not reliable in stats
    todelete = sel.loc[sel["farm"]==farm,:]
    # todelete = todelete.loc[todelete["start"].isna()==False,:]
    for lines in todelete.dropna().index.values:
        act.loc[(act["date"]>=todelete["start"][lines])&\
                      (act["date"]<=todelete["end"][lines]),"act_new"] = np.nan
        ax[0].fill_betweenx([act["act_new"].min(),act["act_new"].max()],
                            todelete["start"][lines],todelete["end"][lines],
                            color = "grey",alpha = 0.2)
        ax[1].fill_betweenx([-800,800],
                            todelete["start"][lines],todelete["end"][lines],
                            color = "grey",alpha = 0.2)
    
    # breakpoints
    n_bkps = todelete["breakpoints"].max()
    labels = n_bkps*["_"]
    
    # calculate stats
    sumstat = (act[["date","act_new"]].groupby(by = "date").agg({"act_new" : ["count","median","std",q10,q90]})).reset_index()
    sumstat.columns = sumstat.columns.droplevel()
    sumstat.columns = ["date","count",
                       "median","std",
                       "q10","q90"]
    sumstat.loc[sumstat["count"]==0,"count"]=np.nan
    first_idx = sumstat["median"].first_valid_index()
    last_idx = sumstat["median"].last_valid_index()
    # print(first_idx, last_idx)
    sumstat=sumstat.loc[first_idx:last_idx].reset_index(drop=1)
    
    # plot stats
    ax[0].fill_between(sumstat["date"],sumstat["q10"],sumstat["q90"], 
                     color = "mediumseagreen", alpha = 0.5)
    ax[0].plot(sumstat["date"],sumstat["median"], linewidth = 2, color = "teal")
    ax2 = ax[0].twinx()
    ax2.plot(sumstat["date"],sumstat["count"], linewidth = 0.5,ls='--', color = "purple")
    ax2.grid(False)     
    ax2.set_ylim([0,sumstat["count"].max()+20])
    ax[0].set_xlabel("date")
    ax2.set_ylabel("count - number of animals")
    ax[0].set_title("farm = "+str(farm)+ ", summary stats of activity data, corrected for estrus + no. of cows")
    ax[0].set_ylabel("median  - IQR of activity")

    # find exact breakpoints
    if n_bkps == 0:
        # correct to mean + std of 
        print(0,len(sumstat)-1)
        sumstat['rolling_median'] = sumstat["median"].rolling(100,center=True,min_periods=7).median()
        sumstat.loc[sumstat["median"].isna(),"rolling_median"]=np.nan
        labels = labels + ["_","10-90% IQR","median activity","rolling median"] 
    else:
        # find breakpoints - set breakpoints per farm based on visual assessment
        signal = sumstat.loc[sumstat["median"].isna() == False,["q10","median","q90"]].to_numpy()
        algo = rpt.Dynp(model="l1").fit(signal)
        result = algo.predict(n_bkps=n_bkps)  # find correct number of breaks
        
        #rpt.display(signal, result)
        print(result)
        brkpts = sumstat.loc[sumstat["median"].isna() == False].iloc[result[:-1]]["date"]
        resultrange = [0]+result
        labels = labels + ["_","10-90% IQR","median activity","break"] + (n_bkps-1)*["_"] +["rolling median"] 
    
        # plot breakpoints
        for i in range(0,len(brkpts)):
            print(brkpts.iloc[i])
            ax[0].set_ylim([ax[0].get_ylim()[0], ax[0].get_ylim()[1]])
            ax[0].plot([brkpts.iloc[i],brkpts.iloc[i]],
                       [ax[0].get_ylim()[0], ax[0].get_ylim()[1]],
                       color = 'royalblue',linestyle = '--',linewidth = 2)
            ax[1].plot([brkpts.iloc[i],brkpts.iloc[i]],
                       [-800,800],
                       color = 'royalblue',linestyle = '--',linewidth = 2)
        
        # calculate rolling median per period
        sumstat["rolling_median"] = np.nan
        for i in range(1,len(resultrange)):
             print(resultrange[i-1],resultrange[i])
             test = sumstat.loc[sumstat["median"].isna()==False].iloc[resultrange[i-1]:resultrange[i]-1]["median"].rolling(100,center=True,min_periods=7).median()
             sumstat.loc[test.index.values,"rolling_median"] = test
             del test
        
    # plot median rolling corrected
    ax[0].plot(sumstat["date"],
               sumstat["rolling_median"],
               color = "crimson",linewidth = 2, linestyle = '-')
    
    # add legend depending on npbkpts
    l=ax[0].legend(labels=labels,
                 bbox_to_anchor=(1, 1.3))
    ax[0].set_ylim([sumstat["q10"].min()-50,sumstat["q90"].max()+100])
       
    # calculate median correction
    act = act.merge(sumstat[["date","rolling_median"]])
    act["median_corr"] = act["act_new"]-act["rolling_median"]
    
    # reacalculate stats
    sumstat2 = (act[["date","median_corr"]].groupby(by = "date").agg({"median_corr" : ["median",q10,q90]})).reset_index()
    sumstat2.columns = sumstat2.columns.droplevel()
    sumstat2.columns = ["date",
                       "median",
                       "q10","q90"]
    
    # plot rolling median corrected
    ax[1].fill_between(sumstat2["date"],sumstat2["q10"],sumstat2["q90"], 
                     color = "goldenrod", alpha = 0.5)
    ax[1].plot(sumstat2["date"],sumstat2["median"], linewidth = 2, color = "goldenrod")
    ax[1].set_ylim(sumstat2["q10"].min()-50,sumstat2["q90"].max()+50)
    ax[1].set_xlabel("date")
    ax[1].set_ylabel("rolling median corrected activity")
    ax[1].set_title("corrected activity time series")
    
    # plot thi
    weather = pd.read_csv(os.path.join(path_data,"newweather_" + str(farm)+".txt"),
                       index_col=0)
    weather["time"] = pd.to_datetime(weather["time"], format = "%Y-%m-%d %H:%M:%S")
    weather["date"] = pd.to_datetime(weather["time"].dt.date, format = "%Y-%m-%d") 
    weather = weather.loc[(weather["date"] >= act["date"].min()) & \
                          (weather["date"] <= act["date"].max()),:].reset_index(drop=1)
    # calculate THI = (1.8 × Tmean + 32) − [(0.55 − 0.0055 × RHmean) × (1.8 × Tmean − 26)]
    weather["thi"] = 1.8 * weather["temp"] + 32 - \
                    ((0.55 - 0.0055 * weather["rhum"]) * \
                      (1.8 * weather["temp"] - 26))
    weather = weather[["date","temp","rhum","thi"]].groupby(by = ["date"]).mean().reset_index()
    
    # add to plot : mean THI and max daily temperature
    ax2 = ax[1].twinx()
    ax2.plot(weather["date"], weather["thi"],
            color = "navy",lw=0.5,) 
    ax2.grid(False)   
    ax2.set_ylim([-150,85])
    ax2.set_ylabel("THI")
    ax2.set_yticks([0,25,50,75])
    ax[0].set_xlim(act["date"].min(),act["date"].max())
    ax[1].set_xlim(act["date"].min(),act["date"].max())

    # prepare dataset for saving
    act["rm0004"] = act["f0004"] * act["rolling_median"]
    act["rm0408"] = act["f0408"] * act["rolling_median"]
    act["rm0812"] = act["f0812"] * act["rolling_median"]
    act["rm1216"] = act["f1216"] * act["rolling_median"]
    act["rm1620"] = act["f1620"] * act["rolling_median"]
    act["rm2000"] = act["f2000"] * act["rolling_median"]
    
    # save data
    act.to_csv(os.path.join(path,"results","data","new","act_preprocessed_"+str(farm)+".txt"))
    
    # save figure
    plt.savefig(os.path.join(path,"results","activity","median_corrected_activity_farm_" + str(farm) + ".tif"))

#%% method = with breakpoints, correct per week for 
farms = [41,42,43,44,46,47,48,49,50,51,54,55,57,58,59,61,62,64,65,66,67,69] 


for farm in farms:    
    
    # load farm data 
    act = pd.read_csv(os.path.join(path_data,"activity_estruscorr_farm_" + str(farm) + ".txt"),
                      index_col=0)
    act = act.loc[act["farm_id"].isna()==False,:].sort_values(by=["animal_id","date"]).reset_index(drop=1)
    act["date"] = pd.to_datetime(act["date"], format = "%Y-%m-%d")
    act["month"] = act["date"].dt.month
    
    ###########################################################################
    # correction activity stats attempt #25000                                #
    #   - identify where mean and std are not the consistent in their         #
    #     proportion                                                          #
    #   - delete those datums                                                 #
    #   - correct stats based on median of the day for that herd              #
    ###########################################################################      
    
    #---------------------calculate stats--------------------------------------
    sumstat = (act[["date","act_new"]].groupby(by = "date").agg({"act_new" : ["count","mean","std","median",q10,q90]})).reset_index()
    sumstat.columns = sumstat.columns.droplevel()
    sumstat.columns = ["date","count","mean",
                                      "std","median",
                                      "q10","q90"]
    sumstat["date"] = pd.to_datetime(sumstat["date"], format = "%Y-%m-%d")
    sumstat["smoothedmean"] = sumstat["mean"].rolling(30,center=True,min_periods=1).median()
    sumstat["smoothedmedian"] = sumstat["median"].rolling(30,center=True,min_periods=1).median()
    sumstat["diff"] = (sumstat["smoothedmean"]-sumstat["smoothedmedian"]).abs()
    
    #-----------------plot stats---------------------------------------------
    _,ax = plt.subplots(1,1,figsize=(6,4))
    sumstat["diff"].hist(bins=50)
    
    T=sumstat["diff"].quantile(0.95)  #threshold
    
    _,ax = plt.subplots(2,1,figsize=(15,8),sharex=True)
    ax[1].plot(sumstat["date"],sumstat["diff"])
    ax[0].plot(sumstat["date"],sumstat[["smoothedmean","smoothedmedian","std"]])
    sns.lineplot(data=sumstat.loc[sumstat["diff"]>T],
                 x="date",
                 y="diff",
                 linewidth=0,marker="*",color="r",
                 ax=ax[1])
    ax[0].set_title("farm = " + str(farm))
    (sumstat.iloc[:,2:]).plot(ax=ax)
    
    (sumstat.iloc[:,3::2]).plot(ls="--", lw=1, color="k",ax=ax)
    (sumstat.iloc[:,14:]).plot(ls="--", lw=1, color="r",ax=ax)
    
    
    # remove general trend in the data > subtract median of the day
    sumstat = (act[["date","act_new"]].groupby(by = "date").agg({"act_new" : ["median","std"]})).reset_index()
    sumstat.columns = sumstat.columns.droplevel()
    sumstat.columns = ["date","median","std"]
    sumstat["date"] = pd.to_datetime(sumstat["date"], format = "%Y-%m-%d")
    act = act.merge(sumstat)
    act["med_corr"] = act["act_new"] - act["median"]

    # plot median corrected activity
    _,ax = plt.subplots(1,1,figsize=(15,8))
    sns.lineplot(data=act,x="date",y="med_corr",estimator=None,lw=0.2,hue="std")
    
    # from this plot, clearly the std in summer is higher than in winter. 
    def iqr(x):
        return x.quantile(0.90)-x.quantile(0.10)
    # for individual clustering - only base on std of winter for correction
    sumstat = (
               act.loc[act["month"].isin([1,2,3,4,5,10,11,12]),["animal_id","parity","med_corr"]]
               .groupby(by = ["animal_id","parity"])
               .agg({"med_corr" : ["std",iqr]})).reset_index()
    sumstat.columns = sumstat.columns.droplevel()
    sumstat.columns = ["animal_id","parity","std","iqr"]
    act = act.merge(sumstat)
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2)
    cluster_labels = kmeans.fit_predict(sumstat.iloc[:,2:])
    sumstat["cluster"] = cluster_labels
    
    _,ax = plt.subplots(1,1,figsize=(7,7))
    sns.scatterplot(data=sumstat,x="std",y="iqr",hue="cluster")
    
    
    
    cl=act[["animal_id","parity","med_corr"]].groupby(["animal_id","parity"]).agg({"med_corr":["mean","std"]})
    cowlac = act[["animal_id","parity","date"]].sort_values(by=["animal_id","parity","date"]).drop_duplicates(["animal_id","parity"]).reset_index(drop=1)
    kmeans = KMeans(n_clusters=3)
    cluster_labels = kmeans.fit_predict(sumstat.iloc[:,:])
    
    
    
    
    
    # ---------------- summarize and plot stats activity ------------------
    def q10(x):
        return x.quantile(0.1)
    def q90(x):
        return x.quantile(0.9)
    sumstat = (act[["date","act_new"]].groupby(by = "date").agg({"act_new" : ["count","mean","std","median",q10,q90]})).reset_index()
    sumstat.columns = sumstat.columns.droplevel()
    sumstat.columns = ["date","count","mean",
                                      "std","median",
                                      "q10","q90"]
    sumstat["date"] = pd.to_datetime(sumstat["date"], format = "%Y-%m-%d")

    
    #-----------------plot stats---------------------------------------------
    # plot correction factors
    _,ax = plt.subplots(1,1,figsize=(15,8))
    (sumstat.iloc[:,2:]).plot(ax=ax)
    (sumstat.iloc[:,3::2]).plot(ls="--", lw=1, color="k",ax=ax)
    (sumstat.iloc[:,14:]).plot(ls="--", lw=1, color="r",ax=ax)
    
    #-------------------------------correct individual ts---------------------
    # merge sumstat["median"] with act
    act["date"] = pd.to_datetime(act["date"], format = "%Y-%m-%d")
    act = act.merge(sumstat[["date","median"]], on="date")
    act["act_medcorr"] = act["act_new"]-act["median"]
    
    from sklearn.cluster import KMeans
    cl=act[["animal_id","parity","act_new"]].groupby(["animal_id","parity"]).agg({"act_new":["mean","std"]})
    cowlac = act[["animal_id","parity","date"]].sort_values(by=["animal_id","parity","date"]).drop_duplicates(["animal_id","parity"]).reset_index(drop=1)
    kmeans = KMeans(n_clusters=3)
    cluster_labels = kmeans.fit_predict(cl.iloc[:,:])
    
    cowlac["mean"] = cl["act_new","mean"].values
    cowlac["std"] = cl["act_new","std"].values
    cowlac["cluster"] = cluster_labels    
    
    test = (act.iloc[:,0:-1]).merge(cowlac[["animal_id","parity","cluster"]],on=["animal_id","parity"])
    
    _,ax = plt.subplots(1,1,figsize=(15,8))
    sns.lineplot(data=test,x="date",y="act_new",hue="cluster",estimator=None,linewidth=0.2)
    
    _,ax = plt.subplots(1,1,figsize=(8,8))
    sns.scatterplot(data=cowlac, x="std",y="mean",hue="cluster")
    
    
    
    # --------------------- correct differences in data stats without removing information
    # set number of breakpoints per farm
    no_breaks = {"30":0,
                 "31":0,
                 "33":0,
                 "34":0,
                 "35":1,
                 "38":1,
                 "39":0,
                 "40":3,
                 "43":2,
                 "44":2,
                 "45":2,  
                 "46":1,
                 "47":0,
                 "48":1
                 }        
    n_bkps = no_breaks[str(farm)]
    
    # plot stats ----  %matplotlib qt
    fig,ax = plt.subplots(nrows = 2, ncols = 1, figsize=(15,10))
    
    ax[0].fill_between(sumstat["date"],sumstat["q10"],sumstat["q90"], 
                     color = "mediumseagreen", alpha = 0.5)
    ax[0].plot(sumstat["date"],sumstat["median"], linewidth = 2, color = "teal")
    
    ax[0].plot(sumstat["date"],sumstat["median"], linewidth = 1, color = "red", ls=":")
    ax[0].plot(sumstat["date"],sumstat["q10"], linewidth = 1, color = "red", ls=":")
    ax[0].plot(sumstat["date"],sumstat["q90"], linewidth = 1, color = "red", ls=":")
    
    
    
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







#%% METHOD 2 = with breakpoints
    for farm in farms:    
        
        
        
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
        no_breaks = {"30":0,
                     "31":0,
                     "33":0,
                     "34":0,
                     "35":1,
                     "38":1,
                     "39":0,
                     "40":3,
                     "43":2,
                     "44":2,
                     "45":2,  
                     "46":1,
                     "47":0,
                     "48":1
                     }        
        n_bkps = no_breaks[str(farm)]
        
        # plot stats ----  %matplotlib qt
        fig,ax = plt.subplots(nrows = 2, ncols = 1, figsize=(15,10))
        
        ax[0].fill_between(sumstat["date"],sumstat["q10"],sumstat["q90"], 
                         color = "mediumseagreen", alpha = 0.5)
        ax[0].plot(sumstat["date"],sumstat["median"], linewidth = 2, color = "teal")
        
        ax[0].plot(sumstat["date"],sumstat["median"], linewidth = 1, color = "red", ls=":")
        ax[0].plot(sumstat["date"],sumstat["q10"], linewidth = 1, color = "red", ls=":")
        ax[0].plot(sumstat["date"],sumstat["q90"], linewidth = 1, color = "red", ls=":")
        
        
        
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



