# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:12:42 2023

@author: u0084712
"""



# visualise individual activity curves


import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataanalysis")
os.chdir(path)


#%% import packages

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as savgol
# from datetime import date
# import openpyxl

#%% file path

path_data = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","datapreprocessing",
                    "results")

# farm selected
farms = [30,31,33,34,35,38,39,40,43,44,45,46,48] 


#%% read and visualise data

for farm in farms:
    # activity
    act = pd.read_csv(os.path.join(path_data,"data", "act_preprocessed_" 
                                    + str(farm) + ".txt"), 
                      usecols = ['farm_id', 'animal_id', 'parity', 'date', 'act_corr'])
    act["date"] = pd.to_datetime(act["date"],format='%Y-%m-%d')
    act = act.rename(columns = {'act_corr' : 'activity'})
    
    # read milk yield data to add dim / parity information
    milk = pd.read_csv(os.path.join(path_data,"data", "milk_preprocessed_" 
                                    + str(farm) + ".txt"),
                       usecols = ["animal_id","parity","date","dim"])
    milk["date"] = pd.to_datetime(milk["date"],format='%Y-%m-%d')
    
    # merge act and milk, delete milk
    act = pd.merge(act, milk, how = "inner",on = ["animal_id","parity","date"])
    del milk

    #-------------------------- visualisation per dim/parity ------------------
    act["pargroup"] = (
        (pd.concat([act["parity"],pd.DataFrame(3*np.ones((len(act),1)))], axis = 1))
        .min(axis = 1)
        )
    sumstat = (
        (act[["pargroup","dim","activity"]].groupby(by = ["pargroup","dim"])
          .agg({"activity" : ["count","mean","std"]})).reset_index()
        )
    sumstat.columns = sumstat.columns.droplevel()
    sumstat.columns = ["pargroup","dim","count","mean","std"]

    fig,ax = plt.subplots(3,1,figsize = (16,11))
    for parity in sumstat["pargroup"].drop_duplicates().astype(int):
        print(parity)
        subset = sumstat.loc[sumstat["pargroup"]==parity,:]
        ax[parity-1].set_ylim(0,1300)
        ax[parity-1].fill_between(subset["dim"],
                                  subset["mean"]-2*subset["std"],
                                  subset["mean"]+2*subset["std"],
                                  color = "palevioletred", alpha = 0.5)
        ax[parity-1].plot(subset["dim"],
                          subset["mean"],
                          linewidth = 2, color = "crimson")
        ax[parity-1].set_xlim(0,subset["dim"].max())
        ax[parity-1].set_ylim(-(subset["mean"]+2.2*subset["std"]).max(),(subset["mean"]+2.2*subset["std"]).max())
        if parity == 1:
            ax[parity-1].set_title("farm  " + str(farm) + ", standardised activity, parity = " + \
                                    str(round(parity)), fontsize = 14)
        else:
            ax[parity-1].set_title("parity = " + str(parity))
        if parity == 3:
            ax[parity-1].set_xlabel("dim [d]")
        ax[parity-1].set_ylabel("daily activity, mean+2*std, [unit]", color = "red")
        ax[parity-1].plot([sumstat["dim"].min(), sumstat["dim"].max()],
                          [sumstat["mean"].mean(), sumstat["mean"].mean()],
                          color = "black",linestyle = "--",lw = 1.5)
        #todo: legend and grid
        ax2 = ax[parity-1].twinx()
        ax2.plot(subset["dim"],
                  subset["count"],
                  linewidth = 2, color = "blue")
        ax2.grid(False)
        ax2.set_ylabel("number of animals", color = "blue")
    plt.savefig(os.path.join(path,"results","activity","corr_activity_stats_dim_" + str(farm) + ".tif"))
    plt.close()
    
    #--------------------------- individual curves ----------------------------
    fig,ax = plt.subplots(3,1,figsize = (16,11), sharex = True)
    act = act.sort_values(by = ["animal_id","parity","dim"]).reset_index(drop=1)

    # first parity
    cowlac = act.loc[act["pargroup"]==1,["animal_id","parity"]].drop_duplicates().reset_index(drop=1)
    par1col = sns.color_palette(palette = "flare",
                                n_colors = len(cowlac))
    for i in range(0,len(cowlac)):
        print(cowlac["animal_id"][i],cowlac["parity"][i])
        df = act.loc[(act["animal_id"] == cowlac["animal_id"][i]) & \
                     (act["parity"] == cowlac["parity"][i]),["dim","activity"]]
        ax[0].plot(df["dim"],df["activity"],axes = ax[0],color = par1col[i],
                   linewidth = 0.4)
    ax[0].set_xlim([0,400])
    ax[0].set_title("farm = " + str(farm) + ", individual activity curves parity 1")
    ax[0].set_ylabel("activity [unit]")
    
    # parity 2
    cowlac = act.loc[act["pargroup"]==2,["animal_id","parity"]].drop_duplicates().reset_index(drop=1)
    par2col = sns.color_palette("ch:s=.25,rot=-.25",
                                n_colors = len(cowlac))
    for i in range(0,len(cowlac)):
        print(cowlac["animal_id"][i],cowlac["parity"][i])
        df = act.loc[(act["animal_id"] == cowlac["animal_id"][i]) & \
                     (act["parity"] == cowlac["parity"][i]),["dim","activity"]]
        ax[1].plot(df["dim"],df["activity"],axes = ax[1],color = par2col[i],
                   linewidth = 0.4)
    ax[1].set_xlim([0,400])
    ax[1].set_title("individual activity curves parity 2")
    ax[1].set_ylabel("activity [unit]")
    
    # parity 3
    cowlac = act.loc[act["pargroup"]== 3,["animal_id","parity"]].drop_duplicates().reset_index(drop=1)
    par3col = sns.color_palette("light:#5A9",
                                n_colors = len(cowlac))
    for i in range(0,len(cowlac)):
        print(cowlac["animal_id"][i],cowlac["parity"][i])
        df = act.loc[(act["animal_id"] == cowlac["animal_id"][i]) & \
                     (act["parity"] == cowlac["parity"][i]),["dim","activity"]]
        ax[2].plot(df["dim"],df["activity"],axes = ax[2],color = par3col[i],
                   linewidth = 0.4)
    ax[2].set_xlim([0,400])
    ax[2].set_title("individual activity curves parity 3+")
    ax[2].set_xlabel("dim [d]")
    ax[2].set_ylabel("activity [unit]")
    plt.savefig(os.path.join(path,"results","activity","corr_activity_individual_dim_" + str(farm) + ".tif"))
    plt.close()
        
            
#%% Activity- individual curves: plot and visualise
            
for farm in farms:
    # activity
    act = pd.read_csv(os.path.join(path_data,"data", "act_preprocessed_" 
                                    + str(farm) + ".txt"), 
                      usecols = ['farm_id', 'animal_id', 'parity', 'date', 'act_corr'])
    act["date"] = pd.to_datetime(act["date"],format='%Y-%m-%d')
    act = act.rename(columns = {'act_corr' : 'activity'})
    act["pargroup"] = (
        (pd.concat([act["parity"],pd.DataFrame(3*np.ones((len(act),1)))], axis = 1))
        .min(axis = 1)
        )
    
    # read milk yield data to add dim / parity information
    milk = pd.read_csv(os.path.join(path_data,"data", "milk_preprocessed_" 
                                    + str(farm) + ".txt"),
                       usecols = ["animal_id","parity","date","dim","dmy"])
    milk["date"] = pd.to_datetime(milk["date"],format='%Y-%m-%d')

    
    # merge act and milk, delete milk
    act = pd.merge(act, milk, how = "inner",on = ["animal_id","parity","date"])
    
    # add pargroup to milk
    milk["pargroup"] = (
        (pd.concat([milk["parity"],pd.DataFrame(3*np.ones((len(act),1)))], axis = 1))
        .min(axis = 1)
        )
    
    # weather information
    wea = pd.read_csv(os.path.join(path_data,"data", "weather_farm_" 
                                    + str(farm) + ".txt"), 
                      usecols = ['date', 'temp', 'thi'])
    wea["date"] = pd.to_datetime(wea["date"],format='%Y-%m-%d')
    wea = wea.loc[(wea["date"] >= act["date"].min()) & \
                  (wea["date"] <= act["date"].max()),:]

    # select individual curves for plotting + quantify peaks
    cowlac = act[["animal_id","parity"]].drop_duplicates().reset_index(drop=1)
    randanimals = cowlac.sample(12)[["animal_id","parity"]].index.values  # random plots
    sns.set_style("whitegrid")
    for i in range(0,len(cowlac)):
        # plot if in randomly selected 
        if i in randanimals:
            print(i)
            df = act.loc[(act["animal_id"] == cowlac["animal_id"][i]) & \
                     (act["parity"] == cowlac["parity"][i]),["dim","date","activity","pargroup"]]
            df = df.sort_values(by = "date").reset_index(drop=1)
                
            # visualise trend
            df["act_sm"] = savgol(df["activity"],7,1)
            
            # milk
            dfm = milk.loc[(milk["animal_id"] == cowlac["animal_id"][i]) & \
                            (milk["parity"] >= cowlac["parity"][i]),:]


            # prepare figure
            fig, ax = plt.subplots(2,1,figsize = (15,9), sharex = True)
            
            # plot THI
            df3 = wea.loc[(wea["date"] >= df["date"].min()) & \
                          (wea["date"] <= df["date"].max()),:]
            ax3 = ax[0].twinx()
            ax3.grid(False)
            ax3.plot(df3["date"],df3["thi"], linestyle = "-",linewidth = 1.5,
                       color = "crimson")
            ax3.set_ylim(-20,df3["thi"].max()+5)
            ax3.fill_between(df3["date"],68,df3["thi"].max()+5, color = "crimson",
                             alpha = 0.2)
            ax3.set_ylabel("thi")
            
            # plot cow behaviour            
            ax[0].grid(True)
            ax2 = ax[0].twiny()
            ax2.plot(df["dim"],df["activity"], linestyle = "-",linewidth = 0,
                       marker= "s", markersize = 0,
                       color = "white" )
            ax2.grid(False)
            ax2.set_xlim([df["dim"].min(),df["dim"].max()])
            ax[0].plot(df["date"],df["activity"], linestyle = "-",linewidth = 1,
                       marker= "s", markersize = 2.3,
                       color = "teal" )
            ax[0].set_xlim([df["date"].min()-pd.Timedelta(1, unit = "d"),
                         df["date"].max()+pd.Timedelta(1, unit = "d")])
            
            # plot smoothed 1st order Savitsky-Golay filter window 7d
            ax[0].plot(df["date"],df["act_sm"], linestyle = "-",linewidth = 0.8,
                       color = "blue")
            ax[0].set_title("farm = " + str(farm) + ", activity of cow " + \
                            str(cowlac["animal_id"][i]) +" in parity " + str(round(cowlac["parity"][i])))
            ax2.set_xlabel("dim")
            ax[0].set_xlabel("date") 
            ax[0].set_ylabel("activity")

            # plot herd avg + std
            df2 = act.loc[(act["date"] >= df["date"].min()) & \
                          (act["date"] <= df["date"].max()),["date","activity"]]
            df2 = df2.sort_values(by = "date").reset_index(drop=1)
            df2 = df2.groupby(by = "date").agg({"activity": ["mean","std"]}).reset_index()
            df2.columns = df2.columns.droplevel()
            df2.columns = ["date","mean","std"]   
                        
            # plot herd behaviour
            ax[0].fill_between(df2["date"],df2["mean"]-df2["std"],df2["mean"]+df2["std"],
                       linewidth = 0.1, color = "cornflowerblue", alpha=0.2)

            # set legends
            ax3.set_zorder(-1)
            ax[0].set_facecolor((1,1,1,0))
            ax[0].legend(labels = ["standardised activity","trend","herd [avg+std]"], 
                      facecolor = "white", loc = "lower right")
            ax3.legend(labels = ["thi", 'thi >= 68'], bbox_to_anchor=(1.03, 1), 
                       loc = 'upper left', borderaxespad=0)

            # plot milk production
            ax[1].plot(dfm["date"], dfm["dmy"],linestyle = "-",linewidth = 1,
                       marker= "s", markersize = 2.3,
                       color = "indigo")
            ax[1].set_title("daily milk production")
            ax[1].set_xlabel("date")
            ax[1].set_ylabel("dmy [kg]")
            
            # plot herd production / parity
            dfm2 = milk.loc[(milk["date"] >= df["date"].min()) & \
                          (milk["date"] <= df["date"].max()) & \
                          (milk["pargroup"] == df["pargroup"].iloc[0]),["date","dmy"]]
            ncows = len((milk.loc[(milk["date"] >= df["date"].min()) & \
                          (milk["date"] <= df["date"].max()) & \
                          (milk["pargroup"] == df["pargroup"].iloc[0]),["animal_id","parity"]]).drop_duplicates())
            dfm2 = dfm2.sort_values(by = "date").reset_index(drop=1)
            dfm2 = dfm2.groupby(by = "date").agg({"dmy": ["mean","std"]}).reset_index()
            dfm2.columns = dfm2.columns.droplevel()
            dfm2.columns = ["date","mean","std"]

            # plot herd production level
            ax[1].fill_between(dfm2["date"],dfm2["mean"]-dfm2["std"],dfm2["mean"]+dfm2["std"],
                       linewidth = 0.1, color = "plum", alpha=0.2)
            ax[1].legend(["dmy","pargroup dmy"])
            ax[1].set_title("daily milk production, ncows pargroup = " + str(ncows))

            # save plots
            plt.savefig(os.path.join(path,"results","activity",
               "example_ind_activity_farm_" + str(farm) + "_cow_" + str(cowlac["animal_id"][i]) + "_withmilk.tif"))
            plt.close()




#%% cow 49149 farm 48

farm = 48
animal = 49149
parity = 3



# activity
act = pd.read_csv(os.path.join(path_data,"data", "act_preprocessed_" 
                                + str(farm) + ".txt"), 
                  usecols = ['farm_id', 'animal_id', 'parity', 'date', 'act_corr'])
act["date"] = pd.to_datetime(act["date"],format='%Y-%m-%d')
act = act.rename(columns = {'act_corr' : 'activity'})

# read milk yield data to add dim / parity information
milk = pd.read_csv(os.path.join(path_data,"data", "milk_preprocessed_" 
                                + str(farm) + ".txt"),
                   usecols = ["animal_id","parity","date","dim", "dmy"])
milk["date"] = pd.to_datetime(milk["date"],format='%Y-%m-%d')

# merge act and milk, delete milk
act = pd.merge(act, milk, how = "inner",on = ["animal_id","parity","date"])



# select activity and milk
df = act.loc[(act["animal_id"] == animal) & \
             (act["parity"] >= parity),["dim","date","activity"]]
df = df.sort_values(by = "date").reset_index(drop=1)

dfm = milk.loc[(milk["animal_id"] == animal) & \
                (milk["parity"] >= parity),:]

# prepare figure
fig, ax = plt.subplots(2,1,figsize = (15,9), sharex = True)

# plot THI
df3 = wea.loc[(wea["date"] >= df["date"].min()) & \
              (wea["date"] <= df["date"].max()),:]
ax3 = ax[0].twinx()
ax3.grid(False)
ax3.plot(df3["date"],df3["thi"], linestyle = "-",linewidth = 1.5,
           color = "crimson")
ax3.set_ylim(-20,df3["thi"].max()+5)
ax3.fill_between(df3["date"],68,df3["thi"].max()+5, color = "crimson",
                 alpha = 0.2)
ax3.set_ylabel("thi")



# plot cow behaviour            
ax[0].grid(True)
ax2 = ax[0].twiny()
ax2.plot(df["dim"],df["activity"], linestyle = "-",linewidth = 0,
           marker= "s", markersize = 0,
           color = "white" )
ax2.grid(False)
ax2.set_xlim([df["dim"].min(),df["dim"].max()])
ax[0].plot(df["date"],df["activity"], linestyle = "-",linewidth = 1,
           marker= "s", markersize = 2.3,
           color = "teal" )
ax[0].set_xlim([df["date"].min()-pd.Timedelta(1, unit = "d"),
             df["date"].max()+pd.Timedelta(1, unit = "d")])

# plot herd avg + std
df2 = act.loc[(act["date"] >= df["date"].min()) & \
              (act["date"] <= df["date"].max()),["date","activity"]]
df2 = df2.sort_values(by = "date").reset_index(drop=1)
df2 = df2.groupby(by = "date").agg({"activity": ["mean","std"]}).reset_index()
df2.columns = df2.columns.droplevel()
df2.columns = ["date","mean","std"]

# plot herd behaviour
ax[0].fill_between(df2["date"],df2["mean"]-df2["std"],df2["mean"]+df2["std"],
           linewidth = 0.1, color = "cornflowerblue", alpha=0.2)

# set legends
ax3.set_zorder(-1)
ax[0].set_facecolor((1,1,1,0))
ax[0].legend(labels = ["standardised activity","herd activity"], 
          facecolor = "white", loc = "lower right")
ax3.legend(labels = ["thi", 'thi >= 68'], bbox_to_anchor=(1.03, 1), 
           loc = 'upper left', borderaxespad=0)
ax[0].set_title("farm " + str(farm) + ", cow " + str(animal) + ", in parity " + str(parity))

# plot milk production
ax[1].plot(dfm["date"], dfm["dmy"],linestyle = "-",linewidth = 1,
           marker= "s", markersize = 2.3,
           color = "indigo")
ax[1].set_title("daily milk production")
ax[1].set_xlabel("date")
ax[1].set_ylabel("dmy [kg]")

dfm2 = milk.loc[(milk["date"] >= df["date"].min()) & \
              (milk["date"] <= df["date"].max()),["date","dmy"]]
dfm2 = dfm2.sort_values(by = "date").reset_index(drop=1)
dfm2 = dfm2.groupby(by = "date").agg({"dmy": ["mean","std"]}).reset_index()
dfm2.columns = dfm2.columns.droplevel()
dfm2.columns = ["date","mean","std"]

# plot herd production level
ax[1].fill_between(dfm2["date"],dfm2["mean"]-dfm2["std"],dfm2["mean"]+dfm2["std"],
           linewidth = 0.1, color = "plum", alpha=0.2)
ax[1].legend(["dmy","herd dmy"])


# save
plt.savefig(os.path.join(path,"results","activity",
   "example_ind_activity_farm_" + str(farm) + "_cow_" + str(animal) + "_withmilk.tif"))
plt.close()




#%%    
    # save dataset
    act["pargroup"] = (
        (pd.concat([act["parity"],pd.DataFrame(3*np.ones((len(act),1)))], axis = 1))
        .min(axis = 1)
        )
    act = act.reset_index(drop=1)
    act.to_csv(os.path.join(path,"data","act_selected_" + str(farm) + ".txt"))
    
    # plot corrected activity per parity over dim
    sumstat = (
        (act[["pargroup","dim","act_new"]].groupby(by = ["pargroup","dim"])
          .agg({"act_new" : ["count","mean","std"]})).reset_index()
        )
    sumstat.columns = sumstat.columns.droplevel()
    sumstat.columns = ["pargroup","dim","count","mean","std"]

    fig,ax = plt.subplots(3,1,figsize = (16,11))
    for parity in sumstat["pargroup"].drop_duplicates().astype(int):
        print(parity)
        subset = sumstat.loc[sumstat["pargroup"]==parity,:]
        ax[parity-1].set_ylim(0,1300)
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
            ax[parity-1].set_title("farm  " + str(farm) + ", activity, parity = " + \
                                    str(round(parity)), fontsize = 14)
        else:
            ax[parity-1].set_title("parity = " + str(parity))
        if parity == 3:
            ax[parity-1].set_xlabel("dim [d]")
        ax[parity-1].set_ylabel("daily activity, mean+2*std, [unit]")
        ax[parity-1].plot([sumstat["dim"].min(), sumstat["dim"].max()],
                          [sumstat["mean"].mean(), sumstat["mean"].mean()],
                          color = "black",linestyle = "--",lw = 1.5)
        #todo: legend and grid
        ax2 = ax[parity-1].twinx()
        ax2.plot(subset["dim"],
                  subset["count"],
                  linewidth = 2, color = "blue")
        ax2.grid(False)
        ax2.set_ylabel("number of animals")
    plt.savefig(os.path.join(path,"results","activity","actnew_stats_dim_" + str(farm) + ".tif"))
    plt.close()


#%% Activity- individual standardised curves - plot
            
for farm in farms:
    act = pd.read_csv(os.path.join(path_data,"activity", "act_preprocessed_" 
                                    + str(farm) + ".txt"), 
                      usecols = ['farm_id', 'animal_id', 'parity', 'date', 'activity'])
    act["date"] = pd.to_datetime(act["date"],format='%Y-%m-%d')
    act["pargroup"] = (
        (pd.concat([act["parity"],pd.DataFrame(3*np.ones((len(act),1)))], axis = 1))
        .min(axis = 1)
        )
    
    
    # read milk yield data to add dim / parity information
    milk = pd.read_csv(os.path.join(path_data,"milk", "milk_preprocessed_" 
                                    + str(farm) + ".txt"),
                       usecols = ["animal_id","parity","date","dim"])
    milk["date"] = pd.to_datetime(milk["date"],format='%Y-%m-%d')
    
    # merge act and milk, delete milk
    act = pd.merge(act, milk, how = "inner",on = ["animal_id","parity","date"])
    act = act.sort_values(by = ["animal_id","parity","date"])
    del milk

    # select individual curves for plotting + quantify peaks
    cowlac = act[["animal_id","parity"]].drop_duplicates().reset_index(drop=1)
    randanimals = cowlac.sample(10)[["animal_id","parity"]].index.values  # random plots
    sns.set_style("whitegrid")
    for i in range(0,len(cowlac)):
        
        # preprocess individual activity
        df = act.loc[(act["animal_id"] == cowlac["animal_id"][i]) & \
                     (act["parity"] == cowlac["parity"][i]),["dim","date","activity"]]
        
        df["act_sm"] = savgol(df["activity"],7,1)
        df["act_res"] = df["activity"] - df["act_sm"]
        # median absolute deviation in 9 days rolling window
        def mad(x):
            return np.median(np.fabs(x - np.median(x)))
        
        df["mad"] = df["act_res"].rolling(window = 9,center = True).apply(mad,raw=True)
        # threshold = at least 80 above smoothed
        df.loc[(df["mad"] < 20) | (df["mad"].isna()), "mad"] = 20
        df["mad_thres"] = df["act_sm"] + 4 * df["mad"]
        
        # set values to smoothed values when above threshold
        df["act_new"] = df["activity"]
        df.loc[df["activity"]>df["mad_thres"],"act_new"] = df["act_sm"]
        
        # add df to act
        act.loc[df.index.values,"act_new"] = df["act_new"]
        
        """------------------------ TEST SMOOTHERS ----------------------------
        # smooth with median smoother window 7 days
        df["act_sm"] = df["activity"].rolling(7).median()
        ax[0].plot(df["date"],df["act_sm"], linestyle = "-",linewidth = 0.8,
                   color = "purple" )
        # smooth with median smoother window 5 days
        df["act_sm"] = df["activity"].rolling(5).median()
        ax[0].plot(df["date"],df["act_sm"], linestyle = "-",linewidth = 0.8,
                   color = "deeppink" )
        
        # smooth with Savitsky-Golay filter 5 days, 2nd order
        df["act_sm"] = savgol(df["activity"],5,2)
        ax[0].plot(df["date"],df["act_sm"], linestyle = "-",linewidth = 0.8,
                   color = "deepskyblue" )
        # smooth with Savitsky-Golay filter 5 days, 2nd order
        df["act_sm"] = savgol(df["activity"],7,2)
        ax[0].plot(df["date"],df["act_sm"], linestyle = "-",linewidth = 0.8,
                   color = "lime" )
        --------------------------------------------------------------------"""
        
        # plot if in randomly selected 
        if i in randanimals:
            print(i)
            fig, ax = plt.subplots(2,1,figsize = (15,8), sharex = True)
            ax[0].grid(True)
            ax2 = ax[0].twiny()
            ax2.plot(df["dim"],df["activity"], linestyle = "-",linewidth = 0,
                       marker= "s", markersize = 0,
                       color = "white" )
            ax2.grid(False)
            ax2.set_xlim([df["dim"].min(),df["dim"].max()])
            ax[0].plot(df["date"],df["activity"], linestyle = "-",linewidth = 1,
                       marker= "s", markersize = 2.3,
                       color = "teal" )
            ax[0].set_xlim([df["date"].min(),df["date"].max()])
            
            # best smoother = 1st order Savitsky-Golay filter window 7d
            ax[0].plot(df["date"],df["act_sm"], linestyle = "-",linewidth = 1.2,
                       color = "blue")
            ax[0].set_title("farm = " + str(farm) + ", activity of cow " + \
                            str(cowlac["animal_id"][i]) +" in parity " + str(round(cowlac["parity"][i])))
            ax2.set_xlabel("dim")
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
            ax2 = ax[1].twiny()
            ax2.plot(df["dim"],df["activity"], linestyle = "-",linewidth = 0,
                       marker= "s", markersize = 0,
                       color = "white" )
            ax2.grid(False)
            ax2.set_xlim([df["dim"].min(),df["dim"].max()]) 
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
               "example_activity_farm_" + str(farm) + "_cow_" + str(cowlac["animal_id"][i]) + ".tif"))
            plt.close()
    
    # save dataset
    act["pargroup"] = (
        (pd.concat([act["parity"],pd.DataFrame(3*np.ones((len(act),1)))], axis = 1))
        .min(axis = 1)
        )
    act = act.reset_index(drop=1)
    act.to_csv(os.path.join(path,"data","act_selected_" + str(farm) + ".txt"))
    
    # plot corrected activity per parity over dim
    sumstat = (
        (act[["pargroup","dim","act_new"]].groupby(by = ["pargroup","dim"])
          .agg({"act_new" : ["count","mean","std"]})).reset_index()
        )
    sumstat.columns = sumstat.columns.droplevel()
    sumstat.columns = ["pargroup","dim","count","mean","std"]

    fig,ax = plt.subplots(3,1,figsize = (16,11))
    for parity in sumstat["pargroup"].drop_duplicates().astype(int):
        print(parity)
        subset = sumstat.loc[sumstat["pargroup"]==parity,:]
        ax[parity-1].set_ylim(0,1300)
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
            ax[parity-1].set_title("farm  " + str(farm) + ", activity, parity = " + \
                                    str(round(parity)), fontsize = 14)
        else:
            ax[parity-1].set_title("parity = " + str(parity))
        if parity == 3:
            ax[parity-1].set_xlabel("dim [d]")
        ax[parity-1].set_ylabel("daily activity, mean+2*std, [unit]")
        ax[parity-1].plot([sumstat["dim"].min(), sumstat["dim"].max()],
                          [sumstat["mean"].mean(), sumstat["mean"].mean()],
                          color = "black",linestyle = "--",lw = 1.5)
        #todo: legend and grid
        ax2 = ax[parity-1].twinx()
        ax2.plot(subset["dim"],
                  subset["count"],
                  linewidth = 2, color = "blue")
        ax2.grid(False)
        ax2.set_ylabel("number of animals")
    plt.savefig(os.path.join(path,"results","activity","actnew_stats_dim_" + str(farm) + ".tif"))
    plt.close()
    
    
#%%  model activity in function of thi

for farm in farms:
    # load data activity with act_new = corrected activity
    act = pd.read_csv(os.path.join(path,"data","act_selected_" + str(farm) + ".txt"),
                      index_col = 0)

    act["date"] = pd.to_datetime(act["date"],format='%Y-%m-%d')
    
    
    # load thi data
    thi = pd.read_csv(os.path.join(path,"data","weatherfeatures_" + str(farm) + ".txt"),
                     index_col = 0 )
    thi["date"] = pd.to_datetime(thi["date"],format='%Y-%m-%d')
    
    # merge thi with act on date
    data = pd.merge(act,thi, on = ["farm_id","date"],how="outer")
    data = data.sort_values(by = ["farm_id","animal_id","pargroup","dim"])
    # delete weather info when no cow data
    data = data.loc[data["animal_id"].isna() == False,:].reset_index(drop=1)
    
    data.loc[(data["dim"]) < 22, "ls"] = "0-21"
    data.loc[(data["dim"] >= 22) & \
             (data["dim"] < 61), "ls"] = "22-60"
    data.loc[(data["dim"] >= 61) & \
             (data["dim"] < 121), "ls"] = "61-120"
    data.loc[(data["dim"] >= 121) & \
             (data["dim"] < 201), "ls"] = "121-200"
    data.loc[(data["dim"] >= 201), "ls"] = ">200"
    
    # if interaction terms, standardise/scale for convergence + int
    data = data.loc[data["thi_avg"].isna()==False,:].reset_index(drop=1)
    data.thi_avg = round(data.thi_avg).astype(int)
    
    data["thi_std"] = (data.thi_avg - data.thi_avg.mean()) / data.thi_avg.std()
    
    
    # add a class for year season
    #  TODO: check if necessary!!!
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
    
    #☻ drop nas for modelling
    data = data.dropna().reset_index(drop=1)
    
    #--------------------------------------------------------------------------
    
    # activity new corrected for estrus as the Y variable of the model
    
    # md = smf.mixedlm("act_new ~ thi_std + C(ls) + C(pargroup) + thi_std*C(ls) + thi_std*C(parity) + C(ymclass)",
    #                  data=data,
    #                  groups = data["animal_id"],
    #                  re_formula = "~thi_std")
        
    # try including a quadratic term for capturing higher act with high thi values    
    md = smf.mixedlm("act_new ~ thi_std + np.power(thi_std, 2) +  np.power(thi_std, 3) + C(ls) + C(pargroup) + thi_std*C(ls) + thi_std*C(pargroup) + C(ymclass)",
                     data=data,
                     groups = data["animal_id"],
                     re_formula = "~thi_std")
    
    
    mdf = md.fit(method=["lbfgs"])
    
    print(mdf.summary())
    
    # correlation = cov / sqrt(varx)*sqrt(vary)
    print(
          "correlation random thi slope and intercept = " + \
          str(round(mdf.cov_re.Group.thi_std / (np.sqrt(mdf.cov_re.Group.Group)*(np.sqrt(mdf.cov_re.thi_std.thi_std))),3))
          )
    
    with open(os.path.join("results","activity","summary_" + str(farm) + ".txt"), 'w') as fh:
        fh.write(mdf.summary().as_text())
    with open(os.path.join("results","activity","randomcorrelation_" + str(farm) + ".txt"), 'w') as fh:
         fh.write(str(round(mdf.cov_re.Group.thi_std / (np.sqrt(mdf.cov_re.Group.Group)*(np.sqrt(mdf.cov_re.thi_std.thi_std))),3)))
        
    # fitted values
    data["fitted_lme"] = mdf.fittedvalues
    data["residual_lme"] = mdf.resid
    
    # check normality/variance and linearity of residuals
    ######## to make the residuals lie around zero with higher THI, a second and third order term was included
    ######## heteroscedastisticiy not 100% at the edges of thi, but that also has to do with amount of data
    ######## 
    fig,ax = plt.subplots(1,1,figsize = (12,6))
    sns.scatterplot(data = data,
                   x = "thi_avg", y = "residual_lme", style = "pargroup", hue = "pargroup",
                   palette="rocket", s = 6)
    ax.set_xlim(ax.get_xlim()[0],ax.get_xlim()[1])
    ax.plot([25,78],[0,0],"k--",lw = 1)
    ax.set_title("model residuals ifo of average thi, farm = "+str(farm))
    ax.set_ylabel("activity")
    ax.set_xlabel("thi")
    ax.legend(labels = ["1","2","3"],fontsize = "small")
    plt.savefig(os.path.join(path,"results","activity","mdl_res_vs_thi_farm_" + \
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
    ax[0].set_ylabel("act excl estrus - model residuals")
    sns.countplot(data = data,
                  x = "thi_avg", ax= ax[1])
    ax[1].set_xlim(-0.5,51.5)
    ax[1].set_title("number of observations at each thi, farm = "+str(farm))
    ax[1].set_xlabel("average thi")
    ax[1].set_ylabel("no. of observations")
    plt.savefig(os.path.join(path,"results","activity","mdl_res2_vs_thi_farm_" + \
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
    
    ax[2].plot(data["fitted_lme"],data["act_new"],
            lw = 0,
            color = "indigo", marker = "x",ms = 3)
    ax[2].plot([200,1600],[200,1600],color = "r",lw = 1.5,ls = "--")
    ax[2].set_xlim(200,1600)
    ax[2].set_ylim(200,1600)
    ax[2].set_xlabel("predicted (estrus corr) act [kg]")
    ax[2].set_ylabel("observed (estrus corr) act [kg]")
    ax[2].set_title("predicted vs. observed plot")
    plt.savefig(os.path.join(path,"results","activity","mdl_res_stats_farm_" + \
                             str(farm)+ ".tif"))
    plt.close()

    # summer visualisation when heatstress is relevant
    years = data["year"].drop_duplicates().sort_values().reset_index(drop=1)
    for year in years:
       
        # model residuals
        subset = data.loc[(data["month"]>0) & (data["month"]<=13) & \
                           (data["year"] == year),:].reset_index(drop=1)
        thi = subset[["date","thi_avg"]].groupby("date").mean().reset_index()
        if len(subset)>1000:
            ds = subset[["date","residual_lme","act_new"]].groupby(by="date").mean().reset_index()
            ds = ds.rename(columns = {"residual_lme" : "avg_res",
                                       "act_new":"avg_act"})
            sub = subset[["date","residual_lme","act_new"]].groupby(by="date").quantile(0.1).reset_index()
            sub = sub.rename(columns = {"residual_lme" : "q10_res",
                                       "act_new":"q10_act"})
            ds["q10_res"] = sub["q10_res"]
            ds["q10_act"] = sub["q10_act"]
            sub = subset[["date","residual_lme","act_new"]].groupby(by="date").quantile(0.9).reset_index()
            sub = sub.rename(columns = {"residual_lme" : "q90_res",
                                       "act_new":"q90_act"})
            ds["q90_res"] = sub["q90_res"]
            ds["q90_act"] = sub["q90_act"]
         
            fig, ax = plt.subplots(2,1,figsize = (16,12))
            ax[0].fill_between(ds["date"],
                            ds["q10_act"],ds["q90_act"],
                            alpha = 0.85, color = "mediumblue", lw= 1)
            ax[0].plot(ds["date"],ds["avg_act"], "lightcyan", lw = 1.5)
            ax[0].plot(ds["date"],ds["avg_act"].mean()*np.ones((len(ds),1)), "k--", lw = 1.5)
            ax[0].set_ylabel("daily activity (estrus corrected) mean+90%CI")
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
            ax[1].set_ylabel("residual activity - mean+90%CI")
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
            plt.savefig(os.path.join(path,"results","activity","mdl_act_res_year_" + str(int(year)) + "_farm_" + \
                                     str(farm)+ ".tif"))
            plt.close()
            
            # episodes of very high THI - plot separately
            # first and last month with high THI
            thihigh = subset.loc[subset["thi_avg"]>68,"month"]
            
            subset = data.loc[(data["month"]>=thihigh.min()) & (data["month"]<=thihigh.max()) & \
                              (data["year"] == year),:].reset_index()
            if len(subset)> 500:
                ds = subset[["date","residual_lme","act_new"]].groupby(by="date").mean().reset_index()
                ds = ds.rename(columns = {"residual_lme" : "avg_res",
                                          "act_new":"avg_act"})
                sub = subset[["date","residual_lme","act_new"]].groupby(by="date").quantile(0.1).reset_index()
                sub = sub.rename(columns = {"residual_lme" : "q10_res",
                                          "act_new":"q10_act"})
                ds["q10_res"] = sub["q10_res"]
                ds["q10_act"] = sub["q10_act"]
                sub = subset[["date","residual_lme","act_new"]].groupby(by="date").quantile(0.9).reset_index()
                sub = sub.rename(columns = {"residual_lme" : "q90_res",
                                          "act_new":"q90_act"})
                ds["q90_res"] = sub["q90_res"]
                ds["q90_act"] = sub["q90_act"]
           
                fig, ax = plt.subplots(2,1,figsize = (16,12))
                ax[0].fill_between(ds["date"],
                                ds["q10_act"],ds["q90_act"],
                                alpha = 0.85, color = "teal", lw= 1)
                ax[0].plot(ds["date"],ds["avg_act"], "lightcyan", lw = 1.5)
                ax[0].plot(ds["date"],ds["avg_act"].mean()*np.ones((len(ds),1)), "k--", lw = 1.5)
                ax[0].set_ylabel("daily activity (estrus corrected) mean+90%CI")
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
                ax[1].set_ylabel("residual activity - mean+90%CI")
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
                plt.savefig(os.path.join(path,"results","activity","mdl_act_res_summer_year_" + str(int(year)) + "_farm_" + \
                                         str(farm)+ ".tif"))
                plt.close()