# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:05:26 2024

@author: u0084712
-------------------------------------------------------------------------------

explore weather to describe the insults cow in Western Europe are submitted to

- selected weather stations
- 







"""


import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataanalysis")
os.chdir(path)

path_data = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","data")

#%% import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial.polynomial import polyfit

# %matplotlib qt


#%% load data

# read data = data per day per region
data = pd.read_csv(os.path.join(path_data,"weather_all_stations_sum2.txt"), index_col=0)


#%% describe data in words / table


# summarize per year and group by coastid
yearsum = (
           data.groupby(by=["coast","year"])
           .agg({
               "HS0":["sum"],
               "HS1":["sum"],
               "HS2":["sum"],
               "HS3":["sum"],
               "HS4":["sum"]
               })
           ).reset_index()
yearsum.columns = yearsum.columns.droplevel()
yearsum.columns = ["coast","year",
                "HS0","HS1","HS2","HS3","HS4"]

# if there's an evolution in hours heat stress 2005 -- 2023: add year class
yearsum["yc"] = (np.floor((yearsum["year"]-2005)/5))*5+2005

# summary per yc, average number of 
summary = round(
          yearsum.groupby(by  = ["coast","yc"])
          .agg({"HS0":["mean","min","max"],
                "HS1":["mean","min","max"],
                "HS2":["mean","min","max"],
                "HS3":["mean","min","max"],
                "HS4":["mean","min","max"]}))

# all regions togehter
summary = pd.concat([summary,
    round(
          yearsum.groupby(by  = ["yc"])
          .agg({"HS0":["mean","min","max"],
                "HS1":["mean","min","max"],
                "HS2":["mean","min","max"],
                "HS3":["mean","min","max"],
                "HS4":["mean","min","max"]}))
    ])

# add row for all
yearsum["all"] = 1
summary = pd.concat([summary,
                     pd.DataFrame(round(
          yearsum.groupby("all")
          .agg({"HS0":["mean","min","max"],
                "HS1":["mean","min","max"],
                "HS2":["mean","min","max"],
                "HS3":["mean","min","max"],
                "HS4":["mean","min","max"]})).to_numpy().T.flatten().reshape(1,15)
          , index = ["all"],columns=summary.columns)])


#%% add extra features

data = data.sort_values(by = ["coast","year","day"]).reset_index(drop=1)

for i in range(0,3):
    print(i)
    wea = data.loc[data["coast"] == i,:].copy()
    
    # total number of hours HS
    wea["HS_tot"] = wea["HS2"] + wea["HS3"] + wea["HS4"]
    
    # define extra features % (in hours) HS present day -1 > day -3
    wea["HS_1"] = np.nan
    wea.iloc[1:].loc[:,"HS_1"] = (wea["HS_tot"].iloc[0:-1].values) / (1*24)*100
    
    wea["HS_3_1"] = np.nan
    wea.iloc[3:].loc[:,"HS_3_1"] = (wea["HS_tot"].iloc[0:-3].values + \
                            wea["HS_tot"].iloc[1:-2].values + \
                            wea["HS_tot"].iloc[2:-1].values) / (3*24)*100
    
    # define extra features % (in hours) recovery present day -1 > day -3
    wea["REC_3_1"] = np.nan
    wea["RECHS"] = 0
    wea.loc[wea["HS_tot"]>0,"RECHS"] = wea.loc[wea["HS_tot"]>0,"HS0"]
    wea.iloc[3:].loc[:,"REC_3_1"] = (wea["RECHS"].iloc[0:-3].values + \
                            wea["RECHS"].iloc[1:-2].values + \
                            wea["RECHS"].iloc[2:-1].values) / (3*24)*100
    
    # define extra features % (in hours) recovery yesterday
    wea["REC_1"] = np.nan
    wea.iloc[1:].loc[:,"REC_1"] = (wea["RECHS"].iloc[0:-1].values)/(24)*100
    
    # try to catch nonlinearity of the effect "heat load"
    wea["HSnl"] = wea["HS2"] + 2*wea["HS3"] + 4*wea["HS4"]
    wea["HSnl13"] = 0
    wea.iloc[3:].loc[:,"HSnl13"] =  wea["HSnl"].iloc[0:-3].values + \
                            wea["HSnl"].iloc[1:-2].values + \
                            wea["HSnl"].iloc[2:-1].values

    # merge into data
    data.loc[wea.index.values,"HS_tot"] = wea["HS_tot"]
    data.loc[wea.index.values,"HS_1"] = wea["HS_1"]
    data.loc[wea.index.values,"HS_3_1"] = wea["HS_3_1"]
    data.loc[wea.index.values,"REC_3_1"] = wea["REC_3_1"]
    data.loc[wea.index.values,"REC_1"] = wea["REC_1"]
    data.loc[wea.index.values,"HSnl"] = wea["HSnl"]
    data.loc[wea.index.values,"HSnl13"] = wea["HSnl13"]

# if HS_tot = 0: REC = 0
del i, wea

sums = data.describe()

#%% visualisations of all data

#---------------------------------TEMPERATURE----------------------------------
# set data in long format
data["pos"] = 0
dataw = (
        pd.concat([data[["coast","pos","Tmin"]].rename(columns={"Tmin" : "temp"}),
                  (pd.concat([data["coast"],data["pos"]+1,data["Tmax"]],axis=1)).rename(columns={"Tmax" : "temp"})])
        ).reset_index(drop=1)

# plot over all years together
sns.set_theme(style="whitegrid")
_, ax = plt.subplots(1,3, figsize = (17,5), width_ratios=[1,1,1]) 
sns.swarmplot(data=dataw.loc[(dataw["coast"]==0) & (dataw.index.values%5==0)],x="pos", y="temp",ax=ax[0], size = 2,hue = "pos", 
              palette = sns.set_palette(["#0000CC","#D80D0D"]),legend=False)
sns.boxplot(data=dataw.loc[dataw["coast"]==0],x="pos",y="temp",ax=ax[0], saturation = 1, 
            palette =sns.set_palette(["#6666FF","#FF9999"]))  # palette red blue
ax[0].set_xticklabels(["min T°","max T°"])
ax[0].set_title("average minimum and maximum temperature\n coast < 10 kms")
ax[0].set_ylabel("temperature [C°]")
ax[0].set_xlabel("")
ax[0].set_ylim(-17,43)

sns.swarmplot(data=dataw.loc[(dataw["coast"]==1) & (dataw.index.values%5==0)],x="pos", y="temp",ax=ax[1], size = 2,hue = "pos", 
              palette = sns.set_palette(["#0000CC","#D80D0D"]),legend=False)
sns.boxplot(data=dataw.loc[dataw["coast"]==1],x="pos",y="temp",ax=ax[1], saturation = 1, 
            palette =sns.set_palette(["#6666FF","#FF9999"]))  # palette red blue
ax[1].set_xticklabels(["min T°","max T°"])
ax[1].set_title("average minimum and maximum temperature\n coast 10 - 50 kms")
ax[1].set_ylabel("temperature [C°]")
ax[1].set_xlabel("")
ax[1].set_ylim(-17,43)

sns.swarmplot(data=dataw.loc[(dataw["coast"]==2) & (dataw.index.values%5==0)],x="pos", y="temp",ax=ax[2], size = 2,hue = "pos", 
              palette = sns.set_palette(["#0000CC","#D80D0D"]),legend=False)
sns.boxplot(data=dataw.loc[dataw["coast"]==2],x="pos",y="temp",ax=ax[2], saturation = 1, 
            palette =sns.set_palette(["#6666FF","#FF9999"]))  # palette red blue
ax[2].set_xticklabels(["min T°","max T°"])
ax[2].set_title("average minimum and maximum temperature\n inland ")
ax[2].set_ylabel("temperature [C°]")
ax[2].set_xlabel("")
ax[2].set_ylim(-17,43)

#-----------------------------------REL HUM------------------------------------
# set data in long format
data["pos"] = 0
dataw = (
        pd.concat([data[["coast","pos","RHmin"]].rename(columns={"RHmin" : "RH"}),
                  (pd.concat([data["coast"],data["pos"]+1,data["RHmax"]],axis=1)).rename(columns={"RHmax" : "RH"})])
        ).reset_index(drop=1)

# plot over all years together
sns.set_theme(style="whitegrid")
_, ax = plt.subplots(1,3, figsize = (17,5), width_ratios=[1,1,1]) 
sns.swarmplot(data=dataw.loc[(dataw["coast"]==0) & (dataw.index.values%5==0)],x="pos", y="RH",ax=ax[0], size = 2,hue = "pos", 
              palette = sns.set_palette(["#FF8000","#009999"]),legend=False)
sns.boxplot(data=dataw.loc[dataw["coast"]==0],x="pos",y="RH",ax=ax[0], saturation = 1, 
            palette =sns.set_palette(["#FFB266","#CCFFFF"]))  # palette red blue
ax[0].set_xticklabels(["min RH","max RH"])
ax[0].set_title("average minimum and maximum relative humidity\n coast < 10 kms")
ax[0].set_ylabel("relative humidity [%]")
ax[0].set_xlabel("")
ax[0].set_ylim(0,100)

sns.swarmplot(data=dataw.loc[(dataw["coast"]==1) & (dataw.index.values%5==0)],x="pos", y="RH",ax=ax[1], size = 2,hue = "pos", 
              palette = sns.set_palette(["#FF8000","#009999"]),legend=False)
sns.boxplot(data=dataw.loc[dataw["coast"]==1],x="pos",y="RH",ax=ax[1], saturation = 1, 
            palette =sns.set_palette(["#FFB266","#CCFFFF"]))  # palette red blue
ax[1].set_xticklabels(["min RH","max RH"])
ax[1].set_title("average minimum and maximum relative humidity\n coast 10 - 50 kms")
ax[1].set_ylabel("relative humidity [%]")
ax[1].set_xlabel("")
ax[1].set_ylim(0,100)

sns.swarmplot(data=dataw.loc[(dataw["coast"]==2) & (dataw.index.values%5==0)],x="pos", y="RH",ax=ax[2], size = 2,hue = "pos", 
              palette = sns.set_palette(["#FF8000","#009999"]),legend=False)
sns.boxplot(data=dataw.loc[dataw["coast"]==2],x="pos",y="RH",ax=ax[2], saturation = 1, 
            palette =sns.set_palette(["#FFB266","#CCFFFF"]))  # palette red blue
ax[2].set_xticklabels(["min RH","max RH"])
ax[2].set_title("average minimum and maximum relative humidity\n inland ")
ax[2].set_ylabel("relative humidity [%]")
ax[2].set_xlabel("")
ax[2].set_ylim(0,100)

#------------------------------------ THI--------------------------------------
# set data in long format
data["pos"] = 0
dataw = (
        pd.concat([data[["coast","pos","THImin"]].rename(columns={"THImin" : "THI"}),
                  (pd.concat([data["coast"],data["pos"]+1,data["THImax"]],axis=1)).rename(columns={"THImax" : "THI"})])
        ).reset_index(drop=1)

# plot over all years together
sns.set_theme(style="whitegrid")
_, ax = plt.subplots(1,3, figsize = (17,5), width_ratios=[1,1,1]) 
sns.swarmplot(data=dataw.loc[(dataw["coast"]==0) & (dataw.index.values%5==0)],x="pos", y="THI",ax=ax[0], size = 2,hue = "pos", 
              palette = sns.set_palette(["royalblue","darkviolet"]),legend=False)
sns.boxplot(data=dataw.loc[dataw["coast"]==0],x="pos",y="THI",ax=ax[0], saturation = 1, 
            palette =sns.set_palette(["lightsteelblue","plum"]))  # palette red blue
ax[0].set_xticklabels(["min THI","max THI"])
ax[0].set_title("average minimum and maximum THI\n coast < 10 kms")
ax[0].set_ylabel("THI")
ax[0].set_xlabel("")
ax[0].set_ylim(5,90)

sns.swarmplot(data=dataw.loc[(dataw["coast"]==1) & (dataw.index.values%5==0)],x="pos", y="THI",ax=ax[1], size = 2,hue = "pos", 
              palette = sns.set_palette(["royalblue","darkviolet"]),legend=False)
sns.boxplot(data=dataw.loc[dataw["coast"]==1],x="pos",y="THI",ax=ax[1], saturation = 1, 
            palette =sns.set_palette(["lightsteelblue","plum"]))  # palette red blue
ax[1].set_xticklabels(["min THI","max THI"])
ax[1].set_title("average minimum and maximum THI\n coast 10 - 50 kms")
ax[1].set_ylabel("THI")
ax[1].set_xlabel("")
ax[1].set_ylim(5,90)

sns.swarmplot(data=dataw.loc[(dataw["coast"]==2) & (dataw.index.values%5==0)],x="pos", y="THI",ax=ax[2], size = 2,hue = "pos", 
              palette = sns.set_palette(["royalblue","darkviolet"]),legend=False)
sns.boxplot(data=dataw.loc[dataw["coast"]==2],x="pos",y="THI",ax=ax[2], saturation = 1, 
            palette =sns.set_palette(["lightsteelblue","plum"]))  # palette red blue
ax[2].set_xticklabels(["min THI","max THI"])
ax[2].set_title("average minimum and maximum THI\n inland ")
ax[2].set_ylabel("THI")
ax[2].set_xlabel("")
ax[2].set_ylim(5,90)

ax[0].axhline(68,color = 'r',linewidth=2)
ax[1].axhline(68,color = 'r',linewidth=2)
ax[2].axhline(68,color = 'r',linewidth=2)

plt.savefig(os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","documentation","Figure_3.png"))


#------------------------------------THI---------------------------------------
_, ax = plt.subplots(3,1, figsize = (8,15), sharex=True) 

subset = data[["year","day","coast","THImin"]].pivot(index=["year","day"],columns = "coast",values = "THImin")
ax[0].hist(x=subset,density=True, bins = np.linspace(20,90,16),label = "coast",
           color=["darkturquoise","#20639B","#ED553B"])
sns.kdeplot(data=subset,palette=["darkturquoise","#20639B","#ED553B"],
            common_norm = False, ax=ax[0])

subset = data[["year","day","coast","THIavg"]].pivot(index=["year","day"],
                                                     columns = "coast",
                                                     values = "THIavg")
ax[1].hist(x=subset,density=True, bins = np.linspace(20,90,16),label = "coast",
           color=["darkturquoise","#20639B","#ED553B"])
sns.kdeplot(data=subset,palette=["darkturquoise","#20639B","#ED553B"],
            common_norm = False, ax=ax[1])

subset = data[["year","day","coast","THImax"]].pivot(index=["year","day"],
                                                     columns = "coast",
                                                     values = "THImax")
ax[2].hist(x=subset,density=True, bins = np.linspace(20,90,16),label = "coast",
           color=["darkturquoise","#20639B","#ED553B"])
sns.kdeplot(data=subset,palette=["darkturquoise","#20639B","#ED553B"],
            common_norm = False, ax=ax[2])


ax[0].set_xlim(18,90)
ax[2].set_xlabel("THI")
ax[0].set_ylabel("density")
ax[1].set_ylabel("density")
ax[2].set_ylabel("density")
ax[0].set_title("daily minimal THI")
ax[1].set_title("daily average THI")
ax[2].set_title("daily maximal THI")
ax[0].legend(["_","coast < 10km","coast 10-50 km","inland"])
# ax[0].legend(["coast < 10km","coast 10-50 km","inland"])
ax[1].legend("")
ax[2].legend("")
del ax, subset


# duration heat stress days
data["pos"] = 0
data["HS"] = (data["HS_tot"]==0).astype(int)
data["test"] = data["HS"].ne(data["HS"].shift()).cumsum()
data.loc[data["HS"]==1,"test"] = 0
test = data[["coast","test","pos"]].groupby(["coast","test"]).count()
test = test.loc[test["pos"] < 100].reset_index()
subset = test.pivot(index=["test"],
                    columns = "coast",
                    values = "pos")

sns.set_style("whitegrid")
_, ax = plt.subplots(1,1, figsize = (6,5)) 
ax.hist(x=subset,density=True,bins = [0,2,4,6,8,10,12,14,21,28,35],label = "coast",
           color=["darkturquoise","#20639B","#ED553B"])
ax.set_xticks([1,3,5,7,9,11,13,18,25,32])
sns.kdeplot(data=subset,palette=["darkturquoise","#20639B","#ED553B"],
            common_norm = False, ax=ax)
ax.set_xlim(0,33)
ax.set_xlabel("duration [days]")
ax.set_ylabel("density")
ax.set_title("successive days with THI at least 1h$\geq 68$")
ax.legend(["_","coast < 10km","coast 10-50 km","inland"])
plt.savefig(os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","documentation","Figure_4.png"))
plt.close()
del test
data = data.drop(columns=["test","pos"])

#------------------------------------------------------------------------------

#recovery - IF HS_tot >0: how often no recovery?

subset = (
          data.loc[data["HS_tot"]>0,["year","day","coast","HS0"]]
          .pivot(index=["year","day"],
                 columns = "coast",
                 values = "HS0")
          )
# problem: takes na as 0
sns.set_style("whitegrid")
_, ax = plt.subplots(1,1, figsize = (6,5)) 
ax.hist(x=subset,density=True,bins = [0,4,8,12,16,20,24],label = "coast",
           color=["darkturquoise","#20639B","#ED553B"])
ax.set_xticks([2,6,10,14,18,22])
ax.set_xticklabels(["[0;4[","[4;8[","[8;12[","[12;16[","[16;20[","[20;24]"])
sns.kdeplot(data=subset,palette=["darkturquoise","#20639B","#ED553B"],
            common_norm = False, ax=ax)
ax.set_xlim(-1,24)
ax.set_xlabel("hours recovery [h]")
ax.set_ylabel("density")
ax.set_title("recovery, hours THI$\leq$64 on days with THI at least 1h$\geq$68")
ax.legend(["_","coast < 10km","coast 10-50 km","inland"])
plt.savefig(os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","documentation","Figure_5.png"))


test = (
        pd.concat([subset[0].value_counts(),subset[1].value_counts(),subset[2].value_counts()],axis=1,join='outer')
        ).reset_index().sort_values(by="index").reset_index(drop=1)
test1 = test[["index",0]]
test1=test1.rename(columns = {0:"count"})
test1["coast"] = 0
test1["count"] = test1["count"]/test1["count"].sum()
test2 = test[["index",1]]
test2["coast"] = 1
test2=test2.rename(columns = {1:"count"})
test2["count"] = test2["count"]/test2["count"].sum()
test3 = test[["index",2]]
test3["coast"] = 2
test3=test3.rename(columns = {2:"count"})
test3["count"] = test3["count"]/test3["count"].sum()
test=pd.concat([test1,test2,test3])
test=test.rename(columns={"index":"hours"}).reset_index(drop=1)

_, ax = plt.subplots(1,1, figsize = (6,5)) 
sns.barplot(data=test,x="hours",y="count",
            hue="coast",palette=["darkturquoise","#20639B","#ED553B"])
sns.kdeplot(data=subset,palette=["darkturquoise","#20639B","#ED553B"],
            common_norm = False, ax=ax)
ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,24])
ax.set_xticklabels([0,2,4,6,8,10,12,14,16,18,20,22,24])
ax.set_xlim(-1,23)
ax.legend().set_title('coast')
labels = ["coast < 10km","coast 10-50 km","inland"]
h, l = ax.get_legend_handles_labels()
ax.legend(h, labels)
ax.set_xlabel("hours recovery [h]")
ax.set_ylabel("density")
ax.set_title("recovery, hours THI$\leq$64 on days with THI at least 1h$\geq$68")

plt.savefig(os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","documentation","Figure_5.png"))
plt.close()
del ax,h,l
del test1,test2,test3


#-------------------------------CORRELATION PLOTS------------------------------
wfnames = ["max THI", "avg THI", "no. of \nrecovery hours",
           "no. of hours \nTHI$\geq$68",
           "% THI$\geq$68 \nin d -1","% THI$\geq$68 \nin d -3 to -1",
           "recovery in \nd -1", "recovery in \nd -3 to -1",
           "severity-based \ninsult","cumulative insult\n in d -3 to -1"
           ]
names = data.columns.to_frame().reset_index(drop=1).iloc[[10,11,12,31,32,33,35,34,36,37],:]
sns.set(font_scale=0.9)
sns.set_style("darkgrid")
cmap = sns.diverging_palette(230, 20, as_cmap=True)
_,ax = plt.subplots(1,1,figsize = (10,9))
sns.heatmap(data=round(data.iloc[:,names.index.values].corr(),2),ax=ax, 
            annot=True,lw=0.2,cmap=cmap,
            mask = np.triu(np.ones_like(data.iloc[:,names.index.values].corr(), dtype=bool))==True,
            xticklabels=wfnames,yticklabels=wfnames,
            cbar=False)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 45)
ax.set_title("correlations between weather features, all regions", fontsize="large")
plt.savefig(os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","documentation","Figure_6.png"))
plt.close()

# highlights
_,ax = plt.subplots(1,1,figsize = (7,6))
sns.scatterplot(data=data,x="HS_tot",y="HSnl",color = "blue")
x = data["HS_tot"].values
y = data["HSnl"].values
a,b,c= polyfit(x,y,2)
plt.plot(np.sort(x),a+b*np.sort(x)+c*(np.sort(x)**2),"-",color = 'red',lw=3)
plt.text(12.2,47.8,
         "y = " + str(round(a,3)) + " + " + str(round(b,3)) + " *x + " + str(round(c,3)) + "*x²",
         color = "red",
         fontsize = "small",
         fontweight="bold"
         )
ax.set_title('correlation total no. hours THI $\geq$ 68 and severity based insult')
ax.set_xlabel('total no. of hours THI $\geq$ 68')
ax.set_ylabel('non-linear heat stress feature, HSnl')
plt.savefig(os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","documentation","Figure_8.png"))
plt.close()

#%% correlation with daylight -THI

daylight = pd.read_csv(os.path.join(path_data,"daylight.txt"), index_col=0)
daylight["date"] = pd.to_datetime(daylight["date"],format="%Y-%m-%d")
daylight["year"] = daylight["date"].dt.year
daylight["day"] = daylight["date"].dt.dayofyear

data = data.merge(daylight[["year","day","hrs_daylight"]],on=["year","day"],how='outer')
x = data["THIavg"].values
y = data["hrs_daylight"].values
b,m= polyfit(x,y,1)

sns.set_style("whitegrid")
_,ax = plt.subplots(1,1,figsize = (7,6))
sns.scatterplot(data=data,x="THIavg",y="hrs_daylight",hue="season")
plt.plot(x,b+m*x,"-",color = 'red',lw=2)
ax.set_ylim(7.5,17)
ax.set_xlabel("average THI")
ax.set_ylabel("no. of hours daylight [h]")
ax.set_title('correlation THI and daylight, r = ' + \
             str(round(data[["THIavg","hrs_daylight"]].corr()["THIavg"]["hrs_daylight"],3)))
plt.savefig(os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","documentation","Figure_7.png"))
plt.close()



# WEATHER CORRELATION BETWEEN COAST COAST AND INLAND

#%%----------------------------------------------------------


startdates = {1: 2011}
enddates = {1: 2019}
farm = 1
act = pd.read_csv(os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","datapreprocessing",
                    "results","data","act_preprocessed_"
                               + str(farm) + ".txt"),
                  usecols=['date', 'act_new'])
act["date"] = pd.to_datetime(act["date"], format='%Y-%m-%d')
act = act.rename(columns={'act_corr': 'activity'})

act["day"] = act["date"].dt.dayofyear
act["year"] = act["date"].dt.year
test = act[["year","day","act_new"]].groupby(["year","day"]).median().reset_index()

_,ax = plt.subplots(3,1,sharex=True,figsize = (12,8))
sns.scatterplot(data=test,x="day",y="act_new",ax=ax[0])
sns.scatterplot(data=data,x="day",y="THIavg",ax=ax[1])
sns.scatterplot(data=data,x="day",y="hrs_daylight",ax=ax[2])

test2 = test.merge(data[["year","day","THIavg","hrs_daylight"]], on = ["year","day"])
(test2.iloc[:,[2,3,4]]).corr()


# correlation act and THI and daylight
_,ax = plt.subplots(1,2,figsize = (12,5))
x = test2["THIavg"].values
y = test2["act_new"].values
a,b,c= polyfit(x,y,2)
sns.scatterplot(data=test2,x="THIavg",y="act_new",color='blue',ax=ax[0])
ax[0].plot(np.sort(x),a+b*np.sort(x)+c*(np.sort(x)**2),"-",color = 'red',lw=2)
ax[0].set_ylim(450,780)
ax[0].set_xlim(25,np.max(x))
ax[0].set_title('THI vs. median activity')
ax[0].set_xlabel('average THI')
ax[0].set_ylabel('avg. daily activity')

x = test2["hrs_daylight"].values
y = test2["act_new"].values
b,m= polyfit(x,y,1)
sns.scatterplot(data=test2,x="hrs_daylight",y="act_new",color='teal',ax=ax[1])
ax[1].plot(x,b+m*x,"-",color = 'red',lw=2)
ax[1].set_ylim(450,780)
ax[1].set_xlim(np.min(x),np.max(x))
ax[1].set_title('daylight hours vs. median activity')
ax[1].set_xlabel('hours daylight')
ax[1].set_ylabel('avg. daily activity')
plt.savefig(os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","documentation","Figure_9.png"))
plt.close()


