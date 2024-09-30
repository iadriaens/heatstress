# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:25:49 2023

@author: u0084712

-------------------------------------------------------------------------------


============================= THI exploration =================================

Typically, THI and thresholding is used to determine when cows are suffering 
   from heat stress. The commonly accepted thresholds/categories are approx:
           - mild stress: THI 68 -> 71
           - moderate stress: THI 72 -> 79
           - severe stress: THI 80 -> 89
           - critical/dead animals if 90 or above
        ==> in Germany: milk yield decline from THI 60 onwards
        ==> effect expressed per unit THI, yet probably not a linear effect
        ==> thresholds depend on breed: e.g. HF = 68, CB = 79, Jersey = 75

Besides temperature and humidity, also following factors affect true insult to 
    the cows:
        - duration of high THI during the day: if T° drops to <21°C during the
          night, cows can recover better 
        - indoor barn climate:
                * mitigation applied: sprinklers, ventilation, shadow
                * orientation of the barn
                ==> differences can potentially be revealed by comparing
                    effects of heat stress across farms
        - wind speed
        

-------------------------------------------------------------------------------
RQ answered with this script:
    - how often did each of the THI happen for each farm?
    - what is the typical duration and time between high THI episodes
    - what about cooling down during the evening or "THI differences" during 
      days
    - can we define THI features that are more indicative of heat stress in 
      dairy cows than average THI alone? "THI insults" 
          e.g. * duration of high THI during a day
               * level of cooling down at night
               * number of successive days insulted - length of HS period


-------------------------------------------------------------------------------
Output generated:
    - table with THI insults per farm / per date
    - insights in research questions as defined above


-------------------------------------------------------------------------------
References:
https://dairy.extension.wisc.edu/articles/animal-handling-during-heat-stress/
https://celticseaminerals.com/heat-stress-dairy-cows/
https://www.frontiersin.org/articles/10.3389/fanim.2022.946592/full
https://www.welfarm.co.nz/heat-stress/

"""

import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataanalysis")
os.chdir(path)


#%% import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns

# %matplotlib qt


#%% paths and constants

path_data = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","data")

# farm selected
farms = [30,31,33,34,35,38,39,40,43,44,45,46,48] 


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
data["thi_high"] = (data["thi_avg"]>=68).astype(int)

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
    
#%% plot farms and weather stations

# set coordinate system
crs = {'init':'epsg:4326'}

# load data Belgium and provinces
geomap1 =  gpd.read_file(os.path.join(path_data,"BEL_adm","BEL_adm2.shp"))
geomap1 = geomap1.drop(axis=1,columns = ['ID_0', 'NAME_0', 'ID_1', 'NAME_1', 'ID_2', 'TYPE_2',
       'ENGTYPE_2', 'NL_NAME_2', 'VARNAME_2'])
geomap1.columns = ["country","province","geometry"]
geomap1 = geomap1.to_crs('EPSG:4326')

geomap2 = gpd.read_file(os.path.join(path_data,"NLD_adm","gadm41_NLD_1.shp"))
geomap2 = geomap2.drop(axis=1,columns = ['GID_1', 'COUNTRY', 'VARNAME_1', 'NL_NAME_1',
       'TYPE_1', 'ENGTYPE_1', 'CC_1', 'HASC_1', 'ISO_1'])
geomap2.columns = ["country","province","geometry"]
geomap2 = geomap2.to_crs('EPSG:4326')

geomap = gpd.pd.concat([geomap1,geomap2]).reset_index(drop=1)
geomap["number"] = range(0,len(geomap))
del geomap1,geomap2

# load aws data
aws = pd.read_csv(os.path.join(path_data,"aws.txt"), index_col = 0)
geometry2 = [Point(xy) for xy in zip(aws['longitude'], aws['latitude'])]
aws_df = gpd.GeoDataFrame(aws,crs = crs, geometry = geometry2)

# plot all weatherstations
fig,ax = plt.subplots(figsize = (15,15))
geomap.plot(ax=ax, alpha = 0.9, column = "number", cmap = "PiYG")
aws_df.plot(ax=ax, marker = "*", color = "blue", markersize = 50)
ax.set_title("available weather stations in LT database")
ax.set_xlabel("longitude [°E]")
ax.set_ylabel("latitude [°N]")

# save plot
plt.savefig(os.path.join(path,"results","thi","weatherstations_all_farms.tif"))    
plt.close()

# read data of farm
for farm in farms:
    # load farm data
    frm = pd.read_csv(os.path.join(path_data,"farm_" + str(farm) + ".txt"),
                      index_col=0)
    # select ids
    ids = frm.loc[0,["aws_id1","aws_id2","aws_id3"]].reset_index(drop=1)
    ids = ids.to_frame(name = "aws_id")
    ids = ids.dropna()
    aws_farm = pd.merge(aws,ids, how = "inner")
    
    # tranfer longitude and latitude as points
    geometry = [Point(xy) for xy in zip(frm['longitude'], frm['latitude'])]
    geo_df = gpd.GeoDataFrame(frm,crs = crs, geometry = geometry)
    
    # transfer longitude and latitude as points for aws_farm
    geometry = [Point(xy) for xy in zip(aws_farm['longitude'], aws_farm['latitude'])]
    geo_aws = gpd.GeoDataFrame(aws_farm,crs = crs, geometry = geometry )
    
    # plot
    fig,ax = plt.subplots(figsize = (15,15))
    geomap.plot(ax=ax, alpha = 0.9, column = "number", cmap = "PiYG")
    geo_df.plot(ax=ax, marker = "s", color = "orangered", markersize = 80)
    geo_df.plot(ax=ax, marker = "$F$", color = "black", markersize = 60)
    geo_aws.plot(ax=ax, marker = "*", color = "blue", markersize = 80)
    ax.set_title("weather stations used for farm = " +str(farm))
    ax.set_xlabel("longitude [°E]")
    ax.set_ylabel("latitude [°N]")
    
    # save plots
    plt.savefig(os.path.join(path,"results","thi","weatherstations_farm_" + str(farm) + ".tif"))    
    
    ax.set_ylim((frm["latitude"][0]-0.5, frm["latitude"][0]+0.5))
    plt.savefig(os.path.join(path,"results","thi","weatherstations_farm_" + str(farm) + "_zoom.tif"))    
    plt.close()

del fig, ax, frm, geo_aws, geo_df, geomap, geometry, geometry2, ids, crs, aws_farm




#%% explore THI features
#   - level = "time" (all farms together)
#   - level = "farm-time" (per farm)
   
#================================  ALL FARMS ================================== 

# %matplotlib qt

# -------------------------------- thi and temp -------------------------------
fig,ax = plt.subplots(2,1,figsize = (20,12), sharex = True)

# temperature 
sns.lineplot(data=data, x="date", y="temp_min",color = "blue",
             estimator = "mean", errorbar = "sd", ax = ax[0])
sns.lineplot(data=data, x="date", y="temp_max",color = "red",
             estimator = "mean", errorbar = "sd", ax = ax[0])
ax[0].set_xlim(ax[0].get_xlim()[0],ax[0].get_xlim()[1])
# years = np.arange(ax[0].get_xticks()[1],ax[0].get_xticks()[-1],365)
ax[0].set_ylim(ax[0].get_ylim()[0],ax[0].get_ylim()[1])
ax[0].fill_between(ax[0].get_xlim(),25,ax[0].get_ylim()[1],
                   color = "red",alpha = 0.2)
ax[0].fill_between(ax[0].get_xlim(),ax[0].get_ylim()[0],18,
                   color = "blue",alpha = 0.2)
years = np.arange(ax[0].get_xticks()[0],ax[0].get_xticks()[-1],365)
for y in years:
    ax[0].plot([y,y],[ax[0].get_ylim()[0],ax[0].get_ylim()[1]],
               linestyle = "--",color = "grey",linewidth = 1.2)
ax[0].set_title("min and max T°, all farms")
ax[0].set_ylabel("temp [°C]")
ax[0].legend(["min temp","zone cool down","max temp","zone heat stressed"], loc = "lower right")

# thi
ax[1].set_xlabel("date")
sns.lineplot(data=data, x="date", y="thi_max",color = "mediumvioletred",
             estimator = "mean", errorbar = "sd", ax = ax[1])
sns.lineplot(data=data, x="date", y="thi_avg",color = "purple",
             estimator = "mean", errorbar = "sd", ax = ax[1])

ax[1].set_ylim(ax[1].get_ylim()[0],ax[1].get_ylim()[1])
ax[1].fill_between(ax[1].get_xlim(),68,72,
                   color = "lightcoral",alpha = 0.2)
ax[1].fill_between(ax[1].get_xlim(),72,80,
                   color = "crimson",alpha = 0.3)
ax[1].fill_between(ax[1].get_xlim(),80,ax[1].get_ylim()[1],
                   color = "darkred",alpha = 0.4)
ax[1].legend(["max THI","severe HS","mean THI","moderate HS","mild HS"], loc = "lower right")
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

for y in years:
    ax[1].plot([y,y],[ax[1].get_ylim()[0],ax[1].get_ylim()[1]],
               linestyle = "--",color = "grey",linewidth = 1.2)
ax[1].set_title("avg. and max THI + zones, all farms")
ax[1].set_ylabel("THI")

plt.savefig(os.path.join(path,"results","thi","temp_thi_allfarms.tif"))  
plt.close()


# hours mild, moderate, severe heat stress
fig,ax = plt.subplots(1,1,figsize = (18,8))
data = data.sort_values(by = "date").reset_index(drop=1)
data["datecat"] = data["date"].astype(str)
data["thi_hrs_HS"] = data["thi_hrs_mild"] + data["thi_hrs_mod"] + data["thi_hrs_sev"]
sns.barplot(data=data,x="datecat",y="thi_hrs_HS",
            estimator = "mean",
            errorbar = None, width = 1)
ax.set_ylim(ax.get_ylim()[0],ax.get_ylim()[1])
xticks = ax.get_xticks()
ax.set_xticks(xticks[0::365])
xlabels = []
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    print(n,label)
    xlabels.append(label.get_text()[0:4])
ax.set_xticklabels(xlabels)
xticks = ax.get_xticks()
for xt in xticks:
    ax.plot([xt,xt],[ax.get_ylim()[0],ax.get_ylim()[1]],
             linestyle = "--",color = "grey",linewidth = 1.2)
ax.set_xlabel("year")
ax.set_ylabel("total hours THI >= 68 per day")
plt.savefig(os.path.join(path,"results","thi","hrs_high_thi_allfarms.tif"))  
plt.close()


# per mild/mod/sev
data_sum = (
            data[["date","thi_hrs_mild","thi_hrs_mod","thi_hrs_sev"]].groupby(by="date")
            .mean()).reset_index()
fig,ax = plt.subplots(1,1,figsize = (18,8))
ax.bar(data_sum["date"],data_sum["thi_hrs_mild"], color = "mediumseagreen", width=1.5)
ax.bar(data_sum["date"],data_sum["thi_hrs_mod"], color = "deepskyblue",
       bottom = data_sum["thi_hrs_mild"], width=1.5)
ax.bar(data_sum["date"],data_sum["thi_hrs_sev"], color = "crimson", 
       bottom = data_sum["thi_hrs_mild"] + data_sum["thi_hrs_mod"], width=1.5)
ax.set_ylim(0,24)
ax.set_xlim(ax.get_xlim()[0],ax.get_xlim()[1])
# years = ax.get_xticks()#[1:]
years = np.arange(ax.get_xticks()[0]-365,ax.get_xticks()[-1],365)
for y in years:
    ax.plot([y,y],[0,24],
             linestyle = "--",color = "grey",linewidth = 1.2)
ax.set_xlabel("date")
ax.set_ylabel("hours [h]")
ax.set_title("hours with mild>=68, moderate>=72, severe>=80 heat stress")
plt.savefig(os.path.join(path,"results","thi","hours_mild_mod_sev_thi.tif"))  
# plt.close()

# zoom per year and add month vertical lines
fig,ax = plt.subplots(1,1,figsize = (8,4))
ax.bar(data_sum["date"],data_sum["thi_hrs_mild"], color = "mediumseagreen", width=1.5)
ax.bar(data_sum["date"],data_sum["thi_hrs_mod"], color = "deepskyblue",
       bottom = data_sum["thi_hrs_mild"], width=1.5)
ax.bar(data_sum["date"],data_sum["thi_hrs_sev"], color = "crimson", 
       bottom = data_sum["thi_hrs_mild"] + data_sum["thi_hrs_mod"], width=1.5)
ax.set_ylim(0,24)
#years = ax.get_xticks()[1:]
years = np.arange(ax.get_xticks()[1],ax.get_xticks()[-1],365)
for y in range(0,len(years)-1):
    print('test ' + str(years[y]) + str(years[y+1]) )
    ax.set_xlim(years[y],years[y+1])
    thisyear = (ax.get_xticklabels()[0]).get_text()[0:4]
    months = ax.get_xticks()
    if len(months) > 8:
        months = months[0::2]
    for m in months:
        ax.plot([m,m],[0,24],
             linestyle = "--",color = "palegoldenrod",linewidth = 1.2)
        ax.set_xlabel("date")
        ax.set_ylabel("hours [h]")
        ax.set_title("hours in " + thisyear + " with mild>=68, moderate>=72, severe>=80 heat stress")
    plt.savefig(os.path.join(path,"results","thi",thisyear + "_hours_mild_mod_sev_thi.tif"))  
plt.close()

# per month
data_sum["year"] = data_sum["date"].dt.year
data_sum["month"] = data_sum["date"].dt.month


# plot per month total number of hours
data_sum2 = (
    data_sum[['year','month', 'thi_hrs_mild', 'thi_hrs_mod', 'thi_hrs_sev']]
    .groupby(by = ["year","month"]).sum()
    ).reset_index()
data_sum2["hrs_total"] = data_sum2[['thi_hrs_mild', 'thi_hrs_mod', 'thi_hrs_sev']].sum(axis=1)
df = pd.melt(data_sum2, id_vars = ["year","month"],value_vars = ['hrs_total','thi_hrs_mild', 'thi_hrs_mod', 'thi_hrs_sev'])


sns.set_style("whitegrid",{"grid.color": ".5", "grid.linestyle": ":"})
fig,ax = plt.subplots(4,5,figsize = (20,16), sharey = True, sharex = True)    
T=-1
for y in data_sum2["year"].drop_duplicates().iloc[0:-1]:
    T = T + 1
    print(y,str(T//5),str(T%5))
    
    sns.barplot(data = df.loc[df['year']==y], x = "month", y = "value", 
                hue = "variable",palette = "bright",ax= ax[int(T//5)][int(T%5)])
    ax[int(T//5)][int(T%5)].set_title(str(y))
    ax[int(T//5)][int(T%5)].axhline(y=200,linewidth=0.5, color='orangered', linestyle ="--")
    ax[int(T//5)][int(T%5)].axhline(y=100,linewidth=0.5, color='darkmagenta', linestyle ="--")
    
    if T == 4:
        ax[int(T//5)][int(T%5)].legend(labels = ["total","mild","moderate","severe"],bbox_to_anchor=(1.55, 1))
        L = ax[int(T//5)][int(T%5)].legend(bbox_to_anchor=(1.55, 1))
        L.get_texts()[0].set_text("total")
        L.get_texts()[1].set_text("mild")
        L.get_texts()[2].set_text("moderate")
        L.get_texts()[3].set_text("severe")
    else:
        ax[int(T//5)][int(T%5)].get_legend().remove()
    if T%5 == 0:
        ax[int(T//5)][int(T%5)].set_ylabel("hours in THI zones")
    if T < 13:
        ax[int(T//5)][int(T%5)].set_xlabel("")

fig.delaxes(ax[3][4])
fig.delaxes(ax[3][3])

plt.savefig(os.path.join(path,"results","thi","total_hours_permonth_mild_mod_sev_thi.tif"))  
plt.close()

# clean workspace
del ax, L, label, m, months, y, years, thisyear, xlabels, xt, xticks, T, n, fig


# quantification in numbers
data_sum["year"] = data_sum["date"].dt.year
data_sum["month"]
data_sum[['year', 'thi_hrs_mild', 'thi_hrs_mod', 'thi_hrs_sev']].groupby(by = 'year').sum()





#%% ============================  INDIVIDUAL FARMS ============================ 


#------------------------------------------------------------------------------

for farm in farms:
    print("farm  = " + str(farm))
    
    # select data
    fdata = data.loc[data["farm_id"]==farm,:].reset_index(drop=1)
    
    # april to october
    fdata410 = data.loc[(data["farm_id"]==farm) & \
                        (data["month"]>=6) & \
                        (data["month"]<=9),:].reset_index(drop=1)

    # plot  THI and temp ------------------------------------------------------
    fig,ax = plt.subplots(2,1,figsize = (18,12), sharex = True)

    # temperature 
    sns.lineplot(data=fdata, x="date", y="temp_min",color = "blue",
                 estimator = "mean", errorbar = "sd", ax = ax[0])
    sns.lineplot(data=fdata, x="date", y="temp_max",color = "red",
                 estimator = "mean", errorbar = "sd", ax = ax[0])
    ax[0].set_xlim(ax[0].get_xlim()[0],ax[0].get_xlim()[1])
    ax[0].set_ylim(ax[0].get_ylim()[0],ax[0].get_ylim()[1])
    ax[0].fill_between(ax[0].get_xlim(),25,ax[0].get_ylim()[1],
                       color = "red",alpha = 0.2)
    ax[0].fill_between(ax[0].get_xlim(),ax[0].get_ylim()[0],18,
                       color = "blue",alpha = 0.2)
    years = np.arange(ax[0].get_xticks()[0],ax[0].get_xticks()[-1],365)
    for y in years:
        ax[0].plot([y,y],[ax[0].get_ylim()[0],ax[0].get_ylim()[1]],
                   linestyle = "--",color = "grey",linewidth = 1.2)
    ax[0].set_title("min and max T°, farm = " + str(farm))
    ax[0].set_ylabel("temp [°C]")
    ax[0].legend(["min temp","zone cool down","max temp","zone heat stressed"], 
                 loc = "lower right")

    # thi
    ax[1].set_xlabel("date")
    sns.lineplot(data=fdata, x="date", y="thi_max",color = "mediumvioletred",
                 estimator = "mean", errorbar = "sd", ax = ax[1])
    sns.lineplot(data=fdata, x="date", y="thi_avg",color = "purple",
                 estimator = "mean", errorbar = "sd", ax = ax[1])

    ax[1].set_ylim(ax[1].get_ylim()[0],ax[1].get_ylim()[1])
    ax[1].fill_between(ax[1].get_xlim(),68,72,
                       color = "lightcoral",alpha = 0.2)
    ax[1].fill_between(ax[1].get_xlim(),72,80,
                       color = "crimson",alpha = 0.3)
    ax[1].fill_between(ax[1].get_xlim(),80,ax[1].get_ylim()[1],
                       color = "darkred",alpha = 0.4)
    ax[1].legend(["max THI","severe HS","mean THI","moderate HS","mild HS"], loc = "lower right")
    
    plt.savefig(os.path.join(path,"results","thi","temp_thi_farm" + str(farm) + ".tif"))  
    plt.close()
    
    
    # plot per week hours of heat stress 
    fdata410["week"] = fdata410["date"].dt.isocalendar().week
    fdata410["weekcat"] = (fdata410["date"] - \
                           pd.to_timedelta(fdata410["date"].dt.dayofweek,
                           unit = "d"))
        
        # todo check!!
    fdata410["weekcat"] = pd.to_datetime(fdata410["weekcat"],format = "%Y-%m-%d")
    fdata_sum = (
        fdata410[['weekcat','thi_hrs_mild', 'thi_hrs_mod', 'thi_hrs_sev']]
        .groupby(by = ['weekcat']).sum()
        ).sort_values(by="weekcat").reset_index()
    fdata_sum["hrs_total"] = fdata_sum[['thi_hrs_mild', 'thi_hrs_mod', 'thi_hrs_sev']].sum(axis=1)
    fdata_sum["month"] = fdata_sum["weekcat"].dt.month
    fdata_sum = fdata_sum.loc[fdata_sum["month"] !=3,:].reset_index(drop=1)
    df = pd.melt(fdata_sum, id_vars = ["month","weekcat"],value_vars = ['hrs_total',
                                                                        'thi_hrs_mild', 'thi_hrs_mod', 'thi_hrs_sev'])
    df["year"] = df["weekcat"].dt.year
    df = df.sort_values(by="weekcat").reset_index(drop=1)
    
    sns.set_style("whitegrid",{"grid.color": ".9", "grid.linestyle": ":"})
    fig,ax = plt.subplots(4,5,figsize = (20,16), sharey = True, sharex = True)    
    T=-1
    for y in (fdata410["year"].drop_duplicates()).iloc[0:-1]:
        T = T + 1
        print(y,str(T//5),str(T%5))
        
        sns.barplot(data = df.loc[df['year']==y].sort_values(by = ["weekcat","variable"]).reset_index(drop=1),
                    x = "weekcat", y = "value", 
                    hue = "variable",palette = "bright",ax= ax[int(T//5)][int(T%5)])
        ax[int(T//5)][int(T%5)].set_title(str(y))
        # ax[int(T//4)][int(T%4)].axhline(y=200,linewidth=0.5, color='orangered', linestyle ="--")
        # ax[int(T//4)][int(T%4)].axhline(y=100,linewidth=0.5, color='darkmagenta', linestyle ="--")
        
        
        if T == 4:
            print()
            ax[int(T//5)][int(T%5)].legend(labels = ["total","mild","moderate","severe"],bbox_to_anchor=(1.55, 1))
        else:
            ax[int(T//5)][int(T%5)].get_legend().remove()
        if T%5 == 0:
            ax[int(T//5)][int(T%5)].set_ylabel("hours in THI zones")
        if T<13:
            ax[int(T//5)][int(T%5)].set_xlabel("")
        else:
            ax[int(T//5)][int(T%5)].set_xlabel("week")
            xlabels = []
            for n, label in enumerate(ax[int(T//5)][int(T%5)].xaxis.get_ticklabels()):
                print(n,label)
                label.set_rotation(45)
                xlabels.append(
                    (ax[int(T//5)][int(T%5)].get_xticklabels()[n]).get_text()[8:10] + \
                    (ax[int(T//5)][int(T%5)].get_xticklabels()[n]).get_text()[4:7]
                    )
                
                # if n not in n_weeks:
                #     label.set_visible(False)
            ax[int(T//5)][int(T%5)].set_xticklabels(xlabels)
            Tel = 0
            for n, label in enumerate(ax[int(T//5)][int(T%5)].get_xticklabels()):
                print(n,label)
                if Tel%5 != 0:
                    label.set_visible(False)
                Tel=Tel+1
    fig.delaxes(ax[3][4])
    fig.delaxes(ax[3][3])

                
    plt.savefig(os.path.join(path,"results","thi","total_hours_perweek_mild_mod_sev_thi_farm_" + str(farm) +".tif"))  
    plt.close()
    






