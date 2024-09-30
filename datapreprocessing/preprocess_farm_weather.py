# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:48:35 2024

@author: u0084712

-------------------------------------------------------------------------------

preprocess weather all farms

"""




import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","datapreprocessing")
os.chdir(path)

#%% 

import pandas as pd
import numpy as np

path_data = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","data")

raft = pd.read_csv(os.path.join(path_data,"raft_locations.txt"))

oldid = pd.read_csv(os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","data","new","farmid_renumber.txt"))


#%% load weatherdata

weather = pd.DataFrame([],columns = ["farm_id","farmname","time","temp","rhum","thi","HS0",
                                     "HS1","HS2","HS3","HS4"])


for farm in oldid.old:
    print(farm)

    wea = pd.read_csv(os.path.join(path_data,"weather_" + str(farm) + ".txt"), 
                      usecols = ["datetime","temp","rel_humidity"])
    wea.columns=["time","temp","rhum"]
    wea["time"] = pd.to_datetime(wea["time"],format = "%Y-%m-%d %H:%M:%S")
    wea["date"] = wea["time"].dt.date
    wea["year"] = wea["time"].dt.year
    wea["day"] = wea["time"].dt.dayofyear
    wea["hour"] = wea["time"].dt.hour
    wea = wea.loc[wea["time"].dt.minute == 0,:].reset_index(drop=1)
    
    wea["farm_id"] = np.ones((len(wea),1))*oldid.loc[oldid["old"]==farm,"new"].values
    wea["farmname"] = oldid.loc[oldid["old"]==farm,"farmname"].repeat(len(wea)).values

    # calculate per hour thi
    wea["thi"] = 1.8 * wea["temp"] + 32 - \
                        ((0.55 - 0.0055 * wea["rhum"]) * \
                         (1.8 * wea["temp"] - 26))
                            
    # drop hours for which temp or rhum are nan
    idx = wea[["temp","rhum"]].dropna().index
    wea = wea.loc[idx].reset_index(drop=1)
    del idx
    
    # set threshold columns [0;64[ - [64;68[ - [68;72[ - [72;80[ - [80;100]
    wea["HS0"] = (wea["thi"]<64).astype(int)
    wea["HS1"] = ((wea["thi"]>=64)&(wea["thi"]<68)).astype(int)
    wea["HS2"] = ((wea["thi"]>=68)&(wea["thi"]<72)).astype(int)
    wea["HS3"] = ((wea["thi"]>=72)&(wea["thi"]<80)).astype(int)
    wea["HS4"] = (wea["thi"]>=80).astype(int)
       
    weather = pd.concat([weather,wea]).reset_index(drop=1)

# set to same order as raft weather
weather = weather[["farm_id","time","temp","rhum","thi","HS0",
                  "HS1","HS2","HS3","HS4","year","day","hour","farmname","date"]]
# save
weather.to_csv(os.path.join(path_data,"new","weather_BENL.txt"))
del weather,farm, oldid, wea, raft

#%% combine BE/NL and summarize

# load raft weather
raftweather = pd.read_csv(os.path.join(path_data,"weather_raft_all.txt"),index_col=0)
benlweather = pd.read_csv(os.path.join(path_data,"new","weather_BENL.txt"),index_col=0)

# merge together
wea = pd.concat([raftweather,benlweather]) 
wea["time"] = pd.to_datetime(wea["time"],format = "%Y-%m-%d %H:%M:%S")
wea["date"] = pd.to_datetime(wea["date"],format = "%Y-%m-%d")
wea[["HS0","HS1","HS2","HS3","HS4"]] = wea[["HS0","HS1","HS2","HS3","HS4"]].astype(int)

del raftweather, benlweather

#%% summaries

# per day summaries, per weather stations
data = (
        wea.groupby(by = ["farm_id","date"])
        .agg({"temp":["count","min","max","mean"],
              "rhum":["min","max","mean"],
              "thi":["min","max","mean"],
              "HS0":"sum",
              "HS1":"sum",
              "HS2":"sum",
              "HS3":"sum",
              "HS4":"sum",
              })
        ).reset_index()

data.columns = data.columns.droplevel()
data.columns = ["farm_id","date","nobs",
                "Tmin","Tmax","Tavg",
                "RHmin","RHmax","RHavg",
                "THImin","THImax","THIavg",
                "HS0","HS1","HS2","HS3","HS4"]



# add hours heatstress (HS2-HS4) in total
data["HS_tot"] = data["HS2"] + data["HS3"] + data["HS4"] 

# delete data with less than 22 hours
data = data.loc[data["nobs"]>=22,:].reset_index(drop=1)

# express HS in % to correct for days with less than 22 hours
data["HSp0"] = data["HS0"]/data["nobs"]*100
data["HSp1"] = data["HS1"]/data["nobs"]*100
data["HSp2"] = data["HS2"]/data["nobs"]*100
data["HSp3"] = data["HS3"]/data["nobs"]*100
data["HSp4"] = data["HS4"]/data["nobs"]*100
data["HSptot"] = data["HS_tot"]/data["nobs"]*100

# write to data
data.to_csv(os.path.join(path, "results", "data","new_weather_all_farms.txt"))








#%% select weather events and classify

"""
heat stress events - selection

THI in barn = 8.75+0.897*THI outdoors (VanderZaag et al. 2023)

THI load of the THI event = first and last day with THI at least once > 65







"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_data = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","data","new")
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","datapreprocessing","results","data","new")

farms = [41,42,43,44,46,47,48,49,50,51,54,55,57,58,59,61,62,64,65,66,67,69] 

for farm in farms:
    # load data
    df = pd.read_csv(os.path.join(path_data,"newweather_" + str(farm) + ".txt"),
                     index_col=0)
    df["time"] = pd.to_datetime(df["time"],format = "%Y-%m-%d %H:%M:%S")
    df["date"] = pd.to_datetime(df["time"].dt.date,format = "%Y-%m-%d")
    df["hour"] = df["time"].dt.hour

    # calculate THI
    df["thi"] = 1.8 * df["temp"] + 32 - \
                    ((0.55 - 0.0055 * df["rhum"]) * \
                      (1.8 * df["temp"] - 26))
    # df["thii"] = 8.75+0.897*df["thi"]  # indoor thi estimation based on vanderzaag
    
    # calculate excess above 66 outdoor threshold == plus minus 68 indoor
    df["thi_excess"] = df["thi"]-66
    df.loc[df["thi_excess"]<0,"thi_excess"] = 0
    df["hcount"] = 0  # count hours with value > 66
    df.loc[df["thi_excess"]>0,"hcount"] = 1
    # df["thii_excess"] = df["thii"]-68
    # df.loc[df["thii_excess"]<0,"thii_excess"] = 0
    
    # daysum
    dsum = df[["date","thi_excess","thi","hcount"]].groupby("date").agg({"thi_excess":["count","sum","max"],
                                                          "thi":["mean"],"hcount":["sum"]}).reset_index()
    dsum.columns=dsum.columns.droplevel()
    dsum.columns = ["date","count","thi_sum","thi_max","thi_mean","no_h_thi66"]
    
    # series of heat stress days - successive days with no. hours thi >66 = 6 or more (25%)
    dsum = dsum.reset_index()
    dsum = dsum.loc[(dsum["no_h_thi66"]>=6),:].reset_index(drop=1)
    dsum["diff"] = dsum["index"].diff()
    dsum.loc[0,"diff"] = 150
    
    # per event: new event if diff > 1
    dsum["no"] = np.nan
    dsum.loc[0,"no"]=1
    dsum.loc[dsum["diff"]>1,"no"] = np.arange(1,(dsum["diff"]>1).sum()+1)
    dsum["no"] = dsum["no"].fillna(method = "ffill")
    
    #--------------------------------------------------------------------------
    # characterise heat stress events outdoor thi
    hs = dsum[["no","diff","thi_sum","thi_max","thi_mean","no_h_thi66"]].groupby("no").agg({"thi_sum":["count","sum"],
                                                                    "thi_max":"max",
                                                                    "diff":"max",
                                                                    "no_h_thi66":"mean"}).reset_index()
    hs.columns=hs.columns.droplevel()
    hs.columns = ["no","duration","total_excess","max_excess","time_since_prev","avg_h_excess"]
        
    ###########################################################################
    # idea development - accumulaton of heat stress / effect of previous HS   #
    #    episodes, taken into account
    #    IF less than 10 days different, add % of previous episode to current 
    #    as the baseline, with % = derived from decay function y=a**x with a=0.7
    ##########################################################################
    
    hs["baseline"] = 0
    idx=hs.loc[hs["time_since_prev"]<=10].index.values-1
    hs.loc[hs["time_since_prev"]<=10,"baseline"] = \
        (0.7**(hs.loc[hs["time_since_prev"]<=10,"time_since_prev"])).values * \
        hs.loc[idx,"total_excess"].values
    del idx

    # add date of start and end
    startdate = dsum[["no","date"]].drop_duplicates("no").rename(columns={"date":"start"})
    enddate = (dsum[["no","date"]].drop_duplicates("no",keep="last")).rename(columns={"date":"end"})
    hs = hs.merge(startdate,on="no")
    hs = hs.merge(enddate,on="no")
    del startdate,enddate
    
    # visualise
    fig,ax = plt.subplots()
    (hs["total_excess"]+hs["baseline"]).hist(bins=50)
    
    # add severity based on quantiles
    hs["severity"] = 2
    hs.loc[(hs["total_excess"]+hs["baseline"])<(hs["total_excess"]+hs["baseline"]).quantile(0.33),"severity"]=1
    hs.loc[(hs["total_excess"]+hs["baseline"])>(hs["total_excess"]+hs["baseline"]).quantile(0.66),"severity"]=3
    
    #A add total heat load = baseline+current excess
    hs["heat_load"] = hs["total_excess"]+hs["baseline"]
    
    # add mean thi of that event
    hs["mean_thi"] = 0
    for i in hs.index.values:
        # print(i)
        subset = dsum.loc[(dsum["date"]>=hs["start"].iloc[i]) & \
                          (dsum["date"]<=hs["end"].iloc[i])]
        hs.loc[i,"mean_thi"] = round(subset["thi_mean"].mean(),2)
    
    #--------------------------------------------------------------------------
    # daysum
    dsum2 = df[["date","thi_excess","thi","hcount"]].groupby("date").agg({"thi_excess":["count","sum","max"],
                                                          "thi":["mean"],"hcount":["sum"]}).reset_index()
    dsum2.columns=dsum2.columns.droplevel()
    dsum2.columns = ["date","count","thi_sum","thi_max","thi_mean","no_h_thi66"]
    dsum2 = dsum2.merge(dsum[["date","no"]],how="outer",on="date")
    dsum2 = dsum2.merge(hs[["no","severity","duration"]],how="outer",on="no")
    
    
    # heat stress based on cumulative hs load
    dsum2["hs"] = dsum2["thi_sum"].ewm(span=5).sum() 
    dsum2.loc[dsum2["hs"]<3,"hs"] = 0
    
    # save hs and dsum
    hs.to_csv(os.path.join(path,"weather_hs_farm_" + str(farm) + ".txt"))
    dsum2.to_csv(os.path.join(path,"weather_daysum_farm_" + str(farm) + ".txt"))
    
    


#%% decay functions
import matplotlib.pyplot as plt
# %matplotlib qt

fig,ax = plt.subplots()
ax.plot(np.linspace(-1,60,61),np.ones([61,1]),'--',color="k")
ax.plot(np.linspace(-1,60,61),0.5*np.ones([61,1]),':',color="k")
ax.plot(np.linspace(-1,60,61),0*np.ones([61,1]),color="k",lw=0.5)

ax.plot([2,2],[0,1],'--',color="r")
ax.plot([0,0],[0,1],'--',color="r")


x = np.linspace(0,60,61)
a = 0.7
b = 1
f = a**x

ax.plot(x,f)
ax.set_xlabel("x in days since heat stress peak")
ax.set_ylabel("decay of cumulative heat stress excess")
ax.set_title("decay function continued heat stress effect\n "+\
             "y = $a^{x}$ with a = " + str(a))
ax.set_xlim(-0.5,10)
del ax, fig, x, f, a, b 


# exponentially moving weighting average
fig,ax = plt.subplots()
ax.grid(True)

span = 5  # span >= 1
alpha = 2/(span+1)
ax.plot(np.linspace(0,20),(1-alpha)**np.linspace(0,20))

com = 2 # center of mass
alpha = 1/(1+com)
ax.plot(np.linspace(0,20),(1-alpha)**np.linspace(0,20))



