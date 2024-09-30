# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 08:59:02 2024

@author: u0084712


-------------------------------------------------------------------------------

preprocessing of the "per farm" weather


"""


import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","datapreprocessing")
os.chdir(path)

path_data = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","data")

import pandas as pd




#%% select farms, set constants and load data

# farm selected
farms = [1, 2, 3, 4, 5, 6, 30,31,33,34,35,38,39,40,43,44,45,46,48]

for farm in farms:
    
    # read data
    wea = pd.read_csv(os.path.join(path_data,"weather_" + str(farm) + ".txt"),
                       usecols=["datetime","temp","rel_humidity"])
    wea = wea.rename(columns = {"rel_humidity" : "rhum"})
    wea["datetime"] = pd.to_datetime(wea["datetime"],format="%Y-%m-%d %H:%M:%S")
    wea["date"] = pd.to_datetime(wea["datetime"].dt.date,format="%Y-%m-%d")
    wea = wea.loc[wea["datetime"].dt.minute == 0,:].reset_index(drop=1)
    wea["year"] = wea["datetime"].dt.year
    wea["day"] = wea["datetime"].dt.dayofyear
    
    # calculate features
    wea["thi"] = 1.8 * wea["temp"] + 32 - \
                    ((0.55 - 0.0055 * wea["rhum"]) * \
                     (1.8 * wea["temp"] - 26))
                        
    # drop hours for which temp or rhum are nan
    idx = wea[["temp","rhum"]].dropna().index
    wea = wea.loc[idx].reset_index(drop=1)
    del idx
    
    # set threshold columns [0;64[ - [64;68[ - [68;72[ - [72;80[ - [80;100]
    wea["HS0"] = (wea["thi"]<64).astype(int)  # hours recovery
    wea["HS1"] = ((wea["thi"]>=64)&(wea["thi"]<68)).astype(int) # neutral
    wea["HS2"] = ((wea["thi"]>=68)&(wea["thi"]<72)).astype(int) # mild HS
    wea["HS3"] = ((wea["thi"]>=72)&(wea["thi"]<80)).astype(int) # moderate HS
    wea["HS4"] = (wea["thi"]>=80).astype(int) # severe HS

    # per day summaries, per weather stations
    data = (
            wea.groupby(by = ["date"])
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
    data.columns = ["date","nobs",
                    "Tmin","Tmax","Tavg",
                    "RHmin","RHmax","RHavg",
                    "THImin","THImax","THIavg",
                    "HS0","HS1","HS2","HS3","HS4"]
    
    # add hours heatstress (HS2-HS4) in total
    data["HS_tot"] = data["HS2"] + data["HS3"] + data["HS4"] 
    
    # write to data
    data.to_csv(os.path.join(path, "results", "data","weather_prep_farm_" + str(farm) + ".txt"))
