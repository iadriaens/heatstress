# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 09:39:28 2023

@author: u0084712

-------------------------------------------------------------------------------
BEFORE running this script:
    
    => ensure proper connection with the server:
            - activate Pulse Secure B-zone with u number and pw (VPN)
            - ensure you have a valid certagent certificate running
            - open cmd and make ssh connection to server with
            ssh -L 55432:localhost:5432 u0084712@biosyst-s-lt01.bm.set.kuleuven.be
            
-------------------------------------------------------------------------------

db access via "CowBase_paper" db for Italian data

            
    
"""

import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataextraction")
os.chdir(path)


#%% import packages for connection and extraction

import pandas as pd
from ServerConnect import LT_connect
import json
import numpy as np

path_out = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","data","new")

#%% connect to CowBase paper database and view farms

# Import the serverSettings as a dictionary from the given json parameter file.
with open(os.path.join(path,"serverSettings_CowBase.json")) as file:
    ssd = json.load(file)
    
# Create a connection to the database
pgres = LT_connect(**ssd["s_parameters"])

# Create a SQL statment and query the statement
sql_statement = """
SELECT DISTINCT(farm_id) 
FROM public.activity
;
"""
# excecute query - farms = farm 1 to 6
farms = pgres.query(query=sql_statement).sort_values(by="farm_id").reset_index(drop=1)
print(farms)

"""
farm_id : 
    1 = Demol - 34 (BE, "west" / coastal)
    2 = Huzen - 38 (NL, "east" / "close to germany")
    3 = Konings - 39 (NL, "west" (but not really close to sea))
    4 = Theuwis - 43 (BE, "east" / limburg)
    5 = Vandenbroek - 44 (BE, central / antwerp north)
    6 = Piazza - Italy
"""

raft = pd.read_csv(os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","data","raft_locations.txt"))
raft["altitude"] = 70 # height of raft ripon

#%% extract data
f = pd.DataFrame([],columns=["farm_id","farmname","latitude","longitude"])
# select data from list of farms
for farm in farms.farm_id:
    print("farm = " + str(farm))
    
    # milk
    print("currently running... MILK")
    sql_statement = f"""
        SELECT milking_id,milking_oid,farm_id,animal_id,lactation_id,milking_system_id,parity,started_at, ended_at, mi, dim, tmy, mylf, mylr, myrf, myrr, eclf, eclr, ecrf, ecrr, milk_t 
        FROM public.milking
        WHERE farm_id IN (%s) 
        ;
        """ % (str(farm))
    
    milk = pgres.query(query=sql_statement)
    fn = "milk_" + str(farm) + ".txt"
    milk.to_csv(os.path.join(path_out,fn))
    del milk, fn, sql_statement
    
    # activity ----------------------------------------------------------------
    print("currently running... ACTIVITY")
    sql_statement = f"""
        SELECT *
        FROM activity
        WHERE farm_id IN (%s)
        ;
        """ % (str(farm))
    act = pgres.query(query=sql_statement)
    fn = "act_" + str(farm) + ".txt"
    act.to_csv(os.path.join(path_out,fn))
    del act, fn, sql_statement    
    
    # farm --------------------------------------------------------------------
    print("currently running... FARM " + str(farm))
    sql_statement = f"""
        SELECT farm_id, farmname, latitude, longitude
        FROM public.farm
        WHERE farm_id IN (%s)
        ;
        """ % (str(farm))
    frm = pgres.query(query=sql_statement)  
    f = pd.concat([f,frm])
    
    # if farm not in raft["farm_id"].values:
    #     print(farm, "not in raft")
    
    #     # weather -----------------------------------------------------------------
    #     print("currently running... WEATHER " + str(farm))
    #     sql_statement = f"""
    #     SELECT *
    #     FROM public.aws
    #     ;
    #     """
    #     aws = pgres.query(query=sql_statement)
    #     aws.to_csv(os.path.join(path_out,"aws.txt"))
            
    #     # merge with aws to get closest weather information
    #     frm["aws_id"]=np.nan
    #     frm["aws_dist"]=np.nan
    #     lat = frm["latitude"].values
    #     long = frm["longitude"].values
        
    #     # longitude and latitude degrees are not the same in our latitude --
    #     #    one degree longitude is approx everywhere equal to 111 kms
    #     #    one degree latitude is approx 70 kms in BE/NL
    #     aws["dist"] = np.sqrt((70*(aws["latitude"]-lat))**2 + \
    #                           (111*(aws["longitude"]-long))**2)
    #     aws.loc[aws["dist"]==0,"dist"] = 1
    #     # select three closest aws to middle out weather information
    #     frm["aws_id1"] = aws.loc[aws["dist"]==aws["dist"].min(), "aws_id"].values
    #     frm["aws_dist1"] = aws.loc[aws["dist"]==aws["dist"].min(), "dist"].values
    #     aws = aws.drop(aws.loc[aws["dist"]==aws["dist"].min()].index, axis = 0)
    #     frm["aws_id2"] = aws.loc[aws["dist"]==aws["dist"].min(), "aws_id"].values
    #     frm["aws_dist2"] = aws.loc[aws["dist"]==aws["dist"].min(), "dist"].values
    #     aws = aws.drop(aws.loc[aws["dist"]==aws["dist"].min()].index, axis = 0)
    #     frm["aws_id3"] = aws.loc[aws["dist"]==aws["dist"].min(), "aws_id"].values
    #     frm["aws_dist3"] = aws.loc[aws["dist"]==aws["dist"].min(), "dist"].values
        
    #     # select weather data from aws1 if dist < 100 km
    #     if frm["aws_dist1"].iloc[0] <= 100:
    #         sql_statement = f"""
    #             SELECT weather_id, aws_id, datetime, temperature, humidity
    #             FROM public.weather
    #             WHERE aws_id IN (%s)
    #             ;
    #             """ % (str(frm["aws_id1"].iloc[0]))
    #         wea1 = pgres.query(query=sql_statement)
    #     else:
    #         wea1 = pd.DataFrame([],columns = ["weather_id","aws_id","datetime","temperature","humidity"])
        
    #     # select weather data from aws2
    #     if frm["aws_dist2"].iloc[0] <= 100:
    #         sql_statement = f"""
    #             SELECT weather_id, aws_id, datetime, temperature, humidity
    #             FROM public.weather
    #             WHERE aws_id IN (%s)
    #             ;
    #             """ % (str(frm["aws_id2"].iloc[0]))
    #         wea2 = pgres.query(query=sql_statement)
    #     else:
    #         wea2 = pd.DataFrame([],columns = ["weather_id","aws_id","datetime","temperature","humidity"])
        
    #     # select weather data from aws3
    #     if frm["aws_dist2"].iloc[0] <= 100:
    #         sql_statement = f"""
    #             SELECT weather_id, aws_id, datetime, temperature, humidity
    #             FROM public.weather
    #             WHERE aws_id IN (%s)
    #             ;
    #             """ % (str(frm["aws_id3"].iloc[0]))
    #         wea3 = pgres.query(query=sql_statement)
    #     else:
    #         wea3 = pd.DataFrame([],columns = ["weather_id","aws_id","datetime","temperature","humidity"])
        
    #     # merge weather data together on datetime values
    #     wea = pd.merge(wea1,wea2, how = "outer",
    #                    on = "datetime").sort_values(by = "datetime").reset_index(drop=1)
    #     wea = wea.drop(columns = ["weather_id_x", "weather_id_y"])
    #     wea.columns = ['aws_id_1', 'datetime', 'temp_1', 'rel_humidity_1',
    #                    'aws_id_2', 'temp_2', 'rel_humidity_2']
    #     wea = pd.merge(wea,wea3, how = "outer",
    #                    on = "datetime").sort_values(by = "datetime").reset_index(drop=1)
    #     wea = wea.drop(columns = ["weather_id"])
    #     wea.columns = ['aws_id1', 'datetime', 'temp1', 'rel_humidity1',
    #            'aws_id2', 'temp2',
    #            'rel_humidity2',
    #            'aws_id3', 'temp3', 'rel_humidity3']
        
    #     # add distance-based weighting factor to weather station and remove dist if no data
    #     wea["dist1"] = frm["aws_dist1"].iloc[0]
    #     wea.loc[wea.loc[wea["temp1"].isna()].index.values,"dist1"] = np.nan
    #     wea["dist2"] = frm["aws_dist2"].iloc[0]
    #     wea.loc[wea.loc[wea["temp2"].isna()].index.values,"dist2"] = np.nan
    #     wea["dist3"] = frm["aws_dist3"].iloc[0]
    #     wea.loc[wea.loc[wea["temp3"].isna()].index.values,"dist3"] = np.nan
    #     wea["totdist"] = wea[["dist1","dist2","dist3"]].sum(axis = 1, skipna = True)
        
    #     wea["f1"] = wea["totdist"] / wea["dist1"]
    #     wea["f2"] = wea["totdist"] / wea["dist2"]
    #     wea["f3"] = wea["totdist"] / wea["dist3"]
    #     wea["totf"] = wea[["f1","f2","f3"]].sum(axis = 1, skipna = True)      
        
    #     wea["w1"] = wea["f1"]  /  wea["totf"]
    #     wea["w2"] = wea["f2"]  /  wea["totf"]
    #     wea["w3"] = wea["f3"]  /  wea["totf"]
    #     wea.loc[wea.loc[wea["w1"].isna()].index.values,["temp1","rel_humidity1"]] = -999
    #     wea.loc[wea.loc[wea["w1"].isna()].index.values,"w1"] = 0
    #     wea.loc[wea.loc[wea["w2"].isna()].index.values,["temp2","rel_humidity2"]] = -999
    #     wea.loc[wea.loc[wea["w2"].isna()].index.values,"w2"] = 0
    #     wea.loc[wea.loc[wea["w3"].isna()].index.values,["temp3","rel_humidity3"]] = -999
    #     wea.loc[wea.loc[wea["w3"].isna()].index.values,"w3"] = 0
        
    #     wea = wea.drop(columns = ["totdist",
    #                               "f1","f2","f3","totf"])
        
    #     # calculate weighted weather
    #     wea["temp"] = wea["w1"] * wea["temp1"] + \
    #                   wea["w2"] * wea["temp2"] + \
    #                   wea["w3"] * wea["temp3"]
    #     wea["rel_humidity"] = wea["w1"] * wea["rel_humidity1"] + \
    #                           wea["w2"] * wea["rel_humidity2"] + \
    #                           wea["w3"] * wea["rel_humidity3"]
    #     wea.loc[(wea["w1"] +wea["w2"]+wea["w3"])==0, "temp"] = np.nan
    #     wea.loc[(wea["w1"] +wea["w2"]+wea["w3"])==0, "rel_humidity"] = np.nan 
        
    #     # keep which aws used
    #     wea.loc[wea.loc[wea["temp1"].isna()].index.values,"aws_id1"] = np.nan
    #     wea.loc[wea.loc[wea["temp2"].isna()].index.values,"aws_id2"] = np.nan
    #     wea.loc[wea.loc[wea["temp3"].isna()].index.values,"aws_id3"] = np.nan
                            
    #     # select cols
    #     wea = wea[["datetime", "aws_id1","aws_id2","aws_id3",
    #                "dist1","dist2","dist3",
    #                "temp", "rel_humidity"]]
    
    #     fn = "weather_" + str(farm) + ".txt"
    #     wea.to_csv(os.path.join(path_out,fn))
    #     del wea1, wea2, wea3,wea
    fn = "farm_" + str(farm) + ".txt"
    frm.to_csv(os.path.join(path_out,fn))
    del frm, fn, sql_statement
    
    
#%% cow and lactation data

for farm in farms.farm_id:
    print(farm)
    # cow data ----------------------------------------------------------------
    print("currently running... COW")
    sql_statement = f"""
        SELECT animal_id, farm_id, birth_date
        FROM public.animal
        WHERE farm_id IN (%s)
        ;
        """ % (str(farm))
    cow = pgres.query(query=sql_statement)
    fn = "cow_" + str(farm) + ".txt"
    cow.to_csv(os.path.join(path_out,fn))
    del cow, fn, sql_statement
    
    # lactation data ----------------------------------------------------------
    print("currently running... LAC")
    sql_statement = f"""
        SELECT lactation_id, farm_id, animal_id, parity, calving, dry_off
        FROM public.lactation
        WHERE farm_id IN (%s)
        ;
        """ % (str(farm))
    lac = pgres.query(query=sql_statement)
    fn = "cow_" + str(farm) + ".txt"
    lac.to_csv(os.path.join(path_out,fn))
    del lac, fn, sql_statement




#%% meteostat data in neighborhood of farms

from meteostat import Point, Hourly, Stations
import datetime as dt

start = dt.datetime.strptime("2005-01-01", "%Y-%m-%d")
end = dt.datetime.strptime("2024-01-01", "%Y-%m-%d")

weather = pd.DataFrame([],columns = ["farm_id","time","temp","rhum","thi","HS0",
                                     "HS1","HS2","HS3","HS4"])

for farm in raft["farm_id"]:
    print(farm)
    long = raft.loc[raft["farm_id"]==farm,"longitude"].values
    lat = raft.loc[raft["farm_id"]==farm,"latitude"].values
    alt = raft.loc[raft["farm_id"]==farm,"altitude"].values

    farmlocation = Point(lat,long,alt)
    farmlocation.method = "weighted"
    farmlocation.max_count = 5
    farmlocation.radius = 60000
    weather_add = Hourly(farmlocation, start, end)
    wea = weather_add.fetch()
    if len(wea)>0:
        wea = wea.reset_index()
        wea = wea[["time","temp","rhum"]]
        wea.to_csv(os.path.join(path_out,"newweather_" + str(farm) + ".txt"))
    
        wea["farm_id"] = farm
        wea["time"] = pd.to_datetime(wea["time"])
        wea["year"] = wea["time"].dt.year
        wea["day"] = wea["time"].dt.dayofyear
        wea["hour"] = wea["time"].dt.hour
        
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
        
        print(raft.loc[raft["farm_id"]==farm,"farmname"].values,"len data = " + str(len(wea)))
        
        weather = pd.concat([weather,wea]).reset_index(drop=1)
del long,lat,alt,farmlocation,weather_add,wea

# weather = weather.drop(columns=["farmname"])

# outer join to add farmname
weather = weather.merge(raft[["farm_id","farmname"]],on="farm_id")

weather["hour"] = weather["time"].dt.hour
weather["date"] = weather["time"].dt.date
# weather["minute"] = weather["time"].dt.minute
# test = weather.loc[weather["minute"]!=0,:]

weather.to_csv(os.path.join(path_out,"weather_raft_all.txt"))

test = weather[["farm_id","farmname","date"]].groupby(by= ["farm_id","farmname"]).agg({"date":["min","max"]})

# load milk data


#%% reload weather data BE/NL

from meteostat import Point, Hourly, Stations
import datetime as dt

start = dt.datetime.strptime("2005-01-01", "%Y-%m-%d")
end = dt.datetime.strptime("2024-01-01", "%Y-%m-%d")

weather = pd.DataFrame([],columns = ["farm_id","time","temp","rhum","thi","HS0",
                                     "HS1","HS2","HS3","HS4"])

benl = pd.read_json(os.path.join(path_out,"farmlocations.json")).T.reset_index()
benl.columns = ["farmname","lat","long","alt"]

farmids = pd.read_csv(os.path.join(path_out,"farmid_renumber.txt"), 
                      usecols = ["new","farmname"])

benl = benl.merge(farmids)
benl.columns = ["farmname","lat","long","alt","farm_id"]

benl.to_csv(os.path.join(path_out,"benl_locations.txt"))

for farm in benl["farm_id"]:
    print(farm)
    long = benl.loc[benl["farm_id"]==farm,"long"].values
    lat = benl.loc[benl["farm_id"]==farm,"lat"].values
    alt = benl.loc[benl["farm_id"]==farm,"alt"].values

    farmlocation = Point(lat,long,alt)
    farmlocation.method = "weighted"
    farmlocation.max_count = 5
    farmlocation.radius = 60000
    weather_add = Hourly(farmlocation, start, end)
    wea = weather_add.fetch()
    if len(wea)>0:
        wea = wea.reset_index()
        wea = wea[["time","temp","rhum"]]
        wea.to_csv(os.path.join(path_out,"newweather_" + str(farm) + ".txt"))
    
        wea["farm_id"] = farm
        wea["time"] = pd.to_datetime(wea["time"])
        wea["year"] = wea["time"].dt.year
        wea["day"] = wea["time"].dt.dayofyear
        wea["hour"] = wea["time"].dt.hour
        
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
        
        print(benl.loc[benl["farm_id"]==farm,"farmname"].values,"len data = " + str(len(wea)))
        
        weather = pd.concat([weather,wea]).reset_index(drop=1)
del long,lat,alt,farmlocation,weather_add,wea

# weather = weather.drop(columns=["farmname"])

# outer join to add farmname
weather = weather.merge(benl[["farm_id","farmname"]],on="farm_id")

weather["hour"] = weather["time"].dt.hour
weather["date"] = weather["time"].dt.date
# weather["minute"] = weather["time"].dt.minute
# test = weather.loc[weather["minute"]!=0,:]

weather.to_csv(os.path.join(path_out,"weather_benl_all.txt"))

test = weather[["farm_id","farmname","date"]].groupby(by= ["farm_id","farmname"]).agg({"date":["min","max"]})
