# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:51:38 2023

@author: u0084712

-------------------------------------------------------------------------------
BEFORE running this script:
    
    => ensure proper connection with the server:
            - activate Pulse Secure B-zone with u number and pw (VPN)
            - ensure you have a valid certagent certificate running
            - open cmd and make ssh connection to server with
            ssh -L 55432:localhost:5432 u-number@biosyst-s-lt01.bm.set.kuleuven.be
            
    
"""


import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataextraction")
os.chdir(path)


#%% import necessary packages
import pandas as pd
from ServerConnect import LT_connect
import json
import numpy as np
import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataextraction")
os.chdir(path)

path_out = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","data")

#%% lt database
# Import the serverSettings as a dictionary from the given json parameter file.
with open(os.path.join(path,"serverSettings.json")) as file:
    ssd = json.load(file)
    
# Create a connection to the database
pgres = LT_connect(**ssd["s_parameters"])

# Create a SQL statment and query the statement
sql_statement = """
SELECT DISTINCT(farm_id) 
FROM public.activity
;
"""
# excecute query
farms = pgres.query(query=sql_statement).sort_values(by="farm_id").reset_index(drop=1)
print(farms)

#%% download data from server

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
    
    # weather -----------------------------------------------------------------
    print("currently running... WEATHER " + str(farm))
    sql_statement = f"""
    SELECT *
    FROM public.aws
    ;
    """
    aws = pgres.query(query=sql_statement)
    aws.to_csv(os.path.join(path_out,"aws.txt"))
        
    # merge with aws to get closest weather information
    frm["aws_id"]=np.nan
    frm["aws_dist"]=np.nan
    lat = frm["latitude"].values
    long = frm["longitude"].values
    
    # longitude and latitude degrees are not the same in our latitude --
    #    one degree longitude is approx everywhere equal to 111 kms
    #    one degree latitude is approx 70 kms in BE/NL
    aws["dist"] = np.sqrt((70*(aws["latitude"]-lat))**2 + \
                          (111*(aws["longitude"]-long))**2)
    # select three closest aws to middle out weather information
    frm["aws_id1"] = aws.loc[aws["dist"]==aws["dist"].min(), "aws_id"].values
    frm["aws_dist1"] = aws.loc[aws["dist"]==aws["dist"].min(), "dist"].values
    aws = aws.drop(aws.loc[aws["dist"]==aws["dist"].min()].index, axis = 0)
    frm["aws_id2"] = aws.loc[aws["dist"]==aws["dist"].min(), "aws_id"].values
    frm["aws_dist2"] = aws.loc[aws["dist"]==aws["dist"].min(), "dist"].values
    aws = aws.drop(aws.loc[aws["dist"]==aws["dist"].min()].index, axis = 0)
    frm["aws_id3"] = aws.loc[aws["dist"]==aws["dist"].min(), "aws_id"].values
    frm["aws_dist3"] = aws.loc[aws["dist"]==aws["dist"].min(), "dist"].values
    
    # select weather data from aws1
    sql_statement = f"""
        SELECT weather_id, aws_id, datetime, temp, rel_humidity, wind_speed_avg_2m, wind_direction
        FROM public.weather
        WHERE aws_id IN (%s)
        ;
        """ % (str(frm["aws_id1"].iloc[0]))
    wea1 = pgres.query(query=sql_statement)
    
    # select weather data from aws2
    sql_statement = f"""
        SELECT weather_id, aws_id, datetime, temp, rel_humidity, wind_speed_avg_2m, wind_direction
        FROM public.weather
        WHERE aws_id IN (%s)
        ;
        """ % (str(frm["aws_id2"].iloc[0]))
    wea2 = pgres.query(query=sql_statement)
    
    # select weather data from aws2
    sql_statement = f"""
        SELECT weather_id, aws_id, datetime, temp, rel_humidity, wind_speed_avg_2m, wind_direction
        FROM public.weather
        WHERE aws_id IN (%s)
        ;
        """ % (str(frm["aws_id3"].iloc[0]))
    wea3 = pgres.query(query=sql_statement)
    
    # merge weather data together on datetime values
    wea = pd.merge(wea1,wea2, how = "outer",
                   on = "datetime").sort_values(by = "datetime").reset_index(drop=1)
    wea = wea.drop(columns = ["weather_id_x", "weather_id_y"])
    wea.columns = ['aws_id_1', 'datetime', 'temp_1', 'rel_humidity_1',
                   'wind_speed_avg_2m_1', 'wind_direction_1', 
                   'aws_id_2', 'temp_2', 'rel_humidity_2', 
                   'wind_speed_avg_2m_2', 'wind_direction_2']
    wea = pd.merge(wea,wea3, how = "outer",
                   on = "datetime").sort_values(by = "datetime").reset_index(drop=1)
    wea = wea.drop(columns = ["weather_id"])
    wea.columns = ['aws_id1', 'datetime', 'temp1', 'rel_humidity1',
           'wind_speed_avg_2m1', 'wind_direction1', 'aws_id2', 'temp2',
           'rel_humidity2', 'wind_speed_avg_2m2', 'wind_direction2',
           'aws_id3', 'temp3', 'rel_humidity3', 'wind_speed_avg_2m3',
           'wind_direction3']
    
    # add distance-based weighting factor to weather station and remove dist if no data
    wea["dist1"] = frm["aws_dist1"].iloc[0]
    wea.loc[wea.loc[wea["temp1"].isna()].index.values,"dist1"] = np.nan
    wea["dist2"] = frm["aws_dist2"].iloc[0]
    wea.loc[wea.loc[wea["temp2"].isna()].index.values,"dist2"] = np.nan
    wea["dist3"] = frm["aws_dist3"].iloc[0]
    wea.loc[wea.loc[wea["temp3"].isna()].index.values,"dist3"] = np.nan
    wea["totdist"] = wea[["dist1","dist2","dist3"]].sum(axis = 1, skipna = True)
    
    wea["f1"] = wea["totdist"] / wea["dist1"]
    wea["f2"] = wea["totdist"] / wea["dist2"]
    wea["f3"] = wea["totdist"] / wea["dist3"]
    wea["totf"] = wea[["f1","f2","f3"]].sum(axis = 1, skipna = True)      
    
    wea["w1"] = wea["f1"]  /  wea["totf"]
    wea["w2"] = wea["f2"]  /  wea["totf"]
    wea["w3"] = wea["f3"]  /  wea["totf"]
    wea.loc[wea.loc[wea["w1"].isna()].index.values,["temp1","rel_humidity1","wind_speed_avg_2m1"]] = -999
    wea.loc[wea.loc[wea["w1"].isna()].index.values,"w1"] = 0
    wea.loc[wea.loc[wea["w2"].isna()].index.values,["temp2","rel_humidity2","wind_speed_avg_2m2"]] = -999
    wea.loc[wea.loc[wea["w2"].isna()].index.values,"w2"] = 0
    wea.loc[wea.loc[wea["w3"].isna()].index.values,["temp3","rel_humidity3","wind_speed_avg_2m3"]] = -999
    wea.loc[wea.loc[wea["w3"].isna()].index.values,"w3"] = 0
    
    wea = wea.drop(columns = ["totdist",
                              "f1","f2","f3","totf"])
    
    # calculate weighted weather
    wea["temp"] = wea["w1"] * wea["temp1"] + \
                  wea["w2"] * wea["temp2"] + \
                  wea["w3"] * wea["temp3"]
    wea["rel_humidity"] = wea["w1"] * wea["rel_humidity1"] + \
                          wea["w2"] * wea["rel_humidity2"] + \
                          wea["w3"] * wea["rel_humidity3"]
    wea["wind_speed"] = wea["w1"] * wea["wind_speed_avg_2m1"] + \
                        wea["w2"] * wea["wind_speed_avg_2m2"] + \
                        wea["w3"] * wea["wind_speed_avg_2m3"]    
    
    # keep which aws used
    wea.loc[wea.loc[wea["temp1"].isna()].index.values,"aws_id1"] = np.nan
    wea.loc[wea.loc[wea["temp2"].isna()].index.values,"aws_id2"] = np.nan
    wea.loc[wea.loc[wea["temp3"].isna()].index.values,"aws_id3"] = np.nan
                        
    # select cols
    wea = wea[["datetime", "aws_id1","aws_id2","aws_id3",
               "dist1","dist2","dist3",
               "temp", "rel_humidity","wind_speed"]]

    fn = "weather_" + str(farm) + ".txt"
    wea.to_csv(os.path.join(path_out,fn))
    fn = "farm_" + str(farm) + ".txt"
    frm.to_csv(os.path.join(path_out,fn))
    del frm, wea, fn, sql_statement
    
    # scc ---------------------------------------------------------------------
    print("currently running... SCC")
    sql_statement = f"""
        SELECT dhi_id, farm_id, animal_id,parity_dhi,measured_on, dim, milk_day_kg, milk_kg, fat_day_percent, protein_day_percent, lactose_day_percent, scc, last_insemination_date
        FROM public.dhi
        WHERE farm_id IN (%s)
        ;
        """ % (str(farm))
    scc = pgres.query(query=sql_statement)
    fn = "scc_" + str(farm) + ".txt"
    scc.to_csv(os.path.join(path_out,fn))
    del scc, fn, sql_statement    
    
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



#%% raft database
with open(os.path.join(path,"serverSettings_raft.json")) as file:
    ssd_raft = json.load(file)

# Create a connection to the database
pgres_raft = LT_connect(**ssd_raft["s_parameters"]) 

# Create a SQL statment and query the statement
sql_statement = """
SELECT DISTINCT(farm_id) 
FROM activity
;
"""

df_farm_raft = pgres.query(query=sql_statement)


print(df_farm_raft)
