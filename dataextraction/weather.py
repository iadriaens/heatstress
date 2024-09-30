# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:45:30 2024

@author: u0084712


-------------------------------------------------------------------------------
Server connection requirements:
    Ivanti secure access VPN with MFA
    CertAgent
    cmd: ssh -L 55432:localhost:5432 u0084712@biosyst-s-lt01.bm.set.kuleuven.be

-------------------------------------------------------------------------------

# download weather data from meteostat directly
# calculate per hour features and save

"""


import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataextraction")
os.chdir(path)


#%% import packages for connection and extraction

import pandas as pd
from ServerConnect import LT_connect
import json
from meteostat import Hourly
Hourly.chuncked = False
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
# import numpy as np
from shapely.geometry import Point
# %matplotlib qt

path_out = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","data")


#%% create connection and connect to db

# Import the serverSettings as a dictionary from the given json parameter file.
with open(os.path.join(path,"serverSettings_CowBase.json")) as file:
    ssd = json.load(file)
    
# Create a connection to the database
pgres = LT_connect(**ssd["s_parameters"])

# set start and end date
startdate = datetime.datetime(2005, 1, 1)
enddate = datetime.datetime(2023, 12, 31, 23, 59)


#%% load data of weather stations

# all weather stations used in the database
sql_statement = f"""
    SELECT *
    FROM public.aws
    ;
    """
aws = pgres.query(query=sql_statement)
del sql_statement
aws.to_csv(os.path.join(path_out,"aws_all_stations_used.txt"))

# select hourly data from weather stations with ids in aws
wea = pd.DataFrame([])
for aws_id in aws["aws_id"]:
    if len(str(aws_id)) == 4:
        aws_id = '0'+str(aws_id)
    else:
        aws_id = str(aws_id)
    print(aws_id)
    df = Hourly(aws_id,startdate,enddate)
    df = df.fetch()
    df["aws_id"] = aws_id
    wea = pd.concat([df,wea])
del df, aws_id

# wea
wea = wea.loc[:,["aws_id","temp","rhum","tsun"]]
wea.to_csv(os.path.join(path_out, "weather_all_stations.txt"))


#%% calculate features 

# load data
wea = pd.read_csv(os.path.join(path_out,"weather_all_stations.txt"))
wea["time"] = pd.to_datetime(wea["time"],format = "%Y-%m-%d %H:%M:%S")

# add year, day of year and hour
wea["year"] = wea["time"].dt.year
wea["day"] = wea["time"].dt.dayofyear
wea["hour"] = wea["time"].dt.hour
wea= wea[["aws_id","time","year","day","hour","temp","rhum","tsun"]]


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



"""
#------------------------------------------------------------------------------
# determine classes > 100% RH = wet = stress from 20°C onwards (THI=68)
x=20
RH=100
y = 1.8 *x + 32 - \
                    ((0.55 - 0.0055 * RH) * \
                     (1.8 * x - 26))
print(y)

# temp when dry in summer on average temp at wettest moment
RH = 80
x=21.80
y = 1.8 *x + 32 - \
                    ((0.55 - 0.0055 * RH) * \
                     (1.8 * x - 26))
print(y)
"""


#%% calculate distance to coast
# load data
aws = pd.read_csv(os.path.join(path_out, "aws_all_stations_used.txt"), 
                  usecols=["aws_id","name","country","region","latitude","longitude"])

# load coast line
lines = gpd.read_file(os.path.join(path_out,"Europe_coastline_shapefile","Europe_coastline.shp"))
lines.to_crs("epsg:4326") #"EPSG:4326"
lines = lines.to_crs(lines.estimate_utm_crs())

aws["dist"]= 0
for i in aws.index.values:
    points_df = pd.DataFrame({'latitude' : [aws["latitude"].loc[i]],
                             'longitude': [aws["longitude"].loc[i]]})
    # print(i)
    points = gpd.GeoDataFrame(points_df, 
                              geometry=gpd.points_from_xy(points_df["longitude"], points_df["latitude"], 
                                                          crs="epsg:4326"))
    points = points.to_crs(lines.crs)
    for index, row in lines.iterrows():
        # print(index,row)
        lines.at[index, 'distance'] = row['geometry'].distance(points.iloc[0]['geometry'])
    
    pd.DataFrame({'Latitude': [51.08929], 'Longitude': [2.6443]}) # Middelkerke
    points = gpd.GeoDataFrame(points_df, geometry=gpd.points_from_xy(points_df["longitude"], points_df["latitude"], crs="epsg:4326"))
    lines = lines.sort_values(by=['distance'], ascending=True)
    aws["dist"][i] = lines["distance"].head(1)/1000 # minimum distance 


# threshold coastal: 15//50 kms from nearby coast line
aws.to_csv(os.path.join(path_out,"aws_all_stations_dist.txt"))



#%% select all weatherstations for which there is data, delete weather stations without
aws = pd.read_csv(os.path.join(path_out,"aws_all_stations_dist.txt"), index_col=0)

counts=(
        wea[["aws_id","year"]].groupby("aws_id")
                             .agg({"year":["count","min","max"]})
        ).reset_index()
aws_ids = counts.loc[(counts["year","count"]>0.8*(counts["year","max"]-counts["year","min"]+1)*365*24),"aws_id"]
aws_sel = aws.merge(aws_ids,how="inner")
aws_sel = aws_sel.sort_values(by="dist").reset_index(drop=1)
aws_sel = aws_sel.loc[aws_sel["latitude"]<52,:].reset_index(drop=1)



aws_sel.to_csv(os.path.join(path_out,"aws_all_stations_dist_selected.txt"))

# COASTAL = 15//50 kms from coastline
aws_inland = aws_sel.loc[aws_sel["dist"]>50,:]
aws_coast1 = aws_sel.loc[(aws_sel["dist"]<=50)&(aws_sel["dist"]>15),:]
aws_coast2 = aws_sel.loc[(aws_sel["dist"]<15),:]


#%% plot weather station locations
# set coordinate system
crs = {'init':'epsg:4326'}

# load data Belgium and provinces
geomap1 =  gpd.read_file(os.path.join(path_out,"BEL_adm","BEL_adm2.shp"))
geomap1 = geomap1.drop(axis=1,columns = ['ID_0', 'NAME_0', 'ID_1', 'NAME_1', 'ID_2', 'TYPE_2',
       'ENGTYPE_2', 'NL_NAME_2', 'VARNAME_2'])
geomap1.columns = ["country","province","geometry"]
geomap1 = geomap1.to_crs('EPSG:4326')

# dutch data and provinces
geomap2 = gpd.read_file(os.path.join(path_out,"NLD_adm","gadm41_NLD_1.shp"))
geomap2 = geomap2.drop(axis=1,columns = ['GID_1', 'COUNTRY', 'VARNAME_1', 'NL_NAME_1',
       'TYPE_1', 'ENGTYPE_1', 'CC_1', 'HASC_1', 'ISO_1'])
geomap2.columns = ["country","province","geometry"]
geomap2 = geomap2.to_crs('EPSG:4326')
geomap2 = geomap2.loc[[3,6,7,11,13],:].reset_index(drop=1)

# german data and provinces
geomap3 = gpd.read_file(os.path.join(path_out,"data","gadm41_DEU_1.shp"))
geomap3 = geomap3.drop(axis=1,columns = ['GID_1', 'COUNTRY', 'VARNAME_1', 'NL_NAME_1',
       'TYPE_1', 'ENGTYPE_1', 'CC_1', 'HASC_1', 'ISO_1'])
geomap3.columns = ["country","province","geometry"]
geomap3 = geomap3.to_crs('EPSG:4326')
geomap3 = geomap3.loc[[0,10,9,11],:].reset_index(drop=1)

# french data and provinces
geomap4 = gpd.read_file(os.path.join(path_out,"data","gadm41_FRA_1.shp"))
geomap4 = geomap4.drop(axis=1,columns = ['GID_1', 'COUNTRY', 'VARNAME_1', 'NL_NAME_1',
       'TYPE_1', 'ENGTYPE_1', 'CC_1', 'HASC_1', 'ISO_1'])
geomap4.columns = ["country","province","geometry"]
geomap4 = geomap4.to_crs('EPSG:4326')
geomap4 = geomap4.loc[[5,6],:].reset_index(drop=1)

# for coloring
geomap1["number"] = range(0,len(geomap1))
geomap2["number"] = range(0,len(geomap2))
geomap3["number"] = range(0,len(geomap3))
geomap4["number"] = range(0,len(geomap4))

# country borders
BE = gpd.read_file(os.path.join(path_out,"BEL_adm","BEL_adm0.shp"))
BE = BE.loc[:,["NAME_0","geometry"]]
BE = BE.to_crs('EPSG:4326')
NL = gpd.read_file(os.path.join(path_out,"NLD_adm","gadm41_NLD_0.shp"))
NL = NL.loc[:,["COUNTRY","geometry"]]
NL = NL.to_crs('EPSG:4326')
FR = gpd.read_file(os.path.join(path_out,"data","gadm41_FRA_0.shp"))
FR = FR.loc[:,["COUNTRY","geometry"]]
FR = FR.to_crs('EPSG:4326')
GE = gpd.read_file(os.path.join(path_out,"data","gadm41_DEU_0.shp"))
GE = GE.loc[:,["COUNTRY","geometry"]]
GE = GE.to_crs('EPSG:4326')

# geomap = gpd.pd.concat([geomap1,geomap2,geomap3,geomap4]).reset_index(drop=1)
# geomap["number"] = range(0,len(geomap))
# del geomap1,geomap2,geomap3,geomap4

# load aws data
# aws = pd.read_csv(os.path.join(path_data,"aws.txt"), index_col = 0)
geometry1 = [Point(xy) for xy in zip(aws_inland['longitude'], aws_inland['latitude'])]
aws_sel_df1 = gpd.GeoDataFrame(aws_inland,crs = crs, geometry = geometry1)
geometry2 = [Point(xy) for xy in zip(aws_coast1['longitude'], aws_coast1['latitude'])]
aws_sel_df2 = gpd.GeoDataFrame(aws_coast1,crs = crs, geometry = geometry2)
geometry3 = [Point(xy) for xy in zip(aws_coast2['longitude'], aws_coast2['latitude'])]
aws_sel_df3 = gpd.GeoDataFrame(aws_coast2,crs = crs, geometry = geometry3)

# plot all weatherstations
fig,ax = plt.subplots(figsize = (15,15))
geomap1.plot(ax=ax, alpha = 0.9, column = "number", cmap = "Blues")
geomap2.plot(ax=ax, alpha = 0.9, column = "number", cmap = "Purples")
geomap3.plot(ax=ax, alpha = 0.9, column = "number", cmap = "Greens")
geomap4.plot(ax=ax, alpha = 0.5, column = "number", cmap = "Reds")
BE.boundary.plot(alpha=1.0, linewidth = 1,edgecolor='b',ax=ax)
NL.boundary.plot(alpha=1.0,linewidth = 1, edgecolor='b',ax=ax)
FR.boundary.plot(alpha=1.0, linewidth = 1,edgecolor='b',ax=ax)
GE.boundary.plot(alpha=1.0, linewidth = 1,edgecolor='b',ax=ax)
ax.set_xlim(0.93,7)
ax.set_ylim(49.5,52)
aws_sel_df1.plot(ax=ax, marker = "*", color = "red", markersize = 200)
aws_sel_df2.plot(ax=ax, marker = "*", color = "darkorange", markersize = 200)
aws_sel_df3.plot(ax=ax, marker = "*", color = "gold", markersize = 200)
ax.set_title("Selected weather stations")
ax.set_xlabel("longitude [°E]", fontsize = "large")
ax.set_ylabel("latitude [°N]", fontsize = "large")

plt.savefig(os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataanalysis","results","thi","selected_aws_weather.tiff"))

del BE, FR, NL, GE, ax, fig, aws_sel_df1, aws_sel_df2, aws_sel_df3
del lines, row, index, geometry1, geometry2, geometry3, i, geomap1
del geomap2, geomap3, geomap4, crs
del aws_coast1, aws_coast2, counts, points, points_df
del aws_inland

#%% select data

aws_sel = pd.read_csv(os.path.join(path_out,"aws_all_stations_dist_selected.txt"))

#coast column
aws_sel["coast"] = 0
aws_sel.loc[(aws_sel["dist"]<=50)&(aws_sel["dist"]>15),"coast"] = 1
aws_sel.loc[(aws_sel["dist"]<15),"coast"] = 2

wea = (pd.merge(wea,aws_sel[["aws_id","coast"]], how = "inner")).reset_index(drop=1)


# per day summaries, per weather stations
data = (
        wea.groupby(by = ["aws_id","year","day"])
        .agg({"coast":["min",'count'],
              "temp":["min","max","mean"],
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
data.columns = ["aws_id","year","day","coast","nobs",
                "Tmin","Tmax","Tavg",
                "RHmin","RHmax","RHavg",
                "THImin","THImax","THIavg",
                "HS0","HS1","HS2","HS3","HS4"]



# add to data TIME HS classes
# min time HS0
idx = (
       wea.loc[:,["aws_id","year","day","HS0","hour"]]
       .sort_values(by = ["aws_id","year","day","HS0","hour"], 
                    ascending = [True,True,True,False,True])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["aws_id","year","day"])
       ).index
sub = wea.loc[idx,["aws_id","year","day","hour"]].rename(columns = {"hour":"HS0_1"})
data = data.merge(sub,how="inner")
# max time HS0
idx = (
       wea.loc[:,["aws_id","year","day","HS0","hour"]]
       .sort_values(by = ["aws_id","year","day","HS0","hour"], 
                    ascending = [True,True,True,False,False])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["aws_id","year","day"])
       ).index
sub = wea.loc[idx,["aws_id","year","day","hour"]].rename(columns = {"hour":"HS0_2"})
data = data.merge(sub,how="inner")
# min time HS1
idx = (
       wea.loc[:,["aws_id","year","day","HS1","hour"]]
       .sort_values(by = ["aws_id","year","day","HS1","hour"], 
                    ascending = [True,True,True,False,True])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["aws_id","year","day"])
       ).index
sub = wea.loc[idx,["aws_id","year","day","hour"]].rename(columns = {"hour":"HS1_1"})
data = data.merge(sub,how="inner")
# max time HS1
idx = (
       wea.loc[:,["aws_id","year","day","HS1","hour"]]
       .sort_values(by = ["aws_id","year","day","HS1","hour"], 
                    ascending = [True,True,True,False,False])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["aws_id","year","day"])
       ).index
sub = wea.loc[idx,["aws_id","year","day","hour"]].rename(columns = {"hour":"HS1_2"})
data = data.merge(sub,how="inner")
# min time HS2
idx = (
       wea.loc[:,["aws_id","year","day","HS2","hour"]]
       .sort_values(by = ["aws_id","year","day","HS2","hour"], 
                    ascending = [True,True,True,False,True])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["aws_id","year","day"])
       ).index
sub = wea.loc[idx,["aws_id","year","day","hour"]].rename(columns = {"hour":"HS2_1"})
data = data.merge(sub,how="inner")
# max time HS2
idx = (
       wea.loc[:,["aws_id","year","day","HS2","hour"]]
       .sort_values(by = ["aws_id","year","day","HS2","hour"], 
                    ascending = [True,True,True,False,False])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["aws_id","year","day"])
       ).index
sub = wea.loc[idx,["aws_id","year","day","hour"]].rename(columns = {"hour":"HS2_2"})
data = data.merge(sub,how="inner")
# min time HS3
idx = (
       wea.loc[:,["aws_id","year","day","HS3","hour"]]
       .sort_values(by = ["aws_id","year","day","HS3","hour"], 
                    ascending = [True,True,True,False,True])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["aws_id","year","day"])
       ).index
sub = wea.loc[idx,["aws_id","year","day","hour"]].rename(columns = {"hour":"HS3_1"})
data = data.merge(sub,how="inner")
# max time HS3
idx = (
       wea.loc[:,["aws_id","year","day","HS3","hour"]]
       .sort_values(by = ["aws_id","year","day","HS3","hour"], 
                    ascending = [True,True,True,False,False])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["aws_id","year","day"])
       ).index
sub = wea.loc[idx,["aws_id","year","day","hour"]].rename(columns = {"hour":"HS3_2"})
data = data.merge(sub,how="inner")
# min time HS4
idx = (
       wea.loc[:,["aws_id","year","day","HS4","hour"]]
       .sort_values(by = ["aws_id","year","day","HS4","hour"], 
                    ascending = [True,True,True,False,True])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["aws_id","year","day"])
       ).index
sub = wea.loc[idx,["aws_id","year","day","hour"]].rename(columns = {"hour":"HS4_1"})
data = data.merge(sub,how="inner")
# max time HS3
idx = (
       wea.loc[:,["aws_id","year","day","HS4","hour"]]
       .sort_values(by = ["aws_id","year","day","HS4","hour"], 
                    ascending = [True,True,True,False,False])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["aws_id","year","day"])
       ).index
sub = wea.loc[idx,["aws_id","year","day","hour"]].rename(columns = {"hour":"HS4_2"})
data = data.merge(sub,how="inner")





# time min temp
idx = (
       wea.loc[:,["aws_id","year","day","temp"]]
       .sort_values(by = ["aws_id","year","day","temp"], 
                    ascending = [True,True,True,True])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["aws_id","year","day"])
       ).index
sub = wea.loc[idx,["aws_id","year","day","hour"]].rename(columns = {"hour":"Tmin_h"})
data = data.merge(sub,how="inner")
# time max temp
idx = (
       wea.loc[:,["aws_id","year","day","temp"]]
       .sort_values(by = ["aws_id","year","day","temp"], 
                    ascending = [True,True,True,False])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["aws_id","year","day"])
       ).index
sub = wea.loc[idx,["aws_id","year","day","hour"]].rename(columns = {"hour":"Tmax_h"})
data = data.merge(sub,how="inner")

# time min rh
idx = (
       wea.loc[:,["aws_id","year","day","rhum"]]
       .sort_values(by = ["aws_id","year","day","rhum"], 
                    ascending = [True,True,True,True])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["aws_id","year","day"])
       ).index
sub = wea.loc[idx,["aws_id","year","day","hour"]].rename(columns = {"hour":"RHmin_h"})
data = data.merge(sub,how="inner")
# time max temp
idx = (
       wea.loc[:,["aws_id","year","day","rhum"]]
       .sort_values(by = ["aws_id","year","day","rhum"], 
                    ascending = [True,True,True,False])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["aws_id","year","day"])
       ).index
sub = wea.loc[idx,["aws_id","year","day","hour"]].rename(columns = {"hour":"RHmax_h"})
data = data.merge(sub,how="inner")


# delete from data days with < 24h measurements (9.3%)
data = data.loc[data["nobs"]>23].reset_index(drop=1)

# save data
data.to_csv(os.path.join(path_out,"weather_all_stations_sum.txt"))
wea.to_csv(os.path.join(path_out,"weather_all_stations_selected.txt"))


#%% summary per region, then quantify

# read weather data
wea = pd.read_csv(os.path.join(path_out,"weather_all_stations_selected.txt"), index_col = 0)

df = wea[["coast","year","day","hour","temp","rhum","thi"]].groupby(by = ["coast","year","day","hour"]).mean().reset_index()

# set threshold columns [0;64[ - [64;68[ - [68;72[ - [72;80[ - [80;100]
df["HS0"] = (df["thi"]<64).astype(int)
df["HS1"] = ((df["thi"]>=64)&(df["thi"]<68)).astype(int)
df["HS2"] = ((df["thi"]>=68)&(df["thi"]<72)).astype(int)
df["HS3"] = ((df["thi"]>=72)&(df["thi"]<80)).astype(int)
df["HS4"] = (df["thi"]>=80).astype(int)

# per day summaries, per weather stations
data = (
        df.groupby(by = ["coast","year","day"])
        .agg({
              "temp":["min","max","mean"],
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
data.columns = ["coast","year","day",
                "Tmin","Tmax","Tavg",
                "RHmin","RHmax","RHavg",
                "THImin","THImax","THIavg",
                "HS0","HS1","HS2","HS3","HS4"]

# add to data TIME HS classes
# min time HS0
idx = (
       df.loc[df["HS0"]==1,["coast","year","day","HS0","hour"]]
       .sort_values(by = ["coast","year","day","HS0","hour"], 
                    ascending = [True,True,True,False,True])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["coast","year","day"])
       ).index
sub = df.loc[idx,["coast","year","day","hour"]].rename(columns = {"hour":"HS0_1"})
data = data.merge(sub,how="outer")
# max time HS0
idx = (
       df.loc[df["HS0"]==1,["coast","year","day","HS0","hour"]]
       .sort_values(by = ["coast","year","day","HS0","hour"], 
                    ascending = [True,True,True,False,False])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["coast","year","day"])
       ).index
sub = df.loc[idx,["coast","year","day","hour"]].rename(columns = {"hour":"HS0_2"})
data = data.merge(sub,how="outer")
# min time HS1
idx = (
       df.loc[df["HS1"]==1,["coast","year","day","HS1","hour"]]
       .sort_values(by = ["coast","year","day","HS1","hour"], 
                    ascending = [True,True,True,False,True])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["coast","year","day"])
       ).index
sub = df.loc[idx,["coast","year","day","hour"]].rename(columns = {"hour":"HS1_1"})
data = data.merge(sub,how="outer")
# max time HS1
idx = (
       df.loc[df["HS1"]==1,["coast","year","day","HS1","hour"]]
       .sort_values(by = ["coast","year","day","HS1","hour"], 
                    ascending = [True,True,True,False,False])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["coast","year","day"])
       ).index
sub = df.loc[idx,["coast","year","day","hour"]].rename(columns = {"hour":"HS1_2"})
data = data.merge(sub,how="outer")
# min time HS2
idx = (
       df.loc[df["HS2"]==1,["coast","year","day","HS2","hour"]]
       .sort_values(by = ["coast","year","day","HS2","hour"], 
                    ascending = [True,True,True,False,True])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["coast","year","day"])
       ).index
sub = df.loc[idx,["coast","year","day","hour"]].rename(columns = {"hour":"HS2_1"})
data = data.merge(sub,how="outer")
# max time HS2
idx = (
       df.loc[df["HS2"]==1,["coast","year","day","HS2","hour"]]
       .sort_values(by = ["coast","year","day","HS2","hour"], 
                    ascending = [True,True,True,False,False])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["coast","year","day"])
       ).index
sub = df.loc[idx,["coast","year","day","hour"]].rename(columns = {"hour":"HS2_2"})
data = data.merge(sub,how="outer")
# min time HS3
idx = (
       df.loc[df["HS3"]==1,["coast","year","day","HS3","hour"]]
       .sort_values(by = ["coast","year","day","HS3","hour"], 
                    ascending = [True,True,True,False,True])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["coast","year","day"])
       ).index
sub = df.loc[idx,["coast","year","day","hour"]].rename(columns = {"hour":"HS3_1"})
data = data.merge(sub,how="outer")
# max time HS3
idx = (
       df.loc[df["HS3"]==1,["coast","year","day","HS3","hour"]]
       .sort_values(by = ["coast","year","day","HS3","hour"], 
                    ascending = [True,True,True,False,False])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["coast","year","day"])
       ).index
sub = df.loc[idx,["coast","year","day","hour"]].rename(columns = {"hour":"HS3_2"})
data = data.merge(sub,how="outer")
# min time HS4
idx = (
       df.loc[df["HS4"]==1,["coast","year","day","HS4","hour"]]
       .sort_values(by = ["coast","year","day","HS4","hour"], 
                    ascending = [True,True,True,False,True])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["coast","year","day"])
       ).index
sub = df.loc[idx,["coast","year","day","hour"]].rename(columns = {"hour":"HS4_1"})
data = data.merge(sub,how="outer")
# max time HS3
idx = (
       df.loc[df["HS4"]==1,["coast","year","day","HS4","hour"]]
       .sort_values(by = ["coast","year","day","HS4","hour"], 
                    ascending = [True,True,True,False,False])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["coast","year","day"])
       ).index
sub = df.loc[idx,["coast","year","day","hour"]].rename(columns = {"hour":"HS4_2"})
data = data.merge(sub,how="outer")

# time min temp
idx = (
       df.loc[:,["coast","year","day","temp"]]
       .sort_values(by = ["coast","year","day","temp"], 
                    ascending = [True,True,True,True])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["coast","year","day"])
       ).index
sub = df.loc[idx,["coast","year","day","hour"]].rename(columns = {"hour":"Tmin_h"})
data = data.merge(sub,how="inner")
# time max temp
idx = (
       df.loc[:,["coast","year","day","temp"]]
       .sort_values(by = ["coast","year","day","temp"], 
                    ascending = [True,True,True,False])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["coast","year","day"])
       ).index
sub = df.loc[idx,["coast","year","day","hour"]].rename(columns = {"hour":"Tmax_h"})
data = data.merge(sub,how="inner")

# time min rh
idx = (
       df.loc[:,["coast","year","day","rhum"]]
       .sort_values(by = ["coast","year","day","rhum"], 
                    ascending = [True,True,True,True])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["coast","year","day"])
       ).index
sub = df.loc[idx,["coast","year","day","hour"]].rename(columns = {"hour":"RHmin_h"})
data = data.merge(sub,how="inner")
# time max temp
idx = (
       df.loc[:,["coast","year","day","rhum"]]
       .sort_values(by = ["coast","year","day","rhum"], 
                    ascending = [True,True,True,False])  # HS = 1, sorting for first hour
       .drop_duplicates(subset = ["coast","year","day"])
       ).index
sub = df.loc[idx,["coast","year","day","hour"]].rename(columns = {"hour":"RHmax_h"})
data = data.merge(sub,how="inner")

fig, ax = plt.subplots(1,4, figsize = (21,4))
test = data[["Tmin_h","RHmax_h"]].value_counts().reset_index().pivot(index="Tmin_h", columns="RHmax_h", values=0)
test=test.fillna(0)
sns.heatmap(test, ax = ax[0])
ax[0].set_title("all data")

test = data.loc[(data["coast"]==0),["Tmin_h","RHmax_h"]].value_counts().reset_index().pivot(index="Tmin_h", columns="RHmax_h", values=0)
test=test.fillna(0)
sns.heatmap(test, ax = ax[1])
ax[1].set_title("coast < 15kms")

test = data.loc[(data["coast"]==1),["Tmin_h","RHmax_h"]].value_counts().reset_index().pivot(index="Tmin_h", columns="RHmax_h", values=0)
test=test.fillna(0)
sns.heatmap(test, ax = ax[2])
ax[2].set_title("coast 15-50 kms")

test = data.loc[(data["coast"]==2),["Tmin_h","RHmax_h"]].value_counts().reset_index().pivot(index="Tmin_h", columns="RHmax_h", values=0)
test=test.fillna(0)
sns.heatmap(test, ax = ax[3])
ax[3].set_title("inland")

# summer only
fig, ax = plt.subplots(1,4, figsize = (21,4))
test = data.loc[(data["day"] > 150)&(data["day"] < 244),["Tmin_h","RHmax_h"]].value_counts().reset_index().pivot(index="Tmin_h", columns="RHmax_h", values=0)
test=test.fillna(0)
sns.heatmap(test, ax = ax[0])
ax[0].set_title("all data")

test = data.loc[(data["day"] > 150)&(data["day"] < 244)&(data["coast"]==0),["Tmin_h","RHmax_h"]].value_counts().reset_index().pivot(index="Tmin_h", columns="RHmax_h", values=0)
test=test.fillna(0)
sns.heatmap(test, ax = ax[1])
ax[1].set_title("coast < 15kms")

test = data.loc[(data["day"] > 150)&(data["day"] < 244)&(data["coast"]==1),["Tmin_h","RHmax_h"]].value_counts().reset_index().pivot(index="Tmin_h", columns="RHmax_h", values=0)
test=test.fillna(0)
sns.heatmap(test, ax = ax[2])
ax[2].set_title("coast 15-50 kms")

test = data.loc[(data["day"] > 150)&(data["day"] < 244)&(data["coast"]==2),["Tmin_h","RHmax_h"]].value_counts().reset_index().pivot(index="Tmin_h", columns="RHmax_h", values=0)
test=test.fillna(0)
sns.heatmap(test, ax = ax[3])
ax[3].set_title("inland")


# summer only
fig, ax = plt.subplots(1,4, figsize = (21,4))
test = data.loc[(data["day"] > 150)&(data["day"] < 244),["Tmax_h","RHmax_h"]].value_counts().reset_index().pivot(index="Tmax_h", columns="RHmax_h", values=0)
test=test.fillna(0)
sns.heatmap(test, ax = ax[0])
ax[0].set_title("all data")

test = data.loc[(data["day"] > 150)&(data["day"] < 244)&(data["coast"]==0),["Tmax_h","RHmax_h"]].value_counts().reset_index().pivot(index="Tmax_h", columns="RHmax_h", values=0)
test=test.fillna(0)
sns.heatmap(test, ax = ax[1])
ax[1].set_title("coast < 15kms")

test = data.loc[(data["day"] > 150)&(data["day"] < 244)&(data["coast"]==1),["Tmax_h","RHmax_h"]].value_counts().reset_index().pivot(index="Tmax_h", columns="RHmax_h", values=0)
test=test.fillna(0)
sns.heatmap(test, ax = ax[2])
ax[2].set_title("coast 15-50 kms")

test = data.loc[(data["day"] > 150)&(data["day"] < 244)&(data["coast"]==2),["Tmax_h","RHmax_h"]].value_counts().reset_index().pivot(index="Tmax_h", columns="RHmax_h", values=0)
test=test.fillna(0)
sns.heatmap(test, ax = ax[3])
ax[3].set_title("inland")




# save data
data.to_csv(os.path.join(path_out,"weather_all_stations_sum2.txt"))
wea.to_csv(os.path.join(path_out,"weather_all_stations_selected2.txt"))




#------------------------------------------------------------------------------

# summarize over all weather stations in all regions

# variables: 
"""
Tmin + std : average min temp on all farms
Tmax + std : average max temp on all farms
Tavg + std : average mean temp on al farms
RHmin + std : average min RH on all farms
RHmax + std : average max RH on all farms
RHavg + std : average mean RH on all farms
THImin + std : average mean THI on all farms
THImax + std : average mean THI on all farms
THIavg + std : average mean THI on all farms
HS0avg + std : average no. of hours in HS0
HS1avg + std : average no. of hours in HS1
HS2avg + std : average no. of hours in HS2
HS3avg + std : average no. of hours in HS3
HS0_f_avg + std : average time HS0 first
HS0_l_avg + std : average time HS0 last
HS1_f_avg + std : average time HS1 first
HS1_l_avg + std : average time HS1 last
HS2_f_avg + std : average time HS2 first
HS2_l_avg + std : average time HS2 first
HS3_f_avg + std : average time HS3 last
HS3_l_avg + std : average time HS3 last

"""




aws_id = 10401
df = (wea.loc[wea.aws_id == aws_id,:]).copy()


#
df["season"] = "winter"
df.loc[df["day"] > 59,"season"] = "spring"
df.loc[df["day"] > 151,"season"] = "summer"
df.loc[df["day"] > 243,"season"] = "autumn"
df.loc[df["day"] > 334,"season"] = "winter"
sns.set(font_scale=1.1)


# temp and relative humidity
# _,ax = plt.subplots(1,1,figsize = (18,15))
g = sns.pairplot(df.iloc[:,[5,6,8,-1]], 
                 diag_kind="kde",
                 hue = "season",
                 palette = {"winter" : "#3333FF","spring" : "#00CC66","summer" : "#CC0000","autumn":"#F97306"},
                 corner = False,
                 # x_vars  = ["temp_min","temp_max","rh_min","rh_max","thi_avg","thi_max"],
                 # y_vars = ["temp_min","temp_max","rh_min","rh_max","thi_avg","thi_max"]
                 )
labels = {"temp":"T°",
          "rhum":"RH",
         "thi":"thi"}
# set labels
for i in range(3):
    for j in range(3):
        xlabel = g.axes[i][j].get_xlabel()
        ylabel = g.axes[i][j].get_ylabel()
        if xlabel in labels.keys():
            g.axes[i][j].set_xlabel(labels[xlabel])
        if ylabel in labels.keys():
            g.axes[i][j].set_ylabel(labels[ylabel])

# set lines and white triangle
for i in range(3):
    for j in range(3):
        x = g.axes[i][j].get_xlim()
        y = g.axes[i][j].get_ylim()
        if j>i:
            x = g.axes[i][j].get_xlim()
            y = g.axes[i][j].get_ylim()
            g.axes[i][j].fill_between(x,y[0],y[1],color = 'w')
        g.axes[i][j].set_xlim(x)
        g.axes[i][j].set_ylim(y)

g.axes[0][1].set_title("AWS = " + (aws.loc[(aws["aws_id"]==aws_id),"name"]).iloc[0])
        
