# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:39:43 2023

@author: u0084712
"""


import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","datapreprocessing")
os.chdir(path)


#%% import packages

import pandas as pd
import numpy as np
import seaborn as sns
# import statsmodels
import matplotlib.pyplot as plt
from datetime import date
import openpyxl

#%% file path

path_data = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","datapreprocessing",
                    "results")

# farm selected
farms = [30,31,33,34,35,38,39,40,43,44,45,46,48] 



#%% quantify number of days data available per farm


sumtable = pd.DataFrame([], columns = ["farm", 
                                       "milk_start","milk_end","milk_nodays",
                                       "milk_cows","milk_lac",
                                       "act_start","act_end","act_nodays",
                                       "act_cows","act_lac",
                                       "scc_start","scc_end","scc_nodays",
                                       "scc_cows","scc_lac"])
sumtable["farm"] = farms
counter = 0
for farm in farms: 
    
    # milk production:
    milk = pd.read_csv(os.path.join(path_data,"milk", "milk_preprocessed_" 
                                    + str(farm) + ".txt"), index_col=0)
    milk["date"] = pd.to_datetime(milk["date"],format='%Y-%m-%d')
    print("first meas = " + str(milk.date.min()))
    print("last meas = " + str(milk.date.max()))
    print("number of days data = " +str((milk.date.max()-milk.date.min()).days) )
    
    cows = milk.groupby(by=["animal_id"]).count()
    print("number of cow lactations = " + str(len(cows)))
    cowlac = milk.groupby(by=["animal_id","parity"]).count()
    print("number of cow lactations = " + str(len(cowlac)))
        
    # put in table
    sumtable.loc[counter,"milk_start"] = milk.date.min()
    sumtable.loc[counter,"milk_end"] = milk.date.max()
    sumtable.loc[counter,"milk_nodays"] = (milk.date.max()-milk.date.min()).days
    sumtable.loc[counter,"milk_cows"] = len(cows)
    sumtable.loc[counter,"milk_lac"] = len(cowlac)
    
    # activity
    act = pd.read_csv(os.path.join(path_data,"activity", "act_preprocessed_" 
                                    + str(farm) + ".txt"), index_col=0)
    act["date"] = pd.to_datetime(act["date"],format='%Y-%m-%d')
    print("first meas = " + str(act.date.min()))
    print("last meas = " + str(act.date.max()))
    print("number of days data = " +str((act.date.max()-act.date.min()).days) )
    
    cows = act.groupby(by=["animal_id"]).count()
    print("number of cow lactations = " + str(len(cows)))
    cowlac = act.groupby(by=["animal_id","parity"]).count()
    print("number of cow lactations = " + str(len(cowlac)))
        
    # put in table
    sumtable.loc[counter,"act_start"] = act.date.min()
    sumtable.loc[counter,"act_end"] = act.date.max()
    sumtable.loc[counter,"act_nodays"] = (act.date.max()-act.date.min()).days
    sumtable.loc[counter,"act_cows"] = len(cows)
    sumtable.loc[counter,"act_lac"] = len(cowlac)
    
    # scc
    scc = pd.read_csv(os.path.join(path_data,"scc", "scc_preprocessed_" 
                                    + str(farm) + ".txt"), index_col=0)
    scc["date"] = pd.to_datetime(scc["measured_on"],format='%Y-%m-%d')
    scc["parity"] = scc["parity_dhi"]
    print("first meas = " + str(scc.date.min()))
    print("last meas = " + str(scc.date.max()))
    print("number of days data = " +str((scc.date.max()-scc.date.min()).days) )
    
    cows = scc.groupby(by=["animal_id"]).count()
    print("number of cow lactations = " + str(len(cows)))
    cowlac = scc.groupby(by=["animal_id","parity"]).count()
    print("number of cow lactations = " + str(len(cowlac)))
        
    # put in table
    sumtable.loc[counter,"scc_start"] = scc.date.min()
    sumtable.loc[counter,"scc_end"] = scc.date.max()
    sumtable.loc[counter,"scc_nodays"] = (scc.date.max()-scc.date.min()).days
    sumtable.loc[counter,"scc_cows"] = len(cows)
    sumtable.loc[counter,"scc_lac"] = len(cowlac)
        
    counter = counter + 1
    
    
    
# excel writer
writer = pd.ExcelWriter(os.path.join(path,"summary_data.xlsx"), engine = 'openpyxl')
sumtable.to_excel(writer,sheet_name = "data_overview", index=False)
# save and close
writer.save()
writer.close()
del writer


#%% Data description per farm - DAILY MILK YIELD

# prepare summary
sumtable = pd.DataFrame([], columns = ["farm", "no_cows",
                                       "avg_305","std_305",
                                       "no_par1","avg_305_1","std_305_1",
                                       "no_par2","avg_305_2","std_305_2",
                                       "no_par3","avg_305_3","std_305_3",
                                       "avg_daily","std_daily",
                                       "avg_daily","std_daily",
                                       "avg_daily","std_daily",
                                       "avg_daily","std_daily"])                                    

sumtable["farm"] = farms
counter = 0
for farm in farms: 
    
    # milk production:
    milk = pd.read_csv(os.path.join(path_data,"milk", "milk_preprocessed_" 
                                    + str(farm) + ".txt"), index_col=0)
    milk["date"] = pd.to_datetime(milk["date"],format='%Y-%m-%d')
    
    # select first 305 days of data
    milk = milk.loc[milk["dim"] < 306,:]
       
    # summary per cowlac 305 day milk production, if > 300 days
    cowlac = (
        milk[["animal_id","parity","dmy"]]
        .groupby(by = ["animal_id","parity"]).count()
        .reset_index()
        )
    cowlac = cowlac.loc[cowlac["dmy"] > 300]
    newmilk = pd.merge(milk,cowlac[["animal_id","parity"]], how= "inner",on = ["animal_id","parity"])
    
    # these are NOT all cows in the dataset, just those to calculate 305day MY on
    
    # number of cows 305 day milk production
    sumtable.loc[counter,"no_cows"] = len(newmilk[["animal_id","parity"]].drop_duplicates())
    sumtable.loc[counter,"no_par1"] = len(newmilk.loc[newmilk["parity"]==1,["animal_id","parity"]].drop_duplicates())
    sumtable.loc[counter,"no_par2"] = len(newmilk.loc[newmilk["parity"]==2,["animal_id","parity"]].drop_duplicates())
    sumtable.loc[counter,"no_par3"] = len(newmilk.loc[newmilk["parity"]>2,["animal_id","parity"]].drop_duplicates())
    
    #  --- all cows
    sumtable.loc[counter,"avg_305"] = round((
                        newmilk[["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).sum()
                        ).mean(),2).dmy
    sumtable.loc[counter,"std_305"] = round((
                        newmilk[["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).sum()
                        ).std(),2).dmy
    
    #  --- parity 1
    sumtable.loc[counter,"avg_305_1"] = round((
                        newmilk.loc[newmilk["parity"] == 1, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).sum()
                        ).mean(),2).dmy
    sumtable.loc[counter,"std_305_1"] = round((
                       newmilk.loc[newmilk["parity"] == 1, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).sum()
                        ).std(),2).dmy
    
    #  --- parity 2
    sumtable.loc[counter,"avg_305_2"] = round((
                        newmilk.loc[newmilk["parity"] == 2, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).sum()
                        ).mean(),2).dmy
    sumtable.loc[counter,"std_305_2"] = round((
                       newmilk.loc[newmilk["parity"] == 2, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).sum()
                        ).std(),2).dmy
    
    #  --- parity 3+
    sumtable.loc[counter,"avg_305_3"] = round((
                        newmilk.loc[newmilk["parity"] > 2, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).sum()
                        ).mean(),2).dmy
    sumtable.loc[counter,"std_305_3"] = round((
                       newmilk.loc[newmilk["parity"] > 2, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).sum()
                        ).std(),2).dmy
    
    
    # -------------------------  avg daily milk yield -------------------------
    #  --- all cows
    sumtable.loc[counter,"avg_daily"] = round((
                        newmilk[["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).mean()
                        ).mean(),2).dmy
    sumtable.loc[counter,"std_daily"] = round((
                        newmilk[["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).mean()
                        ).std(),2).dmy
    
    #  --- parity 1
    sumtable.loc[counter,"avg_daily_1"] = round((
                        newmilk.loc[newmilk["parity"] == 1, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).mean()
                        ).mean(),2).dmy
    sumtable.loc[counter,"std_daily_1"] = round((
                       newmilk.loc[newmilk["parity"] == 1, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).mean()
                        ).std(),2).dmy
    
    #  --- parity 2
    sumtable.loc[counter,"avg_daily_2"] = round((
                        newmilk.loc[newmilk["parity"] == 2, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).mean()
                        ).mean(),2).dmy
    sumtable.loc[counter,"std_daily_2"] = round((
                       newmilk.loc[newmilk["parity"] == 2, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).mean()
                        ).std(),2).dmy
    
    #  --- parity 3+
    sumtable.loc[counter,"avg_daily_3"] = round((
                        newmilk.loc[newmilk["parity"] > 2, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).mean()
                        ).mean(),2).dmy
    sumtable.loc[counter,"std_daily_3"] = round((
                       newmilk.loc[newmilk["parity"] > 2, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).mean()
                        ).std(),2).dmy
    
    counter = counter + 1
    
# excel writer
writer = pd.ExcelWriter(os.path.join(path,"summary_data.xlsx"), engine = 'openpyxl')
sumtable.to_excel(writer,sheet_name = "dmy_overview", index=False)
# save and close
writer.save()
writer.close()
del writer  
    
#%% Data description per farm -  ACTIVITY

#TODO!!!
    
sumtable = pd.DataFrame([], columns = ["farm", "no_cows",
                                       "avg_305","std_305",
                                       "no_par1","avg_305_1","std_305_1",
                                       "no_par2","avg_305_2","std_305_2",
                                       "no_par3","avg_305_3","std_305_3",
                                       "avg_daily","std_daily",
                                       "avg_daily","std_daily",
                                       "avg_daily","std_daily",
                                       "avg_daily","std_daily"])                                    

sumtable["farm"] = farms
counter = 0
for farm in farms: 
    
    # milk production:
    milk = pd.read_csv(os.path.join(path_data,"milk", "milk_preprocessed_" 
                                    + str(farm) + ".txt"), index_col=0)
    milk["date"] = pd.to_datetime(milk["date"],format='%Y-%m-%d')
    
    # select first 305 days of data
    milk = milk.loc[milk["dim"] < 306,:]
       
    # summary per cowlac 305 day milk production, if > 300 days
    cowlac = (
        milk[["animal_id","parity","dmy"]]
        .groupby(by = ["animal_id","parity"]).count()
        .reset_index()
        )
    cowlac = cowlac.loc[cowlac["dmy"] > 300]
    newmilk = pd.merge(milk,cowlac[["animal_id","parity"]], how= "inner",on = ["animal_id","parity"])
    
    # these are NOT all cows in the dataset, just those to calculate 305day MY on
    
    # number of cows 305 day milk production
    sumtable.loc[counter,"no_cows"] = len(newmilk[["animal_id","parity"]].drop_duplicates())
    sumtable.loc[counter,"no_par1"] = len(newmilk.loc[newmilk["parity"]==1,["animal_id","parity"]].drop_duplicates())
    sumtable.loc[counter,"no_par2"] = len(newmilk.loc[newmilk["parity"]==2,["animal_id","parity"]].drop_duplicates())
    sumtable.loc[counter,"no_par3"] = len(newmilk.loc[newmilk["parity"]>2,["animal_id","parity"]].drop_duplicates())
    
    #  --- all cows
    sumtable.loc[counter,"avg_305"] = round((
                        newmilk[["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).sum()
                        ).mean(),2).dmy
    sumtable.loc[counter,"std_305"] = round((
                        newmilk[["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).sum()
                        ).std(),2).dmy
    
    #  --- parity 1
    sumtable.loc[counter,"avg_305_1"] = round((
                        newmilk.loc[newmilk["parity"] == 1, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).sum()
                        ).mean(),2).dmy
    sumtable.loc[counter,"std_305_1"] = round((
                       newmilk.loc[newmilk["parity"] == 1, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).sum()
                        ).std(),2).dmy
    
    #  --- parity 2
    sumtable.loc[counter,"avg_305_2"] = round((
                        newmilk.loc[newmilk["parity"] == 2, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).sum()
                        ).mean(),2).dmy
    sumtable.loc[counter,"std_305_2"] = round((
                       newmilk.loc[newmilk["parity"] == 2, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).sum()
                        ).std(),2).dmy
    
    #  --- parity 3+
    sumtable.loc[counter,"avg_305_3"] = round((
                        newmilk.loc[newmilk["parity"] > 2, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).sum()
                        ).mean(),2).dmy
    sumtable.loc[counter,"std_305_3"] = round((
                       newmilk.loc[newmilk["parity"] > 2, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).sum()
                        ).std(),2).dmy
    
    
    # -------------------------  avg daily milk yield -------------------------
    #  --- all cows
    sumtable.loc[counter,"avg_daily"] = round((
                        newmilk[["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).mean()
                        ).mean(),2).dmy
    sumtable.loc[counter,"std_daily"] = round((
                        newmilk[["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).mean()
                        ).std(),2).dmy
    
    #  --- parity 1
    sumtable.loc[counter,"avg_daily_1"] = round((
                        newmilk.loc[newmilk["parity"] == 1, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).mean()
                        ).mean(),2).dmy
    sumtable.loc[counter,"std_daily_1"] = round((
                       newmilk.loc[newmilk["parity"] == 1, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).mean()
                        ).std(),2).dmy
    
    #  --- parity 2
    sumtable.loc[counter,"avg_daily_2"] = round((
                        newmilk.loc[newmilk["parity"] == 2, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).mean()
                        ).mean(),2).dmy
    sumtable.loc[counter,"std_daily_2"] = round((
                       newmilk.loc[newmilk["parity"] == 2, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).mean()
                        ).std(),2).dmy
    
    #  --- parity 3+
    sumtable.loc[counter,"avg_daily_3"] = round((
                        newmilk.loc[newmilk["parity"] > 2, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).mean()
                        ).mean(),2).dmy
    sumtable.loc[counter,"std_daily_3"] = round((
                       newmilk.loc[newmilk["parity"] > 2, ["animal_id","parity","dmy"]]
                        .groupby(by = ["animal_id","parity"]).mean()
                        ).std(),2).dmy
    
# excel writer
writer = pd.ExcelWriter(os.path.join(path,"summary_data.xlsx"), engine = 'openpyxl')
sumtable.to_excel(writer,sheet_name = "dmy_overview", index=False)
# save and close
writer.save()
writer.close()
del writer  
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # activity
    act = pd.read_csv(os.path.join(path_data,"activity", "act_preprocessed_" 
                                    + str(farm) + ".txt"), index_col=0)
    act["date"] = pd.to_datetime(act["date"],format='%Y-%m-%d')
    print("first meas = " + str(act.date.min()))
    print("last meas = " + str(act.date.max()))
    print("number of days data = " +str((act.date.max()-act.date.min()).days) )
    
    cows = act.groupby(by=["animal_id"]).count()
    print("number of cow lactations = " + str(len(cows)))
    cowlac = act.groupby(by=["animal_id","parity"]).count()
    print("number of cow lactations = " + str(len(cowlac)))
        
    # put in table
    sumtable.loc[counter,"act_start"] = act.date.min()
    sumtable.loc[counter,"act_end"] = act.date.max()
    sumtable.loc[counter,"act_nodays"] = (act.date.max()-act.date.min()).days
    sumtable.loc[counter,"act_cows"] = len(cows)
    sumtable.loc[counter,"act_lac"] = len(cowlac)
    
    # scc
    scc = pd.read_csv(os.path.join(path_data,"scc", "scc_preprocessed_" 
                                    + str(farm) + ".txt"), index_col=0)
    scc["date"] = pd.to_datetime(scc["measured_on"],format='%Y-%m-%d')
    scc["parity"] = scc["parity_dhi"]
    print("first meas = " + str(scc.date.min()))
    print("last meas = " + str(scc.date.max()))
    print("number of days data = " +str((scc.date.max()-scc.date.min()).days) )
    
    cows = scc.groupby(by=["animal_id"]).count()
    print("number of cow lactations = " + str(len(cows)))
    cowlac = scc.groupby(by=["animal_id","parity"]).count()
    print("number of cow lactations = " + str(len(cowlac)))
        
    # put in table
    sumtable.loc[counter,"scc_start"] = scc.date.min()
    sumtable.loc[counter,"scc_end"] = scc.date.max()
    sumtable.loc[counter,"scc_nodays"] = (scc.date.max()-scc.date.min()).days
    sumtable.loc[counter,"scc_cows"] = len(cows)
    sumtable.loc[counter,"scc_lac"] = len(cowlac)
        
    counter = counter + 1
    
    
    
# excel writer
writer = pd.ExcelWriter(os.path.join(path,"summary_data.xlsx"), engine = 'openpyxl')
sumtable.to_excel(writer,sheet_name = "data_overview", index=False)
# save and close
writer.save()
writer.close()
del writer