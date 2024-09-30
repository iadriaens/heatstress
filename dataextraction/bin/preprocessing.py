# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:26:20 2023

@author: adria036
"""
import os
os.chdir(r"C:\Users\adria036\OneDrive - Wageningen University & Research\iAdriaens_doc\Projects\iAdriaens\BEC3\scripts\bec3_dataprep")


#%% load packages

# import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib qt

#%% set filepaths

# path to data
path = os.path.join("C:","/Users","adria036","OneDrive - Wageningen University & Research",
                        "iAdriaens_doc","Projects","iAdriaens","BEC3","data","ltdata"
                        )

#fn = pd.DataFrame(os.listdir(path),columns = ["fn"])
farms = pd.DataFrame([30,31,33,34,35,38,39,40,43,44,45,46,47,48], columns = ["farm"])


#%% preprocess data per farm

for f in farms.farm:
    print("farm = " + str(f))
    # read data from farm
    dact = pd.read_csv(path+"//activity_"+str(f)+".txt", index_col = 0)
    dmilk = pd.read_csv(path+"//milk_"+str(f)+".txt", index_col = 0 )
    dani = pd.read_csv(path+"//ani_"+str(f)+".txt", index_col = 0)
    dlac = pd.read_csv(path+"//lac_"+str(f)+".txt", index_col = 0)
    dscc = pd.read_csv(path+"//scc_"+str(f)+".txt", index_col = 0)

    
    # set datetimes to datetimes
    dact["measured_on"] = pd.to_datetime(dact["measured_on"],format = "%Y-%m-%d %H:%M:%S")
    dmilk["started_at"] = pd.to_datetime(dmilk["started_at"],format = "%Y-%m-%d %H:%M:%S")
    dmilk["ended_at"] = pd.to_datetime(dmilk["ended_at"],format = "%Y-%m-%d %H:%M:%S")
    dani["birth_date"] = pd.to_datetime(dani["birth_date"],format = "%Y-%m-%d %H:%M:%S")
    dscc["measured_on"] = pd.to_datetime(dscc["measured_on"],format = "%Y-%m-%d %H:%M:%S")
    dlac["calving"] = pd.to_datetime(dlac["calving"], format = "%Y-%m-%d %H:%M:%S")
    
    # delete if no tmy data available
    dmilk = dmilk.loc[dmilk["tmy"].isna() == False,:]
    
    # sort milk data and calculate gaps
    dmilk = dmilk.sort_values(by = ["animal_id","started_at"]).reset_index(drop=1)
    dmilk["gap"] = np.nan
    dmilk["gap"].iloc[1:] = dmilk["started_at"][1:].values-dmilk["started_at"][:-1].values
    dmilk["gap"] = dmilk["gap"].astype(float)/(10**9*3600)
    dmilk.loc[dmilk["gap"]<0,"gap"] = np.nan
    
#------------------------------------------------------------------------------    
    #TODO: fix the .loc / copy warnings
    # get all moments where a new lactation starts (gap of 10 days)
    newlac = dmilk.loc[(dmilk.gap>24*10),["animal_id","lactation_id","parity","started_at","gap"]].sort_values(by = ["animal_id","started_at"]).reset_index(drop=1)
    newlac=newlac.rename(columns = {"started_at":"calving"})
    newlac["calving"] = newlac["calving"].dt.date
    new_no = dmilk["lactation_id"].max()+10000
    new_cows = dmilk["animal_id"].drop_duplicates().reset_index(drop=1)
    lacids = pd.DataFrame([])
    new = pd.DataFrame([])
    anew = pd.DataFrame([])
    #  correct newlac
    for cow in new_cows:
        print("cow = " + str(cow))
        
        # extract the lactations for which there's a large gap
        if len(newlac.loc[newlac.animal_id == cow,:])>0:
            sub = newlac.loc[newlac["animal_id"] == cow,:].reset_index(drop=1)
            sub["farm_id"] = f
            sub=sub.drop(columns = ["gap"])
            sub = sub[["lactation_id","farm_id","animal_id","parity","calving"]]
            sub.calving= pd.to_datetime(sub.calving)
            sub.calving = sub.calving.dt.date
        else: 
            sub = pd.DataFrame([])
        
        sub2 = dlac.loc[dlac["animal_id"]==cow,:]
        sub2["calving"] = sub2.loc[:,"calving"].dt.date
        sub2["parity"] = sub2["parity"].astype("int64")
        
        # add sub2 to sub
        sub = pd.concat([sub,sub2])
        
        # if the first lactation is wrongly assigned to "birthdate"= calving date with parity=0, not detected
        test = dmilk.loc[(dmilk["animal_id"]==cow) & (dmilk["parity"] == 0),["lactation_id","farm_id","animal_id","parity","started_at"]].reset_index(drop=1)
        if len(test) > 0:
            sub3 = test.iloc[[0],:]
            sub3.started_at = sub3.started_at.dt.date
            sub3  = sub3.rename(columns = {"started_at" : "calving"})
            sub = pd.concat([sub,sub3])
            del sub3
            
        # order, and delete if less than 14 days difference
        sub = sub.sort_values(by = "calving").reset_index(drop=1)
        sub["calving"] = pd.to_datetime(sub["calving"])
        sub["diff"] = np.nan
        
        
        # calvings that are included
        idx = sub.calving.drop_duplicates().index.values
        sub = sub.loc[idx,:].sort_values(by="calving").reset_index(drop=1)
        
        # find first duplicated parity and increase all later parities with 1 and adjust lactation ids
        idx = sub.loc[sub.parity.duplicated(keep=False)==True,"parity"].index.values
        if len(idx) > 0:
            idx = idx[0]
            if idx > 0:
                par = sub.parity[idx-1]
            else:
                par = -1
        
            sub["lactation_id"][idx:] = new_no + range(1,len(sub["lactation_id"][idx:])+1)
            sub["parity"][idx:] = range(par+1,par+len(sub["lactation_id"][idx:])+1)
        
            # add to now_no to create unique lactation ids
            new_no = new_no + len(sub["lactation_id"][idx:])+1
        
        # correct the milking data for this cow
        data = dmilk.loc[dmilk["animal_id"]==cow,:]
        act = dact.loc[dact["animal_id"]==cow,:]
        for i in range(0, len(sub)):
            print("parity = " + str(sub.parity[i]))
            # correct milk data
            data.loc[data.started_at > pd.to_datetime(sub.calving[i]),"parity"] = sub.parity[i]
            data.loc[data.started_at > pd.to_datetime(sub.calving[i]),"lactation_id"] = sub.lactation_id[i]
            dim = (data.loc[data.started_at > pd.to_datetime(sub.calving[i]),"started_at"] - pd.to_datetime(sub.calving[i]))
            data.loc[data.started_at > pd.to_datetime(sub.calving[i]),"dim"] = dim.astype("int64")/(10**9*24*3600)
            
            # correct activity data
            act.loc[act.measured_on > pd.to_datetime(sub.calving[i]),"parity"] = sub.parity[i]
            act.loc[act.measured_on > pd.to_datetime(sub.calving[i]),"lactation_id"] = sub.lactation_id[i]
            dim = (act.loc[act.measured_on > pd.to_datetime(sub.calving[i]),"measured_on"] - pd.to_datetime(sub.calving[i]))
            act["dim"] = np.nan
            act.loc[act.measured_on > pd.to_datetime(sub.calving[i]),"dim"] = dim.astype("int64")/(10**9*24*3600)
            
        # add the new sub to the lactation_id array
        lacids = pd.concat([lacids,sub])
        new = pd.concat([new,data])
        anew = pd.concat([anew,act])
    
    # tidy up new dataframes
    lacids = lacids.reset_index(drop=1)
    new = new.sort_values(by=["animal_id","started_at"]).reset_index(drop=1)
    anew = anew.sort_values(by=["animal_id","measured_on"]).reset_index(drop=1)
    
    # tidy up workspace and memory
    dact = anew.copy()
    dmilk = new.copy()
    dlac = lacids.copy()
    del sub, sub2, i, dim, new, anew, lacids, act, data, idx, cow    
    del test, par, new_cows, new_no, newlac 

#------------------------------------------------------------------------------
    # activity: combine activity per day (sum)
    diff = dact["measured_on"]-pd.to_datetime(dact["measured_on"].dt.date.min())
    diff = np.floor(diff.astype("int64")/(10**9*24*3600))
    dact["day"] = diff.astype(int)
    act = dact[["animal_id","activity_total","rumination_acc","rumination_time","day"]].groupby(by=["animal_id","day"]).sum()
    act = act.reset_index()
    idx = dact[["farm_id","animal_id","lactation_id","day"]].drop_duplicates().index.values
    new = dact.iloc[idx,:]
    new = new[["farm_id","animal_id","lactation_id","parity","day","measured_on"]]
    new2 = new.merge(act, how = "outer", on = ["animal_id","day"])
    # remove the first measurement of a new lactation (= duplicated)
    new2 = new2.loc[new2[["animal_id","day"]].duplicated()==False,:].reset_index(drop=1)
    
#------------------------------------------------------------------------------
    # select lactations for which data from DIM < 5 and > 75 are available
    subset = dmilk[["animal_id","lactation_id","dim","started_at"]].groupby(by = ["animal_id","lactation_id"]).min().reset_index()
    subset2 = dmilk[["animal_id","lactation_id","dim","started_at"]].groupby(by = ["animal_id","lactation_id"]).max().reset_index()    
    subset["enddim"] = subset2["dim"]
    subset["enddate"] = subset2["started_at"]
    subset = subset.rename(columns = {"dim" : "startdim","startdate":"started_at"})
    subset = subset.sort_values(by = "startdim")
    subset = subset.loc[(subset["startdim"]<=5) & (subset["enddim"]>75),:].reset_index(drop=1)
    
    # select data from animals in subset
    milk = dmilk.merge(subset[["animal_id","lactation_id"]],
                       how = "inner",on = ["animal_id","lactation_id"])
    act = new2.merge(subset[["animal_id","lactation_id"]],
                       how = "inner",on = ["animal_id","lactation_id"]) 
    scc = dscc.merge(subset[["animal_id","lactation_id"]],
                       how = "inner",on = ["animal_id","lactation_id"]) 
    
    # select appropriate weather information
    dweather = pd.read_csv(path+"//weather_information.txt", index_col = 0)
    dweather["datetime"] = pd.to_datetime(dweather["datetime"], format = "%Y-%m-%d %H:%M:%S")
    dfarms = pd.read_csv(path+"//farm_information.txt", index_col = 0)
    startdate = milk["started_at"].min()
    enddate = milk["started_at"].max()
    aws = dfarms.loc[dfarms["farm_id"] == f,"aws_id"].values
    wea = dweather.loc[(dweather["aws_id"] == aws[0]) & (dweather["datetime"] > pd.to_datetime(startdate)) & (dweather["datetime"] < pd.to_datetime(enddate)),: ]

#------------------------------------------------------------------------------
    # write to csv
    milk.to_csv(path+"//farm_" + str(f) + "_milk" + ".txt")
    act.to_csv(path+"//farm_" + str(f) + "_act" + ".txt")
    wea.to_csv(path+"//farm_" + str(f) + "_wea" + ".txt")
    scc.to_csv(path+"//farm_" + str(f) + "_scc" + ".txt")
    
    


#---------------------------------- visualisations-----------------------------
    
    # fig, ax = plt.subplots(nrows=1,ncols=1, figsize= (15,8))
    # cow = 290  #200, 179, etc
    # dset = dmilk.loc[dmilk.animal_id == cow,["animal_id","lactation_id","started_at","dim","tmy","mi","parity","gap"]]
    # dset["relmy"] = dset["tmy"]/dset["mi"]*3600
    # sns.relplot(data = dset, x="dim",y="relmy", hue = "parity", palette = sns.color_palette("tab10"))
    # sns.relplot(data = dset, x="started_at",y="relmy", hue = "parity", palette = sns.color_palette("tab10"))
    # ax.set_ylim([0,4])
    
    # test2 = new.loc[(new["animal_id"]==19)&(new["parity"]==0) & (~new["tmy"].isna()) ,:]
    # cow == 19
    # fig, ax = plt.subplots(nrows=1,ncols=1, figsize= (15,8))
    # ax.plot(test.loc[test.animal_id == cow,"dim"],test.loc[test.animal_id == cow,"tmy"] / \
    #         test.loc[test.animal_id == cow,"mi"]*3600,"o")
    # ax.set_ylim([0,4])