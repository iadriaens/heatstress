# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:26:29 2024

@author: u0084712
"""



farms = [41,42,43,44,46,47,48,49,50,51,54,55,57,58,59,61,62,64,65,66,67,68,69]

path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","data","new")

#%% load data

#milk
milk = pd.read_csv(os.path.join(path,"milk_" + str(farm)+".txt"),
                   usecols = [])
#act
act = pd.read_csv(os.path.join(path,"milk_" + str(farm)+".txt"),
                   usecols = [])
#wea
wea = pd.read_csv()

#%%
"""
TODO:
    - describe farms / data / periods
    - set up dataset with HS events
"""
    