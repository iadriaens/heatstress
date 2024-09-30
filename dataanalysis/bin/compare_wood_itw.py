# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:23:04 2024

@author: u0084712
"""

from scipy.signal import savgol_filter as savgol
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
import numpy as np
import pandas as pd
import os
from dmy_functions import qreg,itw
path = os.path.join("C:/", "Users", "u0084712", "OneDrive - KU Leuven",
                    "projects", "ugent", "heatstress", "dataanalysis")
os.chdir(path)

# %% file path

path_data = os.path.join("C:/", "Users", "u0084712", "OneDrive - KU Leuven",
                         "projects", "ugent", "heatstress", "datapreprocessing",
                         "results")

# farm selected
farms = [1, 2, 3, 4, 5, 6]

startdates = {1: 2011,
              2: 2014,
              3: 2014,
              4: 2017,
              5: 2016,
              6: 2014}
enddates = {1: 2019,
            2: 2020,
            3: 2017,
            4: 2022,
            5: 2019,
            6: 2019}

farm = 5
woodsettings = {"init" : [35,0.25,0.003],   # initial values
                "lb" : [0,0,0],             # lowerbound
                "ub" : [100,5,1],           # upperbound
                }
milk = pd.read_csv(os.path.join(path_data, "data", "milk_preprocessed_"
                                + str(farm) + ".txt"),
                   usecols=["animal_id", "parity", "date", "dim","dmy"])
milk["date"] = pd.to_datetime(milk["date"], format='%Y-%m-%d')


cowlac = milk[["animal_id","parity"]].drop_duplicates().reset_index(drop=1)[0:20]


# adjusted quantile regression model
milk["qr"] = 0
for i in range(0,len(cowlac)):
    
    df = milk.loc[(milk["animal_id"] == cowlac["animal_id"][i]) & \
                    (milk["parity"] == cowlac["parity"][i]) & \
                    (milk["dim"]<120),:]
    X = milk.loc[(milk["animal_id"] == cowlac["animal_id"][i]) & \
                    (milk["parity"] == cowlac["parity"][i]) & \
                    (milk["dim"]<120),"dim"]
    y = milk.loc[(milk["animal_id"] == cowlac["animal_id"][i]) & \
                    (milk["parity"] == cowlac["parity"][i]) & \
                    (milk["dim"]<120),"dmy"]    
    _,mod,_ = qreg(4,X,y,15,0.15,0.7,False)

    woodsettings["init"][0] = df.loc[(df["dim"] > 30) & (df["dim"] < 100),"dmy"].mean()/2        
    woodsettings["ub"][0] = df["dmy"].max()
    
    ax,p,ikbeneenkieken = itw(df["dim"],df["dmy"],
                   woodsettings["init"][0], woodsettings["init"][1], woodsettings["init"][2],
                   woodsettings["lb"], woodsettings["ub"], True)
    ax.plot(X,mod,"purple",lw=3)



# adjust woodsettings and fit wood model
woodsettings["init"][0] = df.loc[(df["dim"] > 30) & (df["dim"] < 100),"dmy"].mean()/2        
woodsettings["ub"][0] = df["dmy"].max()
p = curve_fit(wood, df["dim"], df["dmy"],
              p0 = woodsettings["init"],
              bounds=(woodsettings["lb"],woodsettings["ub"]),
              method='trf')
wa = p[0][0]
wb = p[0][1]
wc = p[0][2]

# check and set plotbool
if i in rsample_plot:
    print(i)
    plotbool = True
else:
    plotbool = False

# fit iterative wood model without plotting
