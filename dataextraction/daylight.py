# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:29:44 2024

@author: u0084712
-------------------------------------------------------------------------------

get daylight data




"""

import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataextraction")
os.chdir(path)

import pandas as pd
import datetime
from suntime import Sun, SunTimeException

path_out = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","data")

#%%  hours of daylight

# determine altitude and longitude

lat = 51.475343137254896 # aws["latitude"].mean()   # when derived from all aws
lon = 4.883854901960785  # aws["longitude"].mean()
sun = Sun(lat, lon)

# create date list
base = datetime.datetime(2023,12,31)
date_list = [base - datetime.timedelta(days=x) for x in range(365*19+4)]
date_list = pd.DataFrame(date_list, columns = ["date"]).sort_values(by="date").reset_index(drop=1)
del base

# add datelight time
date_list["hrs_daylight"] = 0
T = 0
for d in date_list["date"]:
    up = sun.get_sunrise_time(d).hour*3600 + \
        sun.get_sunrise_time(d).minute*60
    down = sun.get_sunset_time(d).hour*3600 + \
        sun.get_sunset_time(d).minute*60
    date_list["hrs_daylight"].iloc[T] = (down-up)/3600 #(in hours)
    T=T+1
    
# date_list["hrs_daylight"].plot()

date_list.to_csv(os.path.join(path_out,"daylight.txt"))