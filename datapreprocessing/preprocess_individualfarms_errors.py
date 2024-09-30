# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 09:56:35 2023

@author: u0084712
"""


#------------------------------------ VAN SCHAIJK: data stops > rumination only

# these files contain the actual data: for Van Schaijk: 
data = pd.read_csv(r"C:/Users/u0084712/Downloads/l_vanschaijk_20181019_PrmActivityScr.txt", 
                  sep = ";",header = None)
print(data[-200:])


# this is only the timing of the activity measures
data2 = pd.read_csv(r"C:/Users/u0084712/Downloads/l_vanschaijk_20181019_LimActivityHours.txt", 
                  sep = ";",header = None)
print(data2[-200:])


#-------------------------------------- GUNS: data shifts values, not fewer data points

data3 = pd.read_csv(r"C:/Users/u0084712/Downloads/l_guns_20210926_PrmActivityScr.txt", 
                  sep = ";",header = None)
data3.columns = ["AscId","AscAniId","AscCellTime","AscActivity","AscAttentionValue","AscHeatSickChange100","AscTotalRuminationTime","AscActivityChange",
                "AscRuminationMinutes","AscRuminationMark","AscRuminationChange100","AscCalculated","AscRuminationMinutesAvg","AscRuminationMinutesSd",
                "AscRuminationV","AscRuminationW","AscRuminationX","AscRuminationSumEightHours",
                "AscRuminationSumEightHoursAvg","AscRuminationY","AscRuminationZ","AscCalvingDistressAlertProbability","AscHealthIndex100"]
print(data3[["AscId","AscAniId","AscCellTime","AscActivity"]][-200:])

subset = data3[["AscId","AscAniId","AscCellTime","AscActivity"]].dropna()
animals = subset[["AscAniId","AscCellTime","AscActivity"]].groupby(by="AscAniId").count()
subset2 = subset.loc[subset["AscAniId"] == 255,:]

#%matplotlib qt
subset2["AscActivity"].plot()
subset2["rolact"] = subset2["AscActivity"].rolling(100).median()
subset2["rolact"].plot(color = 'red',linewidth = 1.5)


#------------------------------------- HUZEN f38: AWS (18) data incomplete
