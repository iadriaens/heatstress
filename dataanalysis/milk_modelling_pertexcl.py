# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:09:50 2024

@author: u0084712


-------------------------------------------------------------------------------


quick modelling of heatstress with removal of all perturbations that are not 
linked to warm weather





"""
import os

path = os.path.join("C:/", "Users", "u0084712", "OneDrive - KU Leuven",
                    "projects", "ugent", "heatstress", "dataanalysis")
os.chdir(path)

#%% load packages

# from scipy.signal import savgol_filter as savgol
# import matplotlib
import matplotlib.pyplot as plt
# import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from dmy_functions import wood, itw, pert
from scipy.optimize import curve_fit
import seaborn as sns
import numpy as np
import pandas as pd
from random import sample
from itertools import combinations
from itertools import product

# import scipy.stats as stats
# import openpyxl
import warnings


path_data = os.path.join("C:/", "Users", "u0084712", "OneDrive - KU Leuven",
                         "projects", "ugent", "heatstress", "datapreprocessing",
                         "results")
dpath = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","dataanalysis",
                    "data")
# %matplotlib qt

warnings.filterwarnings("ignore", category = UserWarning)

#%% select farms, set constants and load data

# farm selected
farms = [1, 2, 3, 4, 5, 6]

# define wood settings
woodsettings = {"init" : [35,0.25,0.003],   # initial values
                "lb" : [0,0,0],             # lowerbound
                "ub" : [100,5,1],           # upperbound
                }


#%% preprocessing 

for farm in farms:
    # read milk yield data
    data = pd.read_csv(os.path.join(path_data, "data", "milk_preprocessed_"
                                    + str(farm) + ".txt"),
                       usecols=["farm_id", "animal_id", "parity", "date", "dim", "dmy"])
    data["date"] = pd.to_datetime(data["date"], format='%Y-%m-%d')
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data = data.loc[data["dim"] >=0].reset_index(drop=1)
    
    # prepare pert_no
    data["pert_no"] = 0
    
    # unique lactations
    cowlac = data[["animal_id","parity"]].drop_duplicates().reset_index(drop=1)
    
    # select cows to plot (rsample_plot = [0,1,2,3,4,5])
    rsample_plot = sample(cowlac.index.values.tolist(),3)
    rsample_plot.append(60)
    
    # visualise + model per lactation
    for i in range(0,len(cowlac)):
        # select data
        df = data.loc[(data["animal_id"] == cowlac["animal_id"][i]) & \
                      (data["parity"] == cowlac["parity"][i]),:].copy()
        idx = df.index.values
        
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
        ax,p,mod = itw(df["dim"],df["dmy"], 
                       woodsettings["init"][0], woodsettings["init"][1], woodsettings["init"][2],
                       woodsettings["lb"], woodsettings["ub"], False)
        
        # add mod to df and to milk
        mod = mod.rename("mod")
        data.loc[idx,"mod"] = mod.values
        df.loc[:,"mod"] = mod.values
        df["res"] = df["dmy"]-df["mod"] # residuals
        
        # plot
        if plotbool == True:
            fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize=(20,9))
            sns.lineplot(data = df,x = "dim",y = "dmy",
                    color ="blue",marker = "o", ms = 4, lw = 1.2, label = "dmy")
            ax.plot(df["dim"],df["mod"],color = "purple",lw = 2,label = "itw")
        
        # detect perturbations and add pert to df
        pt = pert(df["dim"],df["dmy"],df["mod"])
        pt["pert"] = pt.iloc[:,4:8].sum(axis=1)  # locations all perturbations
        df["pert"] = pt["pert"].values
        df["pert_no"] = pt["pert_no"].values
        
        
        # estimate noise from non-perturbed to add to mod for replacement
        # this is often too large, so halve it
        df["eps"] = np.random.default_rng().normal(0,
                                             df.loc[df["pert"]==0,"res"].std()*0.5,
                                             len(df))
        df.loc[df["pert"]==0,"eps"] = np.nan
        
        # plot perturbations
        if plotbool == True:
            ax.plot(df.loc[df["pert"]==1,"dim"], df.loc[df["pert"]==1,"dmy"],
                     lw = 0, marker = "x", ms = 4, color = "red")
            ax.plot(df["dim"],df["mod"]+df["eps"],
                    lw = 1, marker = "o", ms = 5, color = "teal",
                    markeredgecolor="w")
            ax.legend()
        
        # during perturbations, data = simulated data
        df["dmy2"] = df["dmy"]
        df.loc[df["pert"]==1,"dmy2"] = df.loc[df["pert"]==1,"mod"] + df.loc[df["pert"]==1,"eps"] 
        
        # estimate wood model using simulated data during perturbations
        woodsettings["init"][0] = df.loc[(df["dim"] > 30) & (df["dim"] < 100),"dmy2"].mean()/2        
        woodsettings["ub"][0] = df["dmy2"].max()
        p = curve_fit(wood, df["dim"], df["dmy2"],
                      p0 = woodsettings["init"],
                      bounds=(woodsettings["lb"],woodsettings["ub"]),
                      method='trf')
        wa = p[0][0]
        wb = p[0][1]
        wc = p[0][2]

        df["mod2"] = wood(df["dim"],wa,wb,wc)
        
        # plot
        if plotbool == True:
            ax.plot(df["dim"],df["mod2"],color="tomato",lw=2,ls="--")
            ax.set_title("farm " + str(farm) + ", cow " + \
                         str(cowlac["animal_id"][i]) + ", lac " + \
                         str(cowlac["parity"][i]))
            plt.savefig(os.path.join(path,"results","milk","dmy_replaced_farm_" + \
                                         str(farm) + "_cow" + \
                                         str(cowlac["animal_id"][i]) + "_lac" + \
                                         str(cowlac["parity"][i]) + ".tif"))
            plt.close()
            
        # add to data    
        data.loc[idx,"pert"] = df["pert"].values
        data.loc[idx,"pert_no"] = df["pert_no"].values + max(data["pert_no"])
        data.loc[idx,"eps"] = df["eps"].values
        data.loc[idx,"mod2"] = df["mod2"].values
        
    del fig, i, idx, mod, p, ax, plotbool, pt, rsample_plot,wa,wb,wc 
        
    # read weather data; if warm day > thi_max > 68
    thi = pd.read_csv(os.path.join(dpath,"weatherfeatures_" + str(farm) + ".txt"),
                      usecols = ["date","thi_max"])
    thi["date"] = pd.to_datetime(thi["date"],format='%Y-%m-%d')
    thi["hot"] = (thi["thi_max"]>=68).astype(int)
    
    # combine thi and data
    data = pd.merge(data,thi[["date","hot"]],how="inner",on="date")
    data = data.sort_values(by = ["animal_id","date"]).reset_index(drop=1)
    
    # all combinations of pert_no, pert and hot
    test = data.loc[data["pert"]==1,["pert_no","hot"]].drop_duplicates()
    test = test.loc[test["hot"]==1,:].rename(columns = {"hot":"keep_pert"})
    
    # merge back into data
    data = pd.merge(data,test,how="outer",on="pert_no")
    data["keep_pert"] = data["keep_pert"].fillna(0)
    
    # when there is a perturbation not related to heat stress, replace by mod+eps
    data["dmy2"] = data["dmy"]
    data.loc[(data["pert"]==1) & (data["keep_pert"]==0),"dmy2"] = \
        data.loc[(data["pert"]==1) & (data["keep_pert"]==0),"mod"] + \
        data.loc[(data["pert"]==1) & (data["keep_pert"]==0),"eps"]
        
    # calculate average parity lactation curve
    data["pargroup"] = (
        (pd.concat([data["parity"], pd.DataFrame(
            3*np.ones((len(data), 1)))], axis=1))
        .min(axis=1)
        )
    parlac = (
        data[["dim","dmy2","pargroup"]]
        .groupby(by = ["pargroup","dim"]).mean()
        ).reset_index().rename(columns={"dmy2" : "paravg"})
    
    # correct dmy for parity-average
    data = pd.merge(data,parlac,how="left",on = ["pargroup","dim"])
    data["dmy2_corr"] = data["dmy2"] - data["paravg"]

    # add ID for modelling
    cowlac = data[["animal_id", "parity"]].drop_duplicates().reset_index(drop=1).reset_index()
    cowlac = cowlac.rename(columns={"index": "ID"})
    data = data.merge(cowlac, how="inner", on=["animal_id", "parity"])
    
    # drop columns to obtain final dataset
    data = data[["farm_id","animal_id","parity","ID","date","dim","dmy","dmy2","pargroup","dmy2_corr"]]
    data.to_csv(os.path.join(path,"results","data","milk_corrected_perturbations_farm_" + \
                             str(farm)+ ".txt"))
   
        
#%% modelling with weather data

for farm in farms:
    # load data
    data = pd.read_csv(os.path.join(path,"results","data","milk_corrected_perturbations_farm_" + \
                             str(farm)+ ".txt"), index_col = 0)
    data["date"] = pd.to_datetime(data["date"], format='%Y-%m-%d')
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["season"] = "winter"
    data.loc[(data["month"] >= 3) & (data["month"] < 6), "season"] = "spring"
    data.loc[(data["month"] >= 6) & (data["month"] < 9), "season"] = "summer"
    data.loc[(data["month"] >= 9) & (data["month"] < 12), "season"] = "autumn"
 
    # add ls class
    data.loc[(data["dim"]) < 22, "ls"] = "0-21"
    data.loc[(data["dim"] >= 22) &
             (data["dim"] < 61), "ls"] = "22-60"
    data.loc[(data["dim"] >= 61) &
             (data["dim"] < 121), "ls"] = "61-120"
    data.loc[(data["dim"] >= 121) &
             (data["dim"] < 201), "ls"] = "121-200"
    data.loc[(data["dim"] >= 201), "ls"] = ">200"
    data.ls = pd.Categorical(data.ls,
                             categories=["0-21", "22-60",
                                         "61-120", "121-200", ">200"],
                             ordered=True)
    
 
    # load weather
    wea = pd.read_csv(os.path.join(path_data, "data", "weather_prep_farm_" + str(farm) + ".txt"),
                      index_col = 0)
    wea["date"] = pd.to_datetime(wea["date"], format='%Y-%m-%d')

    # define extra features % (in hours) HS present day -1 > day -3
    wea["HS_3_1"] = np.nan
    wea.loc[3:,"HS_3_1"] = (wea["HS_tot"].iloc[0:-3].values + \
                            wea["HS_tot"].iloc[1:-2].values + \
                            wea["HS_tot"].iloc[2:-1].values) / (3*24)*100
   
    # define extra features % (in hours) recovery present day -1 > day -3
    wea["REC_3_1"] = np.nan
    wea.loc[3:,"REC_3_1"] = (wea["HS0"].iloc[0:-3].values + \
                            wea["HS0"].iloc[1:-2].values + \
                            wea["HS0"].iloc[2:-1].values) / (3*24)*100
    
    # define extra features % (in hours) recovery yesterday
    wea["REC_1"] = np.nan
    wea.loc[1:,"REC_1"] = (wea["HS0"].iloc[0:-1].values)/(24)*100
    
    # combine data and wea on date
    data = pd.merge(data,wea,on=["date"],how="inner").sort_values(by=["ID","date"]).reset_index(drop=1)
    
    # possible model variables    
    Y = ["dmy2_corr"]
    Z = ["C(year)","C(season)","C(ls)","C(pargroup)","C(year):C(season)","C(ls):C(pargroup)"]
    X = ["THImax","THIavg","HS0", "HS_tot", "HS_3_1", "REC_3_1", "REC_1", "HS_3_1:REC_1" ]
        
    # make valid combinations of the covariates
    eq={}
    for i in range(1,len(Z)+1):
        comb = list(combinations(Z,i))
        print(comb)
        eq[i] = comb
        
    covars = pd.DataFrame([],columns=["eq"])
    for i in range(1,len(Z)+1):
        for el in range(0,len(eq[i])):
            # print(eq[i][el])
            equation = Y[0] + " ~ " + eq[i][el][0]
            for var in range(1,len(eq[i][el])):
                # print(eq[i][el][var])
                equation = equation + " + " + eq[i][el][var]
                
            
            if (("C(year):C(season)" not in equation) and ("C(ls):C(pargroup)" not in equation)):
                covars = pd.concat([covars,pd.DataFrame([equation],columns = ["eq"])])
                print(equation)
            elif (("C(year):C(season)" in equation) and ("C(year) +" in equation) and (" + C(season)" in equation)) and \
                 ("C(ls):C(pargroup)" not in equation):
                 covars = pd.concat([covars,pd.DataFrame([equation],columns = ["eq"])])
                 print(equation)
            elif (("C(ls):C(pargroup)" in equation) and ("C(ls) +" in equation) and (" + C(pargroup)" in equation)) and \
                 ("C(year):C(season)" not in equation):
                 covars = pd.concat([covars,pd.DataFrame([equation],columns = ["eq"])])
                 print(equation)
            elif (("C(ls):C(pargroup)" in equation) and ("C(ls) +" in equation) and (" + C(pargroup)" in equation)) and \
                 (("C(year):C(season)" in equation) and ("C(year) +" in equation) and (" + C(season)" in equation)):
                 covars = pd.concat([covars,pd.DataFrame([equation],columns = ["eq"])])
                 print(equation)
             
    # make valid combinations of the predictors
    for i in range(1,len(X)+1):
        comb = list(combinations(X,i))
        print(comb)
        eq[i] = comb
        
    predics = pd.DataFrame([],columns=["eq"])
    for i in range(1,len(X)+1):
        for el in range(0,len(eq[i])):
            # print(eq[i][el])
            equation = " + " + eq[i][el][0]
            for var in range(1,len(eq[i][el])):
                # print(eq[i][el][var])
                equation = equation + " + " + eq[i][el][var]
                
            if (("HS_3_1:REC_1" not in equation) and ("HS_3_1:REC_3_1" not in equation)):
                predics = pd.concat([predics,pd.DataFrame([equation],columns = ["eq"])])
                print(equation)
            elif (("HS_3_1:REC_1" in equation) and ("HS_3_1 +" in equation) and (" + REC_1" in equation)) and \
                 ("HS_3_1:REC_3_1" not in equation):
                 predics = pd.concat([predics,pd.DataFrame([equation],columns = ["eq"])])
                 print(equation)
            elif (("HS_3_1:REC_3_1" in equation) and ("HS_3_1 +" in equation) and (" + REC_3_1" in equation)) and \
                 ("HS_3_1:REC_1" not in equation):
                 predics = pd.concat([predics,pd.DataFrame([equation],columns = ["eq"])])
                 print(equation)
            elif (("HS_3_1:REC_3_1" in equation) and ("HS_3_1 +" in equation) and (" + REC_3_1" in equation)) and \
                 (("HS_3_1:REC_1" in equation) and ("HS_3_1 +" in equation) and (" + REC_1" in equation)):
                 predics = pd.concat([predics,pd.DataFrame([equation],columns = ["eq"])])
                 print(equation)
            
            
    # combine lists covars and predics
    models = pd.DataFrame(list(product(covars["eq"],predics["eq"])),columns = ["cov","pred"])
    models["eq"] = models["cov"] + models["pred"]  
    models = models.reset_index(drop=1)                 
        
    # modelling 
    models["is_converged"] = False
    models["aic"] = np.nan
    data2 = data.copy()
    data2["dmy2_corr"] = (data2["dmy2_corr"] - data2["dmy2_corr"].mean()) / data2["dmy2_corr"].std()
    data2.iloc[:,14:] = (data2.iloc[:,14:]-data2.iloc[:,14:].min())/(data2.iloc[:,14:].max()-data2.iloc[:,14:].min())
    
    for i in range(11500,len(models),1):
        # print(i)
        if ((models.loc[i]["aic"]) > 0) == False :
            print(models["aic"][i])
            if "THIavg" in models["eq"][i]:
                re_formula = "~THIavg"
            elif "THImax" in models["eq"][i]:
                re_formula = "~THImax"
            elif "HS_tot" in models["eq"][i]:
                re_formula = "~HS_tot"
            else:
                re_formula = "~1"
    
            md = smf.mixedlm(models["eq"][i],
                         data=data2,
                         groups=data2["ID"],
                         re_formula=re_formula)
            mdf = md.fit(method=["lbfgs"],reml=False)
            
            models.loc[i,"is_converged"] = mdf.converged
            models.loc[i,"aic"] = mdf.aic
    


# selected models
sel = models.loc[models["aic"] < models["aic"].min()+3,:].reset_index(drop=1) 


modeq = sel.loc[6,"eq"]

md = smf.mixedlm(modeq,
             data=data2,
             groups=data2["ID"],
             re_formula=re_formula)
mdf = md.fit(method=["lbfgs"],reml=False)
print(mdf.summary())



md = smf.mixedlm("dmy2_corr ~ THIavg + HS_3_1 + C(ls) + THIavg*C(ls) + C(year) + C(season) + C(year)*C(season)",
                     data=data2,
                     groups=data2["ID"],
                     re_formula="~THIavg")

mdf = md.fit(method=["lbfgs"])

print(mdf.summary())

var_fe = np.var(np.matmul(md.exog,mdf.fe_params))

# R²
R = {"SSTO": (np.sqrt((data["dmy2_corr"] - data["dmy2_corr"].mean())**2)).sum(),
      "SSE": (np.sqrt(mdf.resid**2)).sum()
      }
R["R2"] = round(1 - (R["SSE"] / R["SSTO"]), 3)
R["fixed"] = var_fe / (var_fe + mdf.cov_re.Group.Group +
                      mdf.cov_re.THIavg.THIavg + mdf.scale)
# conditional R² = proportion variance explained by fixed + random effects
# formula 2.5 of https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0213
R["conditional"] = (var_fe + mdf.cov_re.Group.Group + mdf.cov_re.THIavg.THIavg) / \
    (var_fe + mdf.cov_re.Group.Group + mdf.cov_re.THIavg.THIavg + mdf.scale)




