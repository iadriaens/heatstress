# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:15:38 2023

@author: u0084712


--- analysis paper Gote et al. 2023

STEP1: load preprocessed data
STEP2: modelling - act and milk yield
STEP3: figures and tables - generate results

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
path = os.path.join("C:/", "Users", "u0084712", "OneDrive - KU Leuven",
                    "projects", "ugent", "heatstress", "dataanalysis")
os.chdir(path)

# %% import packages


# %% file path

path_data = os.path.join("C:/", "Users", "u0084712", "OneDrive - KU Leuven",
                         "projects", "ugent", "heatstress", "datapreprocessing",
                         "results")

# farm selected
farms = [1, 2, 3, 4, 5, 6]

# %matplotlib qt

# dates selected
"""
1 - 2011:2019
2 - 2014:2020
3 - 2014:2017
4 - 2017:2022
5 - 2016:2019
6 - 2013:2020
"""
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

#%%###########################################################################
################################   ACTIVITY   ################################
##############################################################################

for farm in farms:
    # activity
    act = pd.read_csv(os.path.join(path_data, "data", "act_preprocessed_"
                                   + str(farm) + ".txt"),
                      usecols=['farm_id', 'animal_id', 'parity', 'date', 'act_corr'])
    act["date"] = pd.to_datetime(act["date"], format='%Y-%m-%d')
    act = act.rename(columns={'act_corr': 'activity'})

    # select dates
    act = act.loc[(act["date"].dt.year >= startdates[farm]) &
                  (act["date"].dt.year <= enddates[farm]), :].reset_index(drop=1)

    # read milk yield data to add dim / parity information
    milk = pd.read_csv(os.path.join(path_data, "data", "milk_preprocessed_"
                                    + str(farm) + ".txt"),
                       usecols=["animal_id", "parity", "date", "dim"])
    milk["date"] = pd.to_datetime(milk["date"], format='%Y-%m-%d')

    # merge act and milk, delete milk
    act = pd.merge(act, milk, how="inner", on=["animal_id", "parity", "date"])
    act = act.sort_values(by=["animal_id", "date"]).reset_index(drop=1)
    del milk

    # -------------------------- visualisation per dim/parity ------------------
    act["pargroup"] = (
        (pd.concat([act["parity"], pd.DataFrame(
            3*np.ones((len(act), 1)))], axis=1))
        .min(axis=1)
    )
    sumstat = (
        (act[["pargroup", "dim", "activity"]].groupby(by=["pargroup", "dim"])
         .agg({"activity": ["count", "mean", "std"]})).reset_index()
    )
    sumstat.columns = sumstat.columns.droplevel()
    sumstat.columns = ["pargroup", "dim", "count", "mean", "std"]

    fig, ax = plt.subplots(3, 1, figsize=(16, 11))
    for parity in sumstat["pargroup"].drop_duplicates().astype(int):
        print(parity)
        subset = sumstat.loc[sumstat["pargroup"] == parity, :]
        ax[parity-1].set_ylim(0, 1300)
        ax[parity-1].fill_between(subset["dim"],
                                  subset["mean"]-2*subset["std"],
                                  subset["mean"]+2*subset["std"],
                                  color="palevioletred", alpha=0.5)
        ax[parity-1].plot(subset["dim"],
                          subset["mean"],
                          linewidth=2, color="crimson")
        ax[parity-1].set_xlim(0, subset["dim"].max())
        ax[parity-1].set_ylim(-(subset["mean"]+2.2*subset["std"]).max(),
                              (subset["mean"]+2.2*subset["std"]).max())
        if parity == 1:
            ax[parity-1].set_title("farm  " + str(farm) + ", standardised activity, parity = " +
                                   str(round(parity)), fontsize=14)
        else:
            ax[parity-1].set_title("parity = " + str(parity))
        if parity == 3:
            ax[parity-1].set_xlabel("dim [d]")
        ax[parity -
            1].set_ylabel("daily activity, mean+2*std, [steps]", color="red")
        ax[parity-1].plot([sumstat["dim"].min(), sumstat["dim"].max()],
                          [sumstat["mean"].mean(), sumstat["mean"].mean()],
                          color="black", linestyle="--", lw=1.5)
        ax[parity-1].set_ylim(-12, 12)
        #TODO: legend and grid
        ax2 = ax[parity-1].twinx()
        ax2.plot(subset["dim"],
                 subset["count"],
                 linewidth=2, color="blue")
        ax2.grid(False)
        ax2.set_ylabel("number of animals", color="blue")
    plt.savefig(os.path.join(path, "results", "activity",
                "scorr_activity_stats_dim_" + str(farm) + ".tif"))
    plt.close()

    # --------------------------- individual curves ----------------------------
    fig, ax = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    act = act.sort_values(
        by=["animal_id", "parity", "dim"]).reset_index(drop=1)

    # first parity
    cowlac = act.loc[act["pargroup"] == 1, ["animal_id",
                                            "parity"]].drop_duplicates().reset_index(drop=1)
    par1col = sns.color_palette(palette="flare",
                                n_colors=len(cowlac))
    for i in range(0, len(cowlac)):
        print(cowlac["animal_id"][i], cowlac["parity"][i])
        df = act.loc[(act["animal_id"] == cowlac["animal_id"][i]) &
                     (act["parity"] == cowlac["parity"][i]), ["dim", "activity"]]
        ax[0].plot(df["dim"], df["activity"], axes=ax[0], color=par1col[i],
                   linewidth=0.4)
    ax[0].set_xlim([0, 400])
    ax[0].set_title("farm = " + str(farm) +
                    ", individual activity curves parity 1")
    ax[0].set_ylabel("activity [unit]")

    # parity 2
    cowlac = act.loc[act["pargroup"] == 2, ["animal_id",
                                            "parity"]].drop_duplicates().reset_index(drop=1)
    par2col = sns.color_palette("ch:s=.25,rot=-.25",
                                n_colors=len(cowlac))
    for i in range(0, len(cowlac)):
        print(cowlac["animal_id"][i], cowlac["parity"][i])
        df = act.loc[(act["animal_id"] == cowlac["animal_id"][i]) &
                     (act["parity"] == cowlac["parity"][i]), ["dim", "activity"]]
        ax[1].plot(df["dim"], df["activity"], axes=ax[1], color=par2col[i],
                   linewidth=0.4)
    ax[1].set_xlim([0, 400])
    ax[1].set_title("individual activity curves parity 2")
    ax[1].set_ylabel("activity [unit]")

    # parity 3
    cowlac = act.loc[act["pargroup"] == 3, ["animal_id",
                                            "parity"]].drop_duplicates().reset_index(drop=1)
    par3col = sns.color_palette("light:#5A9",
                                n_colors=len(cowlac))
    for i in range(0, len(cowlac)):
        print(cowlac["animal_id"][i], cowlac["parity"][i])
        df = act.loc[(act["animal_id"] == cowlac["animal_id"][i]) &
                     (act["parity"] == cowlac["parity"][i]), ["dim", "activity"]]
        ax[2].plot(df["dim"], df["activity"], axes=ax[2], color=par3col[i],
                   linewidth=0.4)
    ax[2].set_xlim([0, 400])
    ax[2].set_title("individual activity curves parity 3+")
    ax[2].set_xlabel("dim [d]")
    ax[2].set_ylabel("activity [unit]")
    plt.savefig(os.path.join(path, "results", "activity",
                "scorr_activity_individual_dim_" + str(farm) + ".tif"))
    plt.close()

# %% ACTIVITY: indiviual curves - explore
for farm in farms:
    # activity
    act = pd.read_csv(os.path.join(path_data, "data", "act_preprocessed_"
                                   + str(farm) + ".txt"),
                      usecols=['farm_id', 'animal_id', 'parity', 'date', 'act_corr'])
    act["date"] = pd.to_datetime(act["date"], format='%Y-%m-%d')

    # select dates
    act = act.loc[(act["date"].dt.year >= startdates[farm]) &
                  (act["date"].dt.year <= enddates[farm]), :].reset_index(drop=1)

    act = act.rename(columns={'act_corr': 'activity'})
    act["pargroup"] = (
        (pd.concat([act["parity"], pd.DataFrame(
            3*np.ones((len(act), 1)))], axis=1))
        .min(axis=1)
    )

    # read milk yield data to add dim / parity information
    milk = pd.read_csv(os.path.join(path_data, "data", "milk_preprocessed_"
                                    + str(farm) + ".txt"),
                       usecols=["animal_id", "parity", "date", "dim", "dmy"])
    milk["date"] = pd.to_datetime(milk["date"], format='%Y-%m-%d')

    # merge act and milk and sort values
    act = pd.merge(act, milk, how="inner", on=["animal_id", "parity", "date"])
    act = act.sort_values(by=["animal_id", "date"]).reset_index(drop=1)

    # add pargroup to milk
    milk["pargroup"] = (
        (pd.concat([milk["parity"], pd.DataFrame(
            3*np.ones((len(act), 1)))], axis=1))
        .min(axis=1)
    )

    # weather information
    wea = pd.read_csv(os.path.join(path_data, "data", "weather_farm_"
                                   + str(farm) + ".txt"),
                      usecols=['date', 'temp', 'thi'])
    wea["date"] = pd.to_datetime(wea["date"], format='%Y-%m-%d')
    wea = wea.loc[(wea["date"] >= act["date"].min()) &
                  (wea["date"] <= act["date"].max()), :]

    # select individual curves for plotting + quantify peaks
    cowlac = act[["animal_id", "parity"]].drop_duplicates().reset_index(drop=1)
    randanimals = cowlac.sample(
        12)[["animal_id", "parity"]].index.values  # random plots
    sns.set_style("whitegrid")
    # paper= farm5 cow=7528 par = 5 i=27
    for i in range(0, len(cowlac)):
        # plot if in randomly selected
        if i in randanimals:
            print(i)
            df = act.loc[(act["animal_id"] == cowlac["animal_id"][i]) &
                         (act["parity"] == cowlac["parity"][i]), ["dim", "date", "activity", "pargroup"]]
            df = df.sort_values(by="date").reset_index(drop=1)

            # visualise trend
            df["act_sm"] = savgol(df["activity"], 7, 1)

            # milk
            dfm = milk.loc[(milk["animal_id"] == cowlac["animal_id"][i]) &
                           (milk["parity"] >= cowlac["parity"][i]), :]

            # prepare figure
            plt.rcParams.update({'font.size': 12})
            fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
            ax[0].grid(True)
            # plot THI
            df3 = wea.loc[(wea["date"] >= df["date"].min()) &
                          (wea["date"] <= df["date"].max()), :]
            ax3 = ax[0].twinx()
            ax3.grid(False)
            ax3.plot(df3["date"], df3["thi"], linestyle="-", linewidth=1.5,
                     color="crimson")
            
            ax3.set_ylim(-20, df3["thi"].max()+5)
            ax3.fill_between(df3["date"], 68, df3["thi"].max()+5, color="crimson",
                             alpha=0.2)
            ax3.set_ylabel("THI")
            
            # # vertical lines when THI is high
            # new = df3.loc[df3["thi"]>68,:]
            # new["gap"] = ((new["date"].diff()).dt.days)
            # new["gap"].iloc[0] = 20
            # new2 = new.loc[new["gap"]>1,:].reset_index()
            # for dt in new2["date"]:
            #     ax3.plot([dt,dt],[-20,df3["thi"].max()+5],color="maroon",lw=0.9,ls="--")
            # new = (df3.loc[df3["thi"]>68,:]).sort_values(by="date",ascending=False).reset_index(drop=1)
            # new["gap"] = ((new["date"].diff()).dt.days)
            # new["gap"].iloc[0] = -20
            # new2 = new.loc[new["gap"]<-1,:].reset_index()
            # for dt in new2["date"]:
            #     ax3.plot([dt,dt],[-20,df3["thi"].max()+5],color="crimson",lw=0.9,ls="--")
            # ax3.set_ylim([-20,df3["thi"].max()+5])

            # plot cow behaviour
            ax[0].set_title("")
            ax2 = ax[0].twiny()
            ax2.grid(False)
            ax2.plot(df["dim"], df["activity"], linestyle="-", linewidth=0,
                     marker="s", markersize=0,
                     color="white")
            
            ax2.set_xlim([df["dim"].min(), df["dim"].max()])
            ax[0].plot(df["date"], df["activity"], linestyle="-", linewidth=1,
                       marker="s", markersize=2.3,
                       color="teal")
            ax[0].set_xlim([df["date"].min()-pd.Timedelta(1, unit="d"),
                            df["date"].max()+pd.Timedelta(1, unit="d")])

            # plot smoothed 1st order Savitsky-Golay filter window 7d
            ax[0].plot(df["date"], df["act_sm"], linestyle="-", linewidth=1,
                       color="blue")
            ax[0].set_title("farm " + str(farm) + ", cow " +
                            str(cowlac["animal_id"][i]) + ", parity " + str(round(cowlac["parity"][i])))
            ax2.set_xlabel("DIM")
            # ax[0].set_xlabel("date")
            ax[0].set_ylabel("$ACT_s$")

            # plot herd avg + std
            df2 = act.loc[(act["date"] >= df["date"].min()) &
                          (act["date"] <= df["date"].max()), ["date", "activity"]]
            df2 = df2.sort_values(by="date").reset_index(drop=1)
            df2 = df2.groupby(by="date").agg(
                {"activity": ["mean", "std"]}).reset_index()
            df2.columns = df2.columns.droplevel()
            df2.columns = ["date", "mean", "std"]

            # plot herd behaviour
            ax[0].fill_between(df2["date"], df2["mean"]-df2["std"], df2["mean"]+df2["std"],
                               linewidth=0.1, color="cornflowerblue", alpha=0.2)

            # set legends
            ax3.set_zorder(-1)
            ax[0].grid(True)
            ax[0].set_facecolor((1, 1, 1, 0))
            ax[0].legend(labels=["standardised activity", "trend", "herd [avg+std]"],
                         facecolor="white", loc="lower right")
            ax3.legend(labels=["THI", 'THI >= 68'], bbox_to_anchor=(1.01, 1.05),
                       loc='upper left', borderaxespad=0)
            ax3.set_zorder(1)

            # plot milk production
            ax[1].plot(dfm["date"], dfm["dmy"], linestyle="-", linewidth=1,
                       marker="s", markersize=2.3,
                       color="indigo")
            # ax[1].set_title("daily milk production")
            ax[1].set_xlabel("date")
            ax[1].set_ylabel("DMY [kg]")
            ax[1].set_ylim(0,65)

            # plot herd production / parity
            dfm2 = milk.loc[(milk["date"] >= df["date"].min()) &
                            (milk["date"] <= df["date"].max()) &
                            (milk["pargroup"] == df["pargroup"].iloc[0]), ["date", "dmy"]]
            ncows = len((milk.loc[(milk["date"] >= df["date"].min()) &
                                  (milk["date"] <= df["date"].max()) &
                                  (milk["pargroup"] == df["pargroup"].iloc[0]), ["animal_id", "parity"]]).drop_duplicates())
            dfm2 = dfm2.sort_values(by="date").reset_index(drop=1)
            dfm2 = dfm2.groupby(by="date").agg(
                {"dmy": ["mean", "std"]}).reset_index()
            dfm2.columns = dfm2.columns.droplevel()
            dfm2.columns = ["date", "mean", "std"]

            # plot herd production level
            ax[1].fill_between(dfm2["date"], dfm2["mean"]-dfm2["std"], dfm2["mean"]+dfm2["std"],
                               linewidth=0.1, color="plum", alpha=0.2)
            ax[1].legend(["dmy", "pargroup dmy"])
            # ax[1].set_title(
            #     "daily milk production, ncows pargroup = " + str(ncows))

            #☺thi data loaded (manually via L915-937)
            ax6 = ax[1].twiny()
            ax6.grid(False)
            ax6.plot(df["dim"], df["activity"], linestyle="-", linewidth=0,
                      marker="s", markersize=0,
                      color="white")
            ax6.set_xlim([df["dim"].min(), df["dim"].max()])
            
            ax5 = ax[1].twinx()
            ax5.grid(False)
            ax5.set_ylabel("standardized lagged\n accumulated temperature ($LAT_s$)")
            # thi["hscum"] = (thi["hscum"] - thi["hscum"].min()) / \
            #                   (thi["hscum"].max()-thi["hscum"].min())
            thi_new = thi.loc[(thi["date"]>= df["date"].min())&(thi["date"]<= df["date"].max()),:]
            ax5.plot(thi_new["date"], thi_new["hscum"], linestyle="-", linewidth=1.5,
                    color="crimson")
            ax5.set_ylim(-0.01, 3.2)
            ax5.set_yticks([0,0.5,1])
            ax5.legend(labels=["$LATs$"], bbox_to_anchor=(1.01, 1.05),
                        loc='upper left', borderaxespad=0)
            
            
            # save plots
            plt.savefig(os.path.join(path, "results", "activity",
                                     "Figure_GOTE2024_farm_" + str(farm) + "_cow_" + \
                                         str(cowlac["animal_id"][i]) + ".jpg"))
            plt.close()
            
            # save plots
            plt.savefig(os.path.join(path, "results", "activity",
                                     "sexample_ind_activity_farm_" + str(farm) + "_cow_" + str(cowlac["animal_id"][i]) + "_withmilk.tif"))
            plt.close()

# %% ACTIVITY: model ifo THI

for farm in farms:

    # activity
    act = pd.read_csv(os.path.join(path_data, "data", "act_preprocessed_"
                                   + str(farm) + ".txt"),
                      usecols=['farm_id', 'animal_id', 'parity', 'date', 'act_corr'])
    act["date"] = pd.to_datetime(act["date"], format='%Y-%m-%d')
    act = act.rename(columns={'act_corr': 'activity'})
    act["pargroup"] = (
        (pd.concat([act["parity"], pd.DataFrame(
            3*np.ones((len(act), 1)))], axis=1))
        .min(axis=1)
    )

    # select dates
    act = act.loc[(act["date"].dt.year >= startdates[farm]) &
                  (act["date"].dt.year <= enddates[farm]), :].reset_index(drop=1)

    # read milk yield data to add dim / parity information
    milk = pd.read_csv(os.path.join(path_data, "data", "milk_preprocessed_"
                                    + str(farm) + ".txt"),
                       usecols=["animal_id", "parity", "date", "dim", "dmy"])
    milk["date"] = pd.to_datetime(milk["date"], format='%Y-%m-%d')

    # merge act and milk, delete milk
    act = pd.merge(act, milk, how="inner", on=["animal_id", "parity", "date"])
    act = act.sort_values(by=["animal_id", "date"]).reset_index(drop=1)

    # add pargroup to milk
    milk["pargroup"] = (
        (pd.concat([milk["parity"], pd.DataFrame(
            3*np.ones((len(act), 1)))], axis=1))
        .min(axis=1)
    )

    # add identifyer for cowlac for the modelling
    cowlac = act[["animal_id", "parity"]].drop_duplicates(
    ).reset_index(drop=1).reset_index()
    cowlac = cowlac.rename(columns={"index": "ID"})
    act = act.merge(cowlac, how="inner", on=["animal_id", "parity"])

    # check number of animals and lactations, and steps/day on average
    cowlac = act[["animal_id", "parity"]].drop_duplicates().reset_index()
    print("Farm " + str(farm) + ", number of cowlac = " + str(len(cowlac)))
    print("Farm " + str(farm) + ", number of cows = " +
          str(len(cowlac["animal_id"].drop_duplicates())))

    # load thi data
    thi = pd.read_csv(os.path.join(path, "data", "weatherfeatures_" + str(farm) + ".txt"),
                      index_col=0)
    thi["date"] = pd.to_datetime(thi["date"], format='%Y-%m-%d')

    # merge thi with act on date
    data = pd.merge(act, thi, on=["farm_id", "date"], how="outer")
    data = data.sort_values(by=["farm_id", "animal_id", "pargroup", "dim"])

    # delete weather info when no cow data
    data = data.loc[data["animal_id"].isna() == False, :].reset_index(drop=1)

    # add ls
    data.loc[(data["dim"]) < 22, "ls"] = "0-21"
    data.loc[(data["dim"] >= 22) &
             (data["dim"] < 61), "ls"] = "22-60"
    data.loc[(data["dim"] >= 61) &
             (data["dim"] < 121), "ls"] = "61-120"
    data.loc[(data["dim"] >= 121) &
             (data["dim"] < 201), "ls"] = "121-200"
    data.loc[(data["dim"] >= 201), "ls"] = ">200"

    # if interaction terms, standardise/scale for convergence + int
    data = data.loc[data["thi_avg"].isna() == False, :].reset_index(drop=1)
    data.thi_avg = round(data.thi_avg).astype(int)

    # data["thi_std"] = (data.thi_avg - data.thi_avg.mean()) / data.thi_avg.std()
    data["thi_std"] = (data.thi_avg - data.thi_avg.min()) / (data.thi_avg.max()-data.thi_avg.min())

    
    
    """
    #!!!NEW -- not implemented, it makes much more sense to do normalisation than min-max standardisation
    #IF outlier-sensitive
    # https://stats.stackexchange.com/questions/547446/z-score-vs-min-max-normalization
    data["thi_std"] = (data["thi_avg"]-data["thi_avg"].min())/(data["thi_avg"].max()-data["thi_avg"].min())
    data["act_new"] = (data["activity"]-data["activity"].min())/(data["activity"].max()-data["activity"].min())
    """
    
    #data["act_new"] = (data["activity"]-data["activity"].mean())/(data["activity"].std()) 
    # change name of activity
    data = data.rename(columns={"activity": "act_new"})

    # """
    # TEST:  add daylight data
    # """
    # dl = pd.read_csv(os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
    #                     "projects","ugent","heatstress","data","daylight.txt"),
    #                  index_col = 0)
    # dl["date"] = pd.to_datetime(dl["date"], format='%Y-%m-%d')
    # data = data.merge(dl, on = ["date"], how="inner")
    # data["hrs_daylight"] = (data["hrs_daylight"]-data["hrs_daylight"].min()) / \
    #                         (data["hrs_daylight"].max()-data["hrs_daylight"].min())
    # data = data.dropna().reset_index(drop=1)
   
    # # try including a quadratic term for capturing higher act with high thi values
    # md = smf.mixedlm("act_new ~ hrs_daylight + thi_std + np.power(thi_std, 2) +  np.power(thi_std, 3) + C(ls) + C(pargroup) + thi_std*C(ls) + C(year) + C(season) + C(year)*C(season)",
    #                  data=data,
    #                  groups=data["ID"],
    #                  re_formula="~thi_std")

    # mdf = md.fit(method=["lbfgs"])

    # print(mdf.summary())
    
    # """
    # TEST:  add daylight data
    # """


    
    
    # # year month combi
    # ys = data[["year", "month"]].drop_duplicates().reset_index(drop=1)
    # ys = ys.sort_values(by=["year", "month"])
    # ys["ymclass"] = np.linspace(1, len(ys), len(ys), endpoint=True, dtype=int)
    # data = pd.merge(data, ys, on=["year", "month"])

    # drop nas for modelling
    data = data.dropna().reset_index(drop=1)
    
    # add season 
    data["season"] = "winter"
    data.loc[(data["month"] >= 3) & (data["month"] < 6), "season"] = "spring"
    data.loc[(data["month"] >= 6) & (data["month"] < 9), "season"] = "summer"
    data.loc[(data["month"] >= 9) & (data["month"] < 12), "season"] = "autumn"
    data["season"] = pd.Categorical(data.season,
                                  categories=["winter", "spring",
                                              "summer", "autumn"],
                                  ordered=False)
   
    # avoid singularity of the model matrices
    if farm == 6:
        data = data.loc[(data["year"] > 2013) & (
            data["year"] < 2020), :].reset_index(drop=1)

    # --------------------------------------------------------------------------

    # activity new corrected for estrus as the Y variable of the model

    # md = smf.mixedlm("act_new ~ thi_std + C(ls) + C(pargroup) + thi_std*C(ls) + thi_std*C(parity) + C(ymclass)",
    #                  data=data,
    #                  groups = data["animal_id"],
    #                  re_formula = "~thi_std")

    # try including a quadratic term for capturing higher act with high thi values
    md = smf.mixedlm("act_new ~ thi_std + np.power(thi_std, 2) +  np.power(thi_std, 3) + C(ls) + C(pargroup) + thi_std*C(ls) + C(year) + C(season) + C(year)*C(season)",
                     data=data,
                     groups=data["ID"],
                     re_formula="~thi_std")

    mdf = md.fit(method=["lbfgs"])

    print(mdf.summary())

    # correlation = cov / sqrt(varx*vary)
    print(
        "correlation random thi slope and intercept = " + \
        str(round(mdf.cov_re.Group.thi_std / (np.sqrt(mdf.cov_re.Group.Group \
            * mdf.cov_re.thi_std.thi_std)), 3))
    )

    with open(os.path.join("results", "activity", "2summary_" + str(farm) + ".txt"), 'w') as fh:
        fh.write(mdf.summary().as_text())
    with open(os.path.join("results", "activity", "2randomcorrelation_" + str(farm) + ".txt"), 'w') as fh:
        fh.write(str(round(mdf.cov_re.Group.thi_std /
                 (np.sqrt(mdf.cov_re.Group.Group)*(np.sqrt(mdf.cov_re.thi_std.thi_std))), 3)))
    with open(os.path.join("results", "activity", "2wald_" + str(farm) + ".txt"), 'w') as fh:
        fh.write(str(mdf.wald_test_terms(scalar=True)))

    # calculate SSE vs SSTO
    R = {"SSTO": (np.sqrt((data["act_new"] - data["act_new"].mean())**2)).sum(),
         "SSE": (np.sqrt(mdf.resid**2)).sum()
         }
    R["R2"] = round(1 - (R["SSE"] / R["SSTO"]), 3)

    # add results to results table
    wt = mdf.wald_test_terms(scalar=True)

    if farm == farms[0]:  # create table
        restable = pd.DataFrame(columns=["farm1", "farm2", "farm3", "farm4", "farm5", "farm6"],
                                index=["R2",
                                       "R2(m)",
                                       "R2(c)",
                                       wt.table.index.values[0],
                                       wt.table.index.values[1],
                                       wt.table.index.values[2],
                                       wt.table.index.values[3],
                                       wt.table.index.values[4],
                                       wt.table.index.values[5],
                                       wt.table.index.values[6],
                                       wt.table.index.values[7],
                                       wt.table.index.values[8],
                                       wt.table.index.values[9],
                                       "random effects correlation",
                                       "residual error variance"])

    restable["farm"+str(farm)]["R2"] = R["R2"]
    restable["farm"+str(farm)][wt.table.index.values[0]] = wt.table.pvalue[0]
    restable["farm"+str(farm)][wt.table.index.values[1]] = wt.table.pvalue[1]
    restable["farm"+str(farm)][wt.table.index.values[2]] = wt.table.pvalue[2]
    restable["farm"+str(farm)][wt.table.index.values[3]] = wt.table.pvalue[3]
    restable["farm"+str(farm)][wt.table.index.values[4]] = wt.table.pvalue[4]
    restable["farm"+str(farm)][wt.table.index.values[5]] = wt.table.pvalue[5]
    restable["farm"+str(farm)][wt.table.index.values[6]] = wt.table.pvalue[6]
    restable["farm"+str(farm)][wt.table.index.values[7]] = wt.table.pvalue[7]
    restable["farm"+str(farm)][wt.table.index.values[8]] = wt.table.pvalue[8]
    restable["farm"+str(farm)][wt.table.index.values[9]] = wt.table.pvalue[9]
    restable["farm"+str(farm)]["random effects correlation"] = \
        round(mdf.cov_re.Group.thi_std / (np.sqrt(mdf.cov_re.Group.Group)
              * (np.sqrt(mdf.cov_re.thi_std.thi_std))), 3)
    restable["farm"+str(farm)]["residual error variance"] = round(mdf.scale, 3)

    # fixed effects regression only ################"
    # from statsmodels.formula.api import ols
    # lmf = ols(formula="act_new ~ thi_std + np.power(thi_std, 2) + np.power(thi_std, 3) + C(ls) + C(pargroup) + thi_std*C(ls) + C(year) + C(year)*C(season)",
    #           data=data)
    # lm = lmf.fit()
    # print(lm.summary2())
    # varlm = lm.scale  # variance error  sigma squared epsilon
    varlm = np.var(mdf.k_fe*mdf.bse_fe)

    # marginal R² = proportion of variance
    # explained by the fixed effects
    # formula 2.4 of https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0213
    # R²(LMM_m) = varlm / (varlm + mdf.cov_re.Group.Group + mdf_re.thi_std.thi_std)
    R["fixed"] = varlm / (varlm + mdf.cov_re.Group.Group +
                          mdf.cov_re.thi_std.thi_std + mdf.scale)
    # conditional R² = proportion variance explained by fixed + random effects
    # formula 2.5 of https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0213
    R["conditional"] = (varlm + mdf.cov_re.Group.Group + mdf.cov_re.thi_std.thi_std) / \
        (varlm + mdf.cov_re.Group.Group + mdf.cov_re.thi_std.thi_std + mdf.scale)

    restable["farm"+str(farm)]["R2(m)"] = round(R["fixed"], 3)
    restable["farm"+str(farm)]["R2(c)"] = round(R["conditional"], 3)

    print(restable)

    if farm == farms[-1]:
        with open(os.path.join("results", "activity", "2results_activity.txt"), 'w') as fh:
            fh.write(restable.to_string())

    # fitted values
    data["fitted_lme"] = mdf.fittedvalues
    data["residual_lme"] = mdf.resid

    # check normality/variance and linearity of residuals
    # to make the residuals lie around zero with higher THI, a second and third order term was included
    # heteroscedastisticiy not 100% at the edges of thi, but that also has to do with amount of data
    ########
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.scatterplot(data=data,
                    x="thi_avg", y="residual_lme", style="pargroup", hue="pargroup",
                    palette="rocket", s=6)
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
    ax.plot([25, 78], [0, 0], "k--", lw=1)
    ax.set_title("model residuals ifo of average thi, farm = "+str(farm))
    ax.set_ylabel("activity")
    ax.set_xlabel("thi")
    ax.legend(labels=["1", "2", "3"], fontsize="small")
    plt.savefig(os.path.join(path, "results", "activity", "smdl_res_vs_thi_farm_" +
                             str(farm) + ".tif"))
    plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(17, 14))
    sns.boxplot(data=data,
                x="thi_avg", y="residual_lme",
                fliersize=1, ax=ax[0])
    ax[0].plot([-0.50, 52], [0, 0], lw=1, color="r", ls="--")
    ax[0].set_xlim(-0.5, 51.5)
    ax[0].set_title("model residuals ifo average thi, farm = " + str(farm))
    ax[0].set_xlabel("average thi")
    ax[0].set_ylabel("act excl estrus - model residuals")
    sns.countplot(data=data,
                  x="thi_avg", ax=ax[1])
    ax[1].set_xlim(-0.5, 51.5)
    ax[1].set_title("number of observations at each thi, farm = "+str(farm))
    ax[1].set_xlabel("average thi")
    ax[1].set_ylabel("no. of observations")
    plt.savefig(os.path.join(path, "results", "activity", "smdl_res2_vs_thi_farm_" +
                             str(farm) + ".tif"))
    plt.close()

    # normality of residuals and predicted vs observed plot
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    sns.distplot(mdf.resid, hist=False, kde_kws={
                 "shade": True, "lw": 1}, fit=stats.norm, ax=ax[0])
    ax[0].set_title(
        "KDE plot of residuals (blue) and normal distribution (black)")
    ax[0].set_xlabel("model residuals")
    ax[0].set_ylabel("density")

    pp = sm.ProbPlot(mdf.resid, fit=True)
    qq = pp.qqplot(marker='.', markerfacecolor='b',
                   markeredgecolor='b', alpha=0.3, ax=ax[1])
    sm.qqline(qq.axes[1], line='45', fmt='r--')
    ax[1].set_title("qq-plot, farm = "+str(farm))
    ax[1].set_xlabel("theoretical quantiles (std)")
    ax[1].set_ylabel("sample quantiles")

    ax[2].plot(data["fitted_lme"], data["act_new"],
               lw=0,
               color="indigo", marker="x", ms=3)
    ax[2].plot([-15, 15], [-15, 15], color="r", lw=1.5, ls="--")
    ax[2].set_xlim(-15, 15)
    ax[2].set_ylim(-15, 15)
    ax[2].set_xlabel("predicted (estrus corr) act")
    ax[2].set_ylabel("observed (estrus corr) act")
    ax[2].set_title("predicted vs. observed plot")
    plt.savefig(os.path.join(path, "results", "activity", "smdl_res_stats_farm_" +
                             str(farm) + ".tif"))
    plt.close()

    # summer visualisation when heatstress is relevant
    years = data["year"].drop_duplicates().sort_values().reset_index(drop=1)
    for year in years:

        # model residuals
        subset = data.loc[(data["month"] > 0) & (data["month"] <= 13) &
                          (data["year"] == year), :].reset_index(drop=1)
        thi = subset[["date", "thi_avg"]].groupby("date").mean().reset_index()
        if len(subset) > 1000:
            ds = subset[["date", "residual_lme", "act_new"]
                        ].groupby(by="date").mean().reset_index()
            ds = ds.rename(columns={"residual_lme": "avg_res",
                                    "act_new": "avg_act"})
            sub = subset[["date", "residual_lme", "act_new"]].groupby(
                by="date").quantile(0.1).reset_index()
            sub = sub.rename(columns={"residual_lme": "q10_res",
                                      "act_new": "q10_act"})
            ds["q10_res"] = sub["q10_res"]
            ds["q10_act"] = sub["q10_act"]
            sub = subset[["date", "residual_lme", "act_new"]].groupby(
                by="date").quantile(0.9).reset_index()
            sub = sub.rename(columns={"residual_lme": "q90_res",
                                      "act_new": "q90_act"})
            ds["q90_res"] = sub["q90_res"]
            ds["q90_act"] = sub["q90_act"]

            fig, ax = plt.subplots(2, 1, figsize=(16, 12))
            ax[0].fill_between(ds["date"],
                               ds["q10_act"], ds["q90_act"],
                               alpha=0.85, color="mediumblue", lw=1)
            ax[0].plot(ds["date"], ds["avg_act"], "lightcyan", lw=1.5)
            ax[0].plot(ds["date"], ds["avg_act"].mean() *
                       np.ones((len(ds), 1)), "k--", lw=1.5)
            ax[0].set_ylabel("daily activity (estrus corrected) mean+90%CI")
            ax[0].set_xlabel("date")
            thi = subset[["date", "thi_avg"]].groupby(
                "date").mean().reset_index()
            ax2 = ax[0].twinx()
            ax2.plot(thi["date"], thi["thi_avg"], 'r', lw=1.5)
            ax2.fill_between(thi["date"],
                             68, 80, color="red", alpha=0.2
                             )
            ax2.grid(visible=False)
            ax2.set_ylim(-40, 80)
            ax2.set_xlim(subset["date"].min(), subset["date"].max())
            ax2.set_ylabel("average daily thi (red)")

            ax[1].fill_between(ds["date"],
                               ds["q10_res"], ds["q90_res"],
                               alpha=0.85, color="mediumblue", lw=1)
            ax[1].plot(ds["date"], ds["avg_res"], "lightcyan", lw=1.5)
            ax[1].plot(ds["date"], np.zeros((len(ds), 1)), "k--", lw=1.5)
            ax[1].set_ylabel("residual activity - mean+90%CI")
            ax2 = ax[1].twinx()
            ax2.plot(thi["date"], thi["thi_avg"], 'r', lw=1.5)
            ax2.grid(visible=False)
            ax2.set_ylim(-40, 80)
            ax2.set_ylabel("average daily thi (red)")
            ax[1].set_xlabel("date")
            ax2.fill_between(thi["date"],
                             68, 80, color="red", alpha=0.2
                             )
            ax2.set_xlim(thi["date"].min(), thi["date"].max())
            plt.savefig(os.path.join(path, "results", "activity", "smdl_act_res_year_" + str(int(year)) + "_farm_" +
                                     str(farm) + ".tif"))
            plt.close()

            # episodes of very high THI - plot separately
            # first and last month with high THI
            thihigh = subset.loc[subset["thi_avg"] > 68, "month"]

            subset = data.loc[(data["month"] >= thihigh.min()) & (data["month"] <= thihigh.max()) &
                              (data["year"] == year), :].reset_index()
            if len(subset) > 500:
                ds = subset[["date", "residual_lme", "act_new"]
                            ].groupby(by="date").mean().reset_index()
                ds = ds.rename(columns={"residual_lme": "avg_res",
                                        "act_new": "avg_act"})
                sub = subset[["date", "residual_lme", "act_new"]].groupby(
                    by="date").quantile(0.1).reset_index()
                sub = sub.rename(columns={"residual_lme": "q10_res",
                                          "act_new": "q10_act"})
                ds["q10_res"] = sub["q10_res"]
                ds["q10_act"] = sub["q10_act"]
                sub = subset[["date", "residual_lme", "act_new"]].groupby(
                    by="date").quantile(0.9).reset_index()
                sub = sub.rename(columns={"residual_lme": "q90_res",
                                          "act_new": "q90_act"})
                ds["q90_res"] = sub["q90_res"]
                ds["q90_act"] = sub["q90_act"]

                fig, ax = plt.subplots(2, 1, figsize=(16, 12))
                ax[0].fill_between(ds["date"],
                                   ds["q10_act"], ds["q90_act"],
                                   alpha=0.85, color="teal", lw=1)
                ax[0].plot(ds["date"], ds["avg_act"], "lightcyan", lw=1.5)
                ax[0].plot(ds["date"], ds["avg_act"].mean() *
                           np.ones((len(ds), 1)), "k--", lw=1.5)
                ax[0].set_ylabel(
                    "daily activity (estrus corrected) mean+90%CI")
                ax[0].set_xlabel("date")
                thi = subset[["date", "thi_avg"]].groupby(
                    "date").mean().reset_index()
                ax2 = ax[0].twinx()
                ax2.plot(thi["date"], thi["thi_avg"], 'r', lw=1.5)
                ax2.fill_between(thi["date"],
                                 68, 80, color="red", alpha=0.2
                                 )
                ax2.grid(visible=False)
                ax2.set_ylim(-40, 80)
                ax2.set_xlim(subset["date"].min(), subset["date"].max())
                ax2.set_ylabel("average daily thi (red)")

                ax[1].fill_between(ds["date"],
                                   ds["q10_res"], ds["q90_res"],
                                   alpha=0.85, color="teal", lw=1)
                ax[1].plot(ds["date"], ds["avg_res"], "lightcyan", lw=1.5)
                ax[1].plot(ds["date"], np.zeros((len(ds), 1)), "k--", lw=1.5)
                ax[1].set_ylabel("residual activity - mean+90%CI")
                thi = subset[["date", "thi_avg"]].groupby(
                    "date").mean().reset_index()
                ax2 = ax[1].twinx()
                ax2.plot(thi["date"], thi["thi_avg"], 'r', lw=1.5)
                ax2.grid(visible=False)
                ax2.set_ylim(-40, 80)
                ax2.set_ylabel("average daily thi (red)")
                ax[1].set_xlabel("date")
                ax2.fill_between(thi["date"],
                                 68, 80, color="red", alpha=0.2
                                 )
                ax2.set_xlim(thi["date"].min(), thi["date"].max())
                plt.savefig(os.path.join(path, "results", "activity", "smdl_act_res_summer_year_" + str(int(year)) + "_farm_" +
                                         str(farm) + ".tif"))
                plt.close()


#%%###########################################################################
#########################   DAILY MILK PRODUCTION  ###########################
##############################################################################

for farm in farms:

    # read milk yield data
    data = pd.read_csv(os.path.join(path_data, "data", "milk_preprocessed_"
                                    + str(farm) + ".txt"),
                       usecols=["farm_id", "animal_id", "parity", "date", "dim", "dmy"])
    data["date"] = pd.to_datetime(data["date"], format='%Y-%m-%d')
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month

    # add parity class
    data["newpar"] = (
        (pd.concat([data["parity"], pd.DataFrame(
            3*np.ones((len(data), 1)))], axis=1))
        .min(axis=1)
    )

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

    # select data in 'years' as activity data + extra criterion > 50 measurements > 400 cowlac
    data = data.loc[(data["date"].dt.year >= startdates[farm]) &
                    (data["date"].dt.year <= enddates[farm]), :].drop_duplicates().reset_index(drop=1)
    test2 = data[["animal_id", "dmy", "parity"]].groupby(
        by=["animal_id", "parity"]).count().reset_index()
    data = (data.merge(test2[["animal_id", "parity"]], how="inner", on=[
            "animal_id", "parity"])).reset_index(drop=1)
    del test2

    # calculate herdpar averages + correct dmy
    herdpar = data[["parity", "dim", "dmy"]].groupby(
        by=["parity", "dim"]).mean().reset_index()
    herdpar = herdpar.rename(columns={"dmy": "paravg"})
    data = pd.merge(data, herdpar, on=["parity", "dim"])
    data["res_pc"] = data["dmy"] - data["paravg"]

    # add identifyer for cowlac for the modelling
    cowlac = data[["animal_id", "parity"]].drop_duplicates().reset_index()
    cowlac = cowlac.rename(columns={"index": "ID"})
    data = data.merge(cowlac, how="inner", on=["animal_id", "parity"])

    # load thi
    thi = pd.read_csv(os.path.join(path, "data", "weatherfeatures_" + str(farm) + ".txt"),
                      usecols=["date", "thi_avg", "temp_hrs_low", "temp_hrs_high"])
    # thi = pd.read_csv(os.path.join(path,"data","weatherfeatures_" + str(farm) + ".txt"),
    #                  index_col = 0, usecols = ["farm_id","date","thi_avg"] )
    thi["date"] = pd.to_datetime(thi["date"], format='%Y-%m-%d')

    # calculate new weather feature thi heat stressed per day = 0.5*hrs mod temp + 2* hrs high temp
    thi["hs"] = 0.5 * (24-thi["temp_hrs_low"]-thi["temp_hrs_high"]) + \
        2 * thi["temp_hrs_high"]
    # fig,ax = plt.subplots(1,1,figsize = (12,6))
    # sns.lineplot(data = thi,
    #                 x = "date", y = "hs")
    # ax2 = ax.twinx()
    # sns.lineplot(data = thi,x="date",y="thi_avg",ax = ax2, color = "red")

    # delete  vars thi
    thi = thi.drop(columns=["temp_hrs_high", "temp_hrs_low"])

    # calculate rolling hs cumulation for delayed heat stress effect from hs in prev 4 days
    hs = thi["hs"].rolling(window=4).sum()
    thi["hscum"] = 0
    thi.iloc[1:, 3] = hs.iloc[0:-1]
    thi = thi.fillna(0)
    # fig,ax = plt.subplots(1,1,figsize = (12,6))
    # sns.lineplot(data = thi,
    #                 x = "date", y = "hscum")
    # ax2 = ax.twinx()
    # ax2.fill_between(thi["date"],
    #                   68,80, color = "red",alpha = 0.2
    #                   )
    # sns.lineplot(data = thi,x="date",y="thi_avg",ax = ax2, color = "red")

    # merge thi with milk on date
    data = pd.merge(data, thi, on=["date"], how="outer")
    data = data.sort_values(
        by=["animal_id", "newpar", "dim"]).reset_index(drop=1)

    # delete weather info when no cow data
    data = data.loc[data["animal_id"].isna() == False, :].reset_index(drop=1)

    # change LS back to non-ordered categorical, idem for parity class
    data.ls = pd.Categorical(data.ls, ordered=False)
    data.newpar = pd.Categorical(data.newpar, ordered=False)
    data = data.loc[data["thi_avg"].isna() == False, :].reset_index(drop=1)
    data.thi_avg = round(data.thi_avg).astype(int)

    # if interaction terms, standardise/scale for convergence
    # avg sd
    # data["thi_std"] = (data.thi_avg - data.thi_avg.mean()) / data.thi_avg.std()
    # data["hscum"] = (data.hscum - data.hscum.mean()) / data.thi_avg.std()
    # standardisation min-max
    data["thi_std"] = (data["thi_avg"] - data["thi_avg"].min()) / \
                      (data["thi_avg"].max()-data["thi_avg"].min())
    data["hsc_std"] = (data["hscum"] - data["hscum"].min()) / \
                      (data["hscum"].max()-data["hscum"].min())

    data["res_pc"] = (data.res_pc - data.res_pc.mean()) / data.res_pc.std()

    # year season
    ys = data[["year", "month"]].drop_duplicates().reset_index(drop=1)
    ys["season"] = "winter"
    ys.loc[(ys["month"] >= 3) & (ys["month"] < 6), "season"] = "spring"
    ys.loc[(ys["month"] >= 6) & (ys["month"] < 9), "season"] = "summer"
    ys.loc[(ys["month"] >= 9) & (ys["month"] < 12), "season"] = "autumn"
    ys["season"] = pd.Categorical(ys.season,
                                  categories=["winter", "spring",
                                              "summer", "autumn"],
                                  ordered=False)
    yscombi = ys[["year", "season"]].drop_duplicates().sort_values(
        by=["year", "season"]).reset_index(drop=1)
    yscombi["ysclass"] = np.linspace(
        1, len(yscombi), len(yscombi), endpoint=True, dtype=int)
    ys = pd.merge(ys, yscombi, on=["year", "season"])
    data = pd.merge(data, ys, on=["year", "month"])
    del ys, yscombi

    # # remove empty classes
    # test = data[["res_pc","year","season"]].groupby(by =["year","season"]).count()
    # test = test.loc[test["res_pc"] > 1,:].reset_index()
    # test = test.drop(columns = ["res_pc"])
    # data2 = data.merge(test,how = "inner")
    # if only one year season combination, select data from next year only to
    # avoid singularity of the model matrices
    if farm == 6:
        data = data.loc[(data["year"] > 2009) & (
            data["year"] < 2023), :].reset_index(drop=1)

    #############################################################################
    # first model: res_pc ~ 1 + thi + ls + parity + thi*ls + thi*parity + ys + (1 + thi | animal_id)

    md = smf.mixedlm("res_pc ~ thi_std + hsc_std + C(ls) + thi_std*C(ls) + C(year) + C(season) + C(year)*C(season)",
                     data=data,
                     groups=data["ID"],
                     re_formula="~thi_std")

    mdf = md.fit(method=["lbfgs"])

    print(mdf.summary())

    # correlation = cov / sqrt(varx)*sqrt(vary)
    print(
        "correlation random thi slope and intercept = " +
        str(round(mdf.cov_re.Group.thi_std / (np.sqrt(mdf.cov_re.Group.Group)
            * (np.sqrt(mdf.cov_re.thi_std.thi_std))), 3))
    )

    with open(os.path.join("results", "milk", "summary_" + str(farm) + ".txt"), 'w') as fh:
        fh.write(mdf.summary().as_text())
    with open(os.path.join("results", "milk", "randomcorrelation_" + str(farm) + ".txt"), 'w') as fh:
        fh.write(str(round(mdf.cov_re.Group.thi_std /
                 (np.sqrt(mdf.cov_re.Group.Group)*(np.sqrt(mdf.cov_re.thi_std.thi_std))), 3)))
    with open(os.path.join("results", "milk", "wald_" + str(farm) + ".txt"), 'w') as fh:
        fh.write(str(mdf.wald_test_terms(scalar=True)))

    # calculate SSE vs SSTO
    R = {"SSTO": (np.sqrt((data["res_pc"] - data["res_pc"].mean())**2)).sum(),
         "SSE": (np.sqrt(mdf.resid**2)).sum()
         }
    R["R2"] = round(1 - (R["SSE"] / R["SSTO"]), 3)

    # add results to results table
    wt = mdf.wald_test_terms(scalar=True)

    if farm == farms[0]:  # create table
        restable = pd.DataFrame(columns=["farm1", "farm2", "farm3", "farm4", "farm5", "farm6"],
                                index=["R2",
                                       "R2(m)",
                                       "R2(c)",
                                       wt.table.index.values[0],
                                       wt.table.index.values[1],
                                       wt.table.index.values[2],
                                       wt.table.index.values[3],
                                       wt.table.index.values[4],
                                       wt.table.index.values[5],
                                       wt.table.index.values[6],
                                       wt.table.index.values[7],
                                       "random effects correlation",
                                       "residual error variance"])

    restable["farm"+str(farm)]["R2"] = R["R2"]
    restable["farm"+str(farm)][wt.table.index.values[0]] = wt.table.pvalue[0]
    restable["farm"+str(farm)][wt.table.index.values[1]] = wt.table.pvalue[1]
    restable["farm"+str(farm)][wt.table.index.values[2]] = wt.table.pvalue[2]
    restable["farm"+str(farm)][wt.table.index.values[3]] = wt.table.pvalue[3]
    restable["farm"+str(farm)][wt.table.index.values[4]] = wt.table.pvalue[4]
    restable["farm"+str(farm)][wt.table.index.values[5]] = wt.table.pvalue[5]
    restable["farm"+str(farm)][wt.table.index.values[6]] = wt.table.pvalue[6]
    restable["farm"+str(farm)][wt.table.index.values[7]] = wt.table.pvalue[7]
    restable["farm"+str(farm)]["random effects correlation"] = \
        round(mdf.cov_re.Group.thi_std / (np.sqrt(mdf.cov_re.Group.Group)
              * (np.sqrt(mdf.cov_re.thi_std.thi_std))), 3)
    restable["farm"+str(farm)]["residual error variance"] = round(mdf.scale, 3)
    # print(restable)

    """
    # fixed effects regression only ################"
    from statsmodels.formula.api import ols
    # lmf = ols(formula="res_pc ~ thi_std + C(ls) + C(newpar) + thi_std*C(ls) + thi_std*C(newpar) + C(ysclass)",
    #           data=data)
    lmf = ols(formula="res_pc ~ thi_std + hsc_std + C(ls) + thi_std*C(ls) + C(year) + C(season) + C(year)*C(season)",
              data=data)
    lm = lmf.fit()
    print(lm.summary2())
    varlm = lm.scale  # variance error sigma squared epsilon
    """
    
    # marginal R² = proportion of variance
    # explained by the fixed effects
    # formula 2.4 of https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0213
    # R²(LMM_m) = varlm / (varlm + mdf.cov_re.Group.Group + mdf_re.thi_std.thi_std)
    
    varlm = np.var(np.matmul(md.exog,mdf.fe_params))
    
    R["fixed"] = varlm / (varlm + mdf.cov_re.Group.Group +
                          mdf.cov_re.thi_std.thi_std + mdf.scale)
    # conditional R² = proportion variance explained by fixed + random effects
    # formula 2.5 of https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0213
    R["conditional"] = (varlm + mdf.cov_re.Group.Group + mdf.cov_re.thi_std.thi_std) / \
        (varlm + mdf.cov_re.Group.Group + mdf.cov_re.thi_std.thi_std + mdf.scale)

    restable["farm"+str(farm)]["R2(m)"] = round(R["fixed"], 3)
    restable["farm"+str(farm)]["R2(c)"] = round(R["conditional"], 3)

    print(restable)

    if farm == farms[-1]:
        with open(os.path.join("results", "milk", "results_milk.txt"), 'w') as fh:
            fh.write(restable.to_string())

    with open(os.path.join("results", "milk", "summary_" + str(farm) + ".txt"), 'w') as fh:
        fh.write(mdf.summary().as_text())

    print(mdf.summary())
    print(mdf.cov_re)

    # fitted values
    data["fitted_lme"] = mdf.fittedvalues
    data["residual_lme"] = mdf.resid

    # check normality/variance and linearity of residuals
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.scatterplot(data=data,
                    x="thi_avg", y="residual_lme", style="newpar", hue="newpar",
                    palette="rocket", s=6)
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
    ax.plot([25, 78], [0, 0], "k--", lw=1)
    ax.set_title("MY model residuals ifo of average thi, farm = "+str(farm))
    ax.set_ylabel("milk yield corrected")
    ax.set_xlabel("thi")
    ax.legend(labels=["1", "2", "3"], fontsize="small")
    plt.savefig(os.path.join(path, "results", "milk", "smdl_res_vs_thi_farm_" +
                             str(farm) + ".tif"))
    plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(17, 14))
    sns.boxplot(data=data,
                x="thi_avg", y="residual_lme",
                fliersize=1, ax=ax[0])
    ax[0].plot([-0.50, 56], [0, 0], lw=1, color="r", ls="--")
    ax[0].set_xlim(-0.5, 55.5)
    ax[0].set_title(
        "milk yield model residuals ifo average thi, farm = " + str(farm))
    ax[0].set_xlabel("average thi")
    ax[0].set_ylabel("milk yield corrected - model residuals")
    sns.countplot(data=data,
                  x="thi_avg", ax=ax[1])
    ax[1].set_xlim(-0.5, 55.5)
    ax[1].set_title("number of observations at each thi, farm = "+str(farm))
    ax[1].set_xlabel("average thi")
    ax[1].set_ylabel("no. of observations")
    plt.savefig(os.path.join(path, "results", "milk", "smdl_res2_vs_thi_farm_" +
                             str(farm) + ".tif"))
    plt.close()

    # normality of residuals and predicted vs observed plot
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    sns.distplot(mdf.resid, hist=False, kde_kws={
                 "shade": True, "lw": 1}, fit=stats.norm, ax=ax[0])
    ax[0].set_title(
        "KDE plot of residuals (blue) and normal distribution (black)")
    ax[0].set_xlabel("model residuals")
    ax[0].set_ylabel("density")

    pp = sm.ProbPlot(mdf.resid, fit=True)
    qq = pp.qqplot(marker='.', markerfacecolor='b',
                   markeredgecolor='b', alpha=0.3, ax=ax[1])
    sm.qqline(qq.axes[1], line='45', fmt='r--')
    ax[1].set_title("qq-plot, farm = "+str(farm))
    ax[1].set_xlabel("theoretical quantiles (std)")
    ax[1].set_ylabel("sample quantiles")

    ax[2].plot(data["fitted_lme"], data["res_pc"],
               lw=0,
               color="indigo", marker="x", ms=3)
    ax[2].plot([-10, 10], [-10, 10], color="r", lw=1.5, ls="--")
    ax[2].set_xlim(-10, 10)
    ax[2].set_ylim(-10, 10)
    ax[2].set_xlabel("predicted (parity corr) MY [kg]")
    ax[2].set_ylabel("observed (parity corr) MY [kg]")
    ax[2].set_title("predicted vs. observed plot")
    plt.savefig(os.path.join(path, "results", "milk", "smdl_res_stats_farm_" +
                             str(farm) + ".tif"))
    plt.close()

    # visualise random effects ifo thi
    re = mdf.random_effects
    re = pd.DataFrame.from_dict(re, orient="index").reset_index()
    re["rc_thi"] = re["thi_std"]/(data["thi_avg"].max()-data["thi_avg"].min())
    re["ic_thi"] = re["Group"] - \
        (re["thi_std"]*data["thi_avg"].min() /
         (data["thi_avg"].max()-data["thi_avg"].min()))
    re = re.sort_values(by="Group").reset_index(drop=1)
    n = len(re)
    cmap = plt.cm.PiYG(np.linspace(0, 1, n))
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    for cow in re.index.values:
        # print(cow)
        x = np.arange(25, 80)
        y = re.loc[cow, "ic_thi"]+re.loc[cow, "rc_thi"]*x
        plt.sca(ax[0])
        plt.plot(x, y, color=cmap[cow])

    ax[0].set_title("cow-individual random effects")
    ax[0].set_xlabel("thi")
    ax[0].set_ylabel("cow-individual random effects")
    sns.scatterplot(data=re, x="rc_thi", y="ic_thi",
                    ax=ax[1], color="orangered")
    ax[1].set_title("random effects correlation")
    ax[1].set_xlabel("random thi slope")
    ax[1].set_ylabel("random thi intercept")
    plt.savefig(os.path.join(path, "results", "milk", "smdl_random_effects_" + str(int(year)) + "_farm_" +
                             str(farm) + ".tif"))
    plt.close()

    # summer visualisation when heatstress is relevant
    years = data["year"].drop_duplicates().sort_values().reset_index(drop=1)
    for year in years:

        # model residuals
        subset = data.loc[(data["month"] > 0) & (data["month"] <= 13) &
                          (data["year"] == year), :].reset_index(drop=1)
        thi = subset[["date", "thi_avg", "hscum"]].groupby(
            "date").mean().reset_index()
        if len(subset) > 1000:
            ds = subset[["date", "residual_lme", "res_pc"]
                        ].groupby(by="date").mean().reset_index()
            ds = ds.rename(columns={"residual_lme": "avg_res",
                                    "res_pc": "avg_my"})
            sub = subset[["date", "residual_lme", "res_pc"]].groupby(
                by="date").quantile(0.1).reset_index()
            sub = sub.rename(columns={"residual_lme": "q10_res",
                                      "res_pc": "q10_my"})
            ds["q10_res"] = sub["q10_res"]
            ds["q10_my"] = sub["q10_my"]
            sub = subset[["date", "residual_lme", "res_pc"]].groupby(
                by="date").quantile(0.9).reset_index()
            sub = sub.rename(columns={"residual_lme": "q90_res",
                                      "res_pc": "q90_my"})
            ds["q90_res"] = sub["q90_res"]
            ds["q90_my"] = sub["q90_my"]

            fig, ax = plt.subplots(2, 1, figsize=(16, 12))
            ax[0].fill_between(ds["date"],
                               ds["q10_my"], ds["q90_my"],
                               alpha=0.85, color="mediumblue", lw=1)
            ax[0].plot(ds["date"], ds["avg_my"], "lightcyan", lw=1.5)
            ax[0].plot(ds["date"], ds["avg_my"].mean() *
                       np.ones((len(ds), 1)), "k--", lw=1.5)
            ax[0].set_ylabel("daily milk yield (parity corrected) mean+90%CI")
            ax[0].set_xlabel("date")
            # thi = subset[["date","thi_avg","hscum"]].groupby("date").mean().reset_index()
            ax2 = ax[0].twinx()
            ax2.plot(thi["date"], thi["thi_avg"], 'r', lw=1.5)
            # ax2.plot(thi["date"],thi["hscum"],'purple',lw=1.5)
            ax2.fill_between(thi["date"],
                             68, 80, color="red", alpha=0.2
                             )
            ax2.grid(visible=False)
            ax2.set_ylim(-40, 80)
            ax2.set_xlim(subset["date"].min(), subset["date"].max())
            ax2.set_ylabel("average daily thi (red)")

            ax[1].fill_between(ds["date"],
                               ds["q10_res"], ds["q90_res"],
                               alpha=0.85, color="mediumblue", lw=1)
            ax[1].plot(ds["date"], ds["avg_res"], "lightcyan", lw=1.5)
            ax[1].plot(ds["date"], np.zeros((len(ds), 1)), "k--", lw=1.5)
            ax[1].set_ylabel("residual milk yield - mean+90%CI")
            ax2 = ax[1].twinx()
            ax2.plot(thi["date"], thi["thi_avg"], 'r', lw=1.5)
            ax2.grid(visible=False)
            ax2.set_ylim(-40, 80)
            ax2.set_ylabel("average daily thi (red)")
            ax[1].set_xlabel("date")
            ax2.fill_between(thi["date"],
                             68, 80, color="red", alpha=0.2
                             )
            ax2.set_xlim(thi["date"].min(), thi["date"].max())
            plt.savefig(os.path.join(path, "results", "milk", "smdl_my_res_year_" + str(int(year)) + "_farm_" +
                                     str(farm) + ".tif"))
            plt.close()

            # episodes of very high THI - plot separately
            # first and last month with high THI
            thihigh = subset.loc[subset["thi_avg"] > 68, "month"]

            subset = data.loc[(data["month"] >= thihigh.min()) & (data["month"] <= thihigh.max()) &
                              (data["year"] == year), :].reset_index()
            if len(subset) > 500:
                ds = subset[["date", "residual_lme", "res_pc"]
                            ].groupby(by="date").mean().reset_index()
                ds = ds.rename(columns={"residual_lme": "avg_res",
                                        "res_pc": "avg_my"})
                sub = subset[["date", "residual_lme", "res_pc"]].groupby(
                    by="date").quantile(0.1).reset_index()
                sub = sub.rename(columns={"residual_lme": "q10_res",
                                          "res_pc": "q10_my"})
                ds["q10_res"] = sub["q10_res"]
                ds["q10_my"] = sub["q10_my"]
                sub = subset[["date", "residual_lme", "res_pc"]].groupby(
                    by="date").quantile(0.9).reset_index()
                sub = sub.rename(columns={"residual_lme": "q90_res",
                                          "res_pc": "q90_my"})
                ds["q90_res"] = sub["q90_res"]
                ds["q90_my"] = sub["q90_my"]

                fig, ax = plt.subplots(2, 1, figsize=(16, 12))
                ax[0].fill_between(ds["date"],
                                   ds["q10_my"], ds["q90_my"],
                                   alpha=0.85, color="teal", lw=1)
                ax[0].plot(ds["date"], ds["avg_my"], "lightcyan", lw=1.5)
                ax[0].plot(ds["date"], ds["avg_my"].mean() *
                           np.ones((len(ds), 1)), "k--", lw=1.5)
                ax[0].set_ylabel(
                    "daily milk yield (parity corrected) mean+90%CI")
                ax[0].set_xlabel("date")
                thi = subset[["date", "thi_avg"]].groupby(
                    "date").mean().reset_index()
                ax2 = ax[0].twinx()
                ax2.plot(thi["date"], thi["thi_avg"], 'r', lw=1.5)
                ax2.fill_between(thi["date"],
                                 68, 80, color="red", alpha=0.2
                                 )
                ax2.grid(visible=False)
                ax2.set_ylim(-40, 80)
                ax2.set_xlim(subset["date"].min(), subset["date"].max())
                ax2.set_ylabel("average daily thi (red)")

                ax[1].fill_between(ds["date"],
                                   ds["q10_res"], ds["q90_res"],
                                   alpha=0.85, color="teal", lw=1)
                ax[1].plot(ds["date"], ds["avg_res"], "lightcyan", lw=1.5)
                ax[1].plot(ds["date"], np.zeros((len(ds), 1)), "k--", lw=1.5)
                ax[1].set_ylabel(
                    "residual milk yield (parity corrected) - mean+90%CI")
                thi = subset[["date", "thi_avg"]].groupby(
                    "date").mean().reset_index()
                ax2 = ax[1].twinx()
                ax2.plot(thi["date"], thi["thi_avg"], 'r', lw=1.5)
                ax2.grid(visible=False)
                ax2.set_ylim(-40, 80)
                ax2.set_ylabel("average daily thi (red)")
                ax[1].set_xlabel("date")
                ax2.fill_between(thi["date"],
                                 68, 80, color="red", alpha=0.2
                                 )
                ax2.set_xlim(thi["date"].min(), thi["date"].max())
                plt.savefig(os.path.join(path, "results", "milk", "smdl_milk_res_summer_year_" + str(int(year)) + "_farm_" +
                                         str(farm) + ".tif"))
                plt.close()
