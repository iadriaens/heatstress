import os
path = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","datapreparation")
os.chdir(path)

#%%

# import necessary packages
import pandas as pd
import numpy as np
from ServerConnectPWD import LT_connect
import serverSettings_mobile as ssm


pgres = LT_connect(pgres_host=ssm.p_host, pgres_port=ssm.p_port, db=ssm.db,
                   ssh=ssm.ssh, ssh_user=ssm.ssh_user,
                   ssh_host=ssm.ssh_host, ssh_pwd=ssm.ssh_pwd,
                   psql_user=ssm.psql_user, psql_pass=ssm.psql_pass)

# output path 
path_out = os.path.join("C:/","Users","u0084712","OneDrive - KU Leuven",
                    "projects","ugent","heatstress","data")

#%% select data

# select farms with activtiy data
sql_statement = f"""
SELECT DISTINCT(farm_id)
FROM activity
;
"""
farms = pgres.query(db=ssm.db, query=sql_statement,
                            psql_user=ssm.psql_user,
                            psql_pass=ssm.psql_pass)

#------------------------------------------------------------------------------
# select activity data
sql_statement = f"""
SELECT *
FROM activity
;
"""
activity = pgres.query(db=ssm.db, query=sql_statement,
                            psql_user=ssm.psql_user,
                            psql_pass=ssm.psql_pass)


# save activity in separate .txt files per farm
for farm in farms.farm_id:
    print("farm = " + str(farm))
    fn = "//activity_" + str(farm) + ".txt"
    act = activity.loc[activity["farm_id"] == farm,:]
    act.to_csv(path+fn)
    
#------------------------------------------------------------------------------
# select milk data from list of farms
sql_statement = f"""
SELECT milking_id,milking_oid, farm_id, animal_id,lactation_id,milking_system_id, weather_id, parity, started_at, ended_at, mi, dim, tmy, mylf, mylr, myrf, myrr, eclf, eclr, ecrf, ecrr, milk_t 
FROM public.milking
WHERE farm_id IN (30,31,33,34,35,38,39,40,43,44,45,46,47,48) 
;
"""
milk2 = pgres.query(db=ssm.db, query=sql_statement,
                                    psql_user=ssm.psql_user,
                                    psql_pass=ssm.psql_pass)

# save milk in separate .txt files per farm
for farm in farms.farm_id:
    print("farm = " + str(farm))
    fn = "//milk_" + str(farm) + ".txt"
    milk = milk2.loc[milk2["farm_id"] == farm,:]
    milk.to_csv(path+fn)

#------------------------------------------------------------------------------
# select milk recording data
sql_statement = f"""
SELECT dhi_id, farm_id, animal_id,lactation_id,measured_on, dim, milk_day_kg, milk_kg, fat_day_percent, protein_day_percent, lactose_day_percent, scc, last_insemination_date
FROM public.dhi
WHERE farm_id IN (30,31,33,34,35,38,39,40,43,44,45,46,47,48) 
;
"""
scc = pgres.query(db=ssm.db, query=sql_statement,
                                    psql_user=ssm.psql_user,
                                    psql_pass=ssm.psql_pass)

# save scc in separate .txt files per farm
for farm in farms.farm_id:
    print("farm = " + str(farm))
    fn = "//scc_" + str(farm) + ".txt"
    sc = scc.loc[scc["farm_id"] == farm,:]
    sc.to_csv(path+fn)


#------------------------------------------------------------------------------
# select lactation data
sql_statement = f"""
SELECT lactation_id, farm_id, animal_id, parity, calving, dry_off
FROM public.lactation
WHERE farm_id IN (30,31,33,34,35,38,39,40,43,44,45,46,47,48) 
;
"""
lac = pgres.query(db=ssm.db, query=sql_statement,
                                    psql_user=ssm.psql_user,
                                    psql_pass=ssm.psql_pass)

# save scc in separate .txt files per farm
for farm in farms.farm_id:
    print("farm = " + str(farm))
    fn = "//lac_" + str(farm) + ".txt"
    lc = lac.loc[lac["farm_id"] == farm,:]
    lc.to_csv(path+fn)


#------------------------------------------------------------------------------
# select cow data
sql_statement = f"""
SELECT animal_id, farm_id, birth_date
FROM public.animal
WHERE farm_id IN (30,31,33,34,35,38,39,40,43,44,45,46,47,48) 
;
"""
ani = pgres.query(db=ssm.db, query=sql_statement,
                                    psql_user=ssm.psql_user,
                                    psql_pass=ssm.psql_pass)

# save scc in separate .txt files per farm
for farm in farms.farm_id:
    print("farm = " + str(farm))
    fn = "//ani_" + str(farm) + ".txt"
    an = ani.loc[ani["farm_id"] == farm,:]
    an.to_csv(path+fn)


#------------------------------------------------------------------------------
## select aws data
sql_statement = f"""
SELECT *
FROM public.aws
;
"""
aws = pgres.query(db=ssm.db, query=sql_statement,
                  psql_user=ssm.psql_user,
                  psql_pass=ssm.psql_pass)


# select farm data with location (altitude, longitude)
sql_statement = f"""
SELECT farm_id, farmname, latitude, longitude
FROM public.farm
WHERE farm_id IN (30,31,33,34,35,38,39,40,43,44,45,46,47,48) 
;
"""
farm = pgres.query(db=ssm.db, query=sql_statement,
                   psql_user=ssm.psql_user,
                   psql_pass=ssm.psql_pass)

# merge with aws to get closest weather information
farm["aws_id"]=np.nan
farm["aws_dist"]=np.nan
for f in range(0,len(farm)):
    print(farm.farm_id[f])
    lat = farm.loc[farm["farm_id"]==farm.farm_id[f],"latitude"].values
    long = farm.loc[farm["farm_id"]==farm.farm_id[f],"longitude"].values
    aws["dist"] = np.sqrt((aws["latitude"]-lat)**2 + \
                          (aws["longitude"]-long)**2)
    farm["aws_id"][f] = aws.loc[aws["dist"]==aws["dist"].min(), "aws_id"].values
    farm["aws_dist"][f] = aws.loc[aws["dist"]==aws["dist"].min(), "dist"].values*1000

# select weather data from selected aws
sql_statement = f"""
SELECT weather_id, aws_id, datetime, temp, rel_humidity
FROM public.weather
WHERE aws_id in (340,350,377,380,6407,6414,6434,6438,6439,6447,6464,6477,6479)
;
"""
wea = pgres.query(db=ssm.db, query=sql_statement,
                                    psql_user=ssm.psql_user,
                                    psql_pass=ssm.psql_pass)

# write weather and farm information to csv
wea.to_csv(path+"//weather_information.txt")
farm.to_csv(path+"//farm_information.txt")





# Create a list of all active lactations
#sql_statement = f"""
#SELECT DISTINCT(lactation_id)
#FROM public.milking
#WHERE farm_id = {31} AND ended_at > '2023-01-01'
#;
#"""
#df_lactations = pgres.query(db=ssm.db, query=sql_statement,
#                            psql_user=ssm.psql_user,
#                            psql_pass=ssm.psql_pass)
#
#print(df_lactations)






