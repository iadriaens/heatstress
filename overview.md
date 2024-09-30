# Heatstress project - BOF UGent

This document describes the final code and research conducted in the context of the cow heatstress project.  

##### __Research question__  
  
Can we quantify the relation between change in individual dairy cow activity and the (potential) drop in milk yield?

##### __Context__  
  
If we can find a link between change in activity and milk yield, we can use activity (immediate effect during hot days) as a proxy for milk yield deviations (secondary effect) early, and identify more or less sensitive cows, specifically taking different covariates into account.
These covariates are e.g. lactation stage, parity, cow health history, etc.

##### __Approach__  
  
In this research, we have conducted several steps:
- Researched how to define weather features that might be more impactful to dairy cows (or are better related to their reaction to heat stress)
- Preprocessed and selected milk yield lactation curves to ensure we can quantify the heat stress related drop correctly
- Preprocessed and selected activity time series to standardise them independently of their hard- and software settings.

##### __Overview__  
  
This work is divided into several folders "__data extraction__", "__data preprocessing__", "__dataanalysis__". Each folder contains the necessary scripts to extract, preprocess, select, visualise and analyse the data. In the following, contents of each folder is described in detail. the "__Data__" folder contains the 
data as available on the Livestock Technology server in May, 2024 from all farms having both activity and milk yield available. Weather is extracted from publicly available databases as queried with the package ["Meteostat"](https://dev.meteostat.net/python/).  


### 1. data
-----------

In the _data_ folder, the _new_ folder contains the latest version of data as extracted from CowBase, with the most recent farm identification numbers. Only farms are included that have both activity and milk production data, and these 
include activity, milk production and weather from Belgian, Dutch and English farms. The weather data are extracted with Meteostat based on the geographical location of the farms. For English farms, when this location was unknown, the location of RAFT 
head office was used as the best guess. Farmid_renumber.txt contains the information that was needed to have the correct farm ids matched after they were renumbered in CowBase (dd. spring 2024). Act_data_selection_breakpoints contains the number and location of activity hardware
 breakpoints visibly noticed when exploring statistical properties over time. 


Furthermore, the following "per farm" files are stored:
- act_xx.txt, containing:
    - index (no variable name, import with ``index_col=True``)
    - activity_id (CowBase database ID)
	- farm_id
    - animal_id (unique animal identifyer)
    - lactation id (unique lactation identifyer)
    - activity_oid (orginal db id)
	- parity
	- measured_on
	- dim (days in lactation)
	- activity_total (total activity in 2h window (steps))
	- rumination_acc 
    - rumination_time
	- updated_on (last time the data of this farm was updated)

- milk_xx.txt, containing (more variables are available, but not selected for extraction):
	- index (no variable name, import with ``index_col=True``)
    - milking_id (CowBase database ID)
	- milking_oid (orginal db id)
    - farm_id
    - animal_id (unique animal identifyer)
    - lactation id (unique lactation identifyer)
	- milking system_id (ams id, unique number is number of farm ams systems)
	- parity
	- started_at (time the milking started)
	- ended_at (time the milking ended, which is typically taken as the reference time for that session)
	- mi (milking interval, diff between current and previous "ended_at" of milking), in hours
	- dim (days in lactation)
	- tmy (total milk yield)
	- mylf, myrf, mylr, myrr (milk yield in each quarter, left-right front, left-right rear)
	- eclf, ecrf, eclr, ecrr (electrical conductivity in each quarter, left-right front, left-right rear)
	- milk_t (milk temperature, typically only in some Lely AMS available) 

- newweather_xx.txt
	- index (no variable name, import with ``index_col=True``)
	- time (yyyy-mm-dd HH:MM:SS)
	- temp (in °Celcius)
	- rhum (relative humidity in %)

The folder also contains some preprocessed files, i.e. 
- "activity_estruscorr_farm_xx.txt".
	- index (no variable name, import with ``index_col=True``)
	- farm_id
    - animal_id (unique animal identifyer)
	- parity
	- date
	- activity (total sum of activity that date)
	- act_new (activity with estrus spikes filtered out)
	- 0004 > 2000 (activity in 4h window eg. 00 to 04, 04 to 08, 08 to 12 etc. uncorrected)
    - dim (days in milk)
	- f0004 > f2000 (fractions of contribution of that 4h window to total daily activity)
	- c0004 > c2000 (4 hour activity sum corrected for estrus activity, proportional to previous contribution of total daily activity)
	- week (week of the year)
	- year

In the combined / preprocessed weather files, "weather_benl_all.txt" and "weather_raft_all.txt", the weather features as calculated and defined by me are stored for all farms.
- weather_xx_all.txt:
	- index (no variable name, import with ``index_col=True``)
	- farm_id
	- time (time of the day)
	- temp
	- rhum
	- thi (temperature humidity index calculated for that time)
	- HS0 > HS4 (1 for the heat stress class it belongs to, 0 for all other classes)
	- year
	- day (day of the year)
	- hour
	- farmname
	- date (no time)

Other files outside the "new" folder contain some data to plot/visualise maps with geopandas, italian data from one farm, and non-updated datasets from a previous version of CowBase.


### 2. dataextraction
---------------------
This folder contains several scripts to connect to and extract from the CowBase database. To extract from CowBase, the script "data_extraction_cowbase.py" can be used.
To run this script, first a connection to the server with VPN need to be made, for which the steps are indicated in the commenting section. The script only focuses on farms for which there is activity data available, implying only Lely farms are queried.
Part of the code is currently commented out, as it served to extract weather data. These are now directly retrieved with the publicly available meteostat API instead of using the stored data in CowBase.
Furthermore we add the heat stress (HS) class to the weather data, and THI is calculated at an hourly basis. With the script "weather.txt", data can be visualised using GeoPandas package,
and a distinction between coastal and non coastal is made. Furthermore, following scripts are available:

- daylight.py (to extract sunset and sundawn at a certain geographical long,lat)
- ServerConnect.py (contains the functions needed to connect to the KULeuven server)
- serverSettings_raft.json (server settings to connect to raft database, now integrated in CowBase)



### 3. datapreprocessing
------------------------
The datapreprocessingfolder contains per variable type (milk, activity, weather, scc, ...) some necessary steps to preprocess and visualise the data 
before one can work with them. It also contains selection steps, for example to ensure that data with gaps, or for which other irregularities are detected, are deleted.
The visualisation happens both at farm and at individual (a selection of) level, such that one gets an idea of the quality of the data. The details of these 
preprocessing steps are described with comments and docstrings. In the following, a general overview of each script and its output is given.

- __preprocess_activity.py__
    0. import // settings // paths and constants
    1. read data
        - read raw activity values (per 2h). Delete data with incorrect dates. Select data based on reliable number of measurements and missing data.
          For research on circadian rythm: merge 2H data into 4H time periods. Select based on longitudinal consistency.
		- summarize data at herd level. Some farms experience sudden or gradual changes in statistical proporties of the activity time series.
		  If these changes are sudden, they can be processed by identifying the breakpoints and standardisation procedures. If they are gradual (for example when the 
		  activity sensor is changed only when e.g. battery is dead, and thus the changing procedure lasts up to 6 months), identification and standardisation is more difficult,
		  and requires eihter hard coding or you introduce errors. The periods in which this happens are characterised by unexpected inconsistency in variability measures, and 
		  I propose to delete these periods. 
        - removing estrus spikes. Although estrus can be affected by heat stress, and might thus be interesting to zoom in to (what are the pattern changes during or after a period of HS for estrus behaviour?),
	      we chose not to look into this question for now. We therefore removed estrus spikes by identifying their specific one-day extreme high pattern, and apply individual time-series rolling thresholds based on 
		  the MAD and surrounding level and variability. Values above the threshold are set to threshold in this case. This procedure might also introduce errors, so can be revised if deemed necessary.
		- plot randomly selected individual curves to check effect of estrus spike removal. Figures are saved in the "results" folder.
		- plot general activity patterns after preprocessing, at week level

	2. stats correction
        - after cutting out data that are inconsistent, and thus unreliable in their statistical properties, we can correct / standardise the time series at herd level.
	      For this, we need to ensure that we don't take the heat stress periods out, meaning that when activity increases during high THI periods, this difference at herd level needs to remain.
		  However, the general pattern in a year, partially caused by temperature and potentially also caused by daylight patterns, need to be removed.
		  To this end, a seven week median and q10 and q90 are used to do a "min-max", but then q10-q90 based standardisation. 
		- method 2 was not used anymore (with breakpoints/ruptures) because of the detected inconsistency in changing stats. This method is OK to use when sudden (all sensors at the same time) shifts are the case, instead of the one-sensor-by-one shifts now noticed at some farms.

- __preprocess_individualfarms_errors.py__  
	0. short script that describes some of the errors that are/were present in the farms.  

- __preprocess_farm_weather.py__  
    0. import // settings // paths and constants
	1. load data - merge and combine all farms and add the heat stress classes
    2. summarize over all farms
	3. Canadian authors have previously determined the relation between indoor and outdoor climate in modern, well ventilated barns. To define the weather features that might express heat stress
	   we use this equation to relate indoor and outdoor climate. Other weather features calculated are based on thi excess above a certain threshold, and 
	   thi classes, (recovery, no stress, mild, moderate, severe). Also, we introduced a time lagged effect of heat stress, in which a 'remainder' is calculated. The reasoning behind this is that
       when a previous heat stress event was not long ago, its effect probably continues in a new phase. This specifically is defined with a exponentially declining function of the previous excess 
       with a halftime of 2 days. The plotting and exploration of this function is included in the end of the script.
	4. decay function exploration, see 3.  

- __preprocess_milk.py__  
    0. import // settings // paths and constants
    1. read and select data
		- correct milking times and intervals (derived from data exploration that errors are caused by floating number inaccuracies in the db).
		- in Lely, the first milk is assigned a milking interval of 48h, whereas it is either non existent or the length of the dry period, so needs correction	
		- when working with ams and daily milk production, milk of the first milking is partially produced today and yesterday, and needs attribution proportionally to the interval and time proportion of that interval in these days
		- very long lactations are rare, and often results from 1. (late) abortions (continuation of milking, no dry period), 2. lack of calving dates entered, or typo in the calving date, 3. other human errors.
		  these are explored and visualised to see what needs to be done to filter this type of errors asmuch as possible) In general, we only work with the first 400d of lactation to avaid including this type of errors. 
		- gaps more than 5 successive days render data irreliable, and are deleted.
		- lactations for which we don't have the start or less than 65 days in total are deleted (not enough data to estimate normal variability and lactation curves reliably)
	2. visualisations at herd/group level plus weather.  Some farms have a circadian rythm, e.g. because of feeding strategy which is seasonal, or more/less calvings in a certain season. Still, no conscious seasonal calving is common in our AMS data, 
       as this would cause the AMS to be too full or not full enough in certain periods, which is both bad.

- __preprocess_scc.py__  
    0. import // settings // paths and constants
	1. read data per farm
		- log scc. Not used because its granularity is insufficient for now. In a later stadium, these data can be added to explain certain 

- __preprocess_italy_gote.py__  
script used to preprocess the data for the Gote et al. manuscript case on heat stress. See manuscript for details. combined preprocessing for all longitudinal data (act and milk).
	1. for activity: estrus spike removal, visualisation, selection based on completeness, breakpoint analysis for standardisation based on changing statistical properties of time series.
	2. milk yield: selection based on completeness and quality of the milk yield lactation curves.
	3. weather based on thi average per day, but also other weather features are calculated and stored (not used in manuscrpt)
	
- __preprocess_weather.py__  
	1. simple preprocessing of weather based on heat stress classes for selected farms.

__summary_data.xlsx__ contains the summary of selected data of farm characteristics, not final version with UK data and with old farmid numbering.  
Preprocessed datasets are stored in "results" > "data".


### 4. dataanalysis
-------------------
"data" contains data as processed by the analyses scripts. "results" contains the results. "bin" contains (snippets of) code not longer used. 

- __describedata.py__	
	- describes preselected and preprocessed data at farm level (stats, numbers)

- __quantifyMILK.py__  
	1. import // settings // paths and constants
	2. visualisation and modelling per lactation with wood, save selected plots
	3. perturbation quantification after iterated wood, with different categories (function). Plot selected and save.
	4. quantification of perturbations: duration severity, etc + excel save
	5. figures + analysis per parity and lactation stage
	6. THI and milk analysis together, residuals as outcome variable
	7. modelling with residuals, mixed models, model evaluations and visualisations

- __quantifyACT.py__  
	1. import // settings // paths and constants
	2. read and visualise the activity data after preprocessing, incl. per parity and LS
	3. visualise individual curves, also per parity
	4. combine 

- __quantifyTHI.py__  (old datasets with farm data only)  
	1. import // settings // paths and constants
	2. add THI features and weather features to dataset
	3. plot and visualise (ao with geopandas)
	4. analyse also for coastal/non coastal

- __explore_weather.py__  
	1. import // settings // paths and constants
	2. load data of farms
	3. visualise, quantify and explore data of weather and related features at farms, incl thi
	4. explore individual features over all farms together
	5. explore correlations between features


- __testmodels.py__  
	1. import // settings // paths and constants
	2. load data of farms, and select daat in which act is constant without jumps
	3. add features of weather to dataset for modelling
	4. pargroup correction + visualisations
	5. linear models with mixed effects + correlations between slope and intercept for dmy var
	6. comparisons with fixed effects only (benchmarking)
	

- __milk_modelling_pertexcl.py__  
	1. import // settings // paths and constants
	2. load data of farms (selection farms only for case gote incl. italian)
	3. sample and plot perturbation data for lactation curves of selected cow numbers
	4. model lactations with wood and calculate perturbations with function in dmy_functions
	5. combine plotting with thi/weather data for exploration purposes
	6. loading weather data, and add extra features
	7. set up covariates combinations for modelling with LMM
	8. select models based on GOF, and quantify results


- __explore_weather_2.py__  
    1. import // settings // paths and constants
	2. load data of farms PER HOUR
	3. explore and visualise new features with subdisvisions in coastal or non coastal
	4. visualise T, RH, THI features per region
	5. visualise duration, recovery
	6. visualise recovery features
	7. explore relation with daylight hours


- __explore_act_per_hour.py__
	1. import // settings // paths and constants
	2. load data of farms previously selected (excl UK)
	3. plot and visualise how activity changes during the day on hot days in function of thi
	4. combine data in 4h time slots
	5. explore rumination changes


- __test_smoothed_milk.py__  
	1. import // settings // paths and constants
	2. load data of farms
	3. model lactation curves with Wood + visualise²
	4. iterative wood model + visualise


- __casegote2023.py__  
	1. import // settings // paths and constants
	2. load data of farms for case study of Gote et al 2023
	3. select data with no weird jumps in activity	
	4. model activity first, combine with thi and milk for selection purposes
	5. visualisation of activity per pargroup + summary statistics 
	6. exploration of individual curves of activity, together with milk and thi for plots
	7. modelling ifo thi, after subtraction of median herd group with LMM
	8. calculation of GOF and necessary stats for paper modelling + storage of results
	9. same procedure for dmy, combine with thi and milk for selection purposes
	10. visualisation of activity per pargroup + summary statistics 
	11. exploration of individual curves of activity, together with milk and thi for plots
	12. modelling ifo thi, after subtraction of median herd group with LMM
	13. calculation of GOF and necessary stats for paper modelling + storage of results
	

- __qregexample.py__  
	1. import // settings // paths and constants
	2. prepare settings and example data	
	3. quantile regression with linear programming - with vectors 
	4. plotting


- __quantreg.py__  
	1. generalised function of quantile regression with linear programmuing in python with WEIGHTS


- __dmy_functions.py__  
	1. functions needed to model daily (or other freq) milk yield
	2. wood function (_wood_)
	3. residual calculation + save (_woodres_)
	4. iterated wood model (_itw_) with or without plots
	5. perturbations (_pert_) with or without severity indication
	6. quantile regression (_qreg_) with weights and using linear programming


- __alldata_analysis.py__  
	1. import // settings // paths and constants
	2. load (all farms incl. UK) data
	3. prepare data - merge together act, milk, new hs heat stress features
	4. preprocessing: smoothing, combine, modelling, selection of events HS
	5. events dataset - contains all events + referneces (delta compared to non affected ref baseline)
	6. plotting and visualisation of data in events settings
	7. selection step (mild/severe HS)
	8. modelling statsmodels LM
	9. results summary + visualisation FE


### 5. documentation
--------------------

The main file "20260602_heatstress.pptx" contains visualisations and conclusions of all relevant (exploratory) work I have done in the past year. 
The manuscript Gote_etal_2023.docx contains the analysis of the effects of HS at herd level, with the main conclusion that animals with higher production levels than their herd mates
are more sensitive to heat stress (stronger decline) than animals with lower levels, and idem for activity. No answer was given yet on the link between milk and activity. 
BEC3 documents are documents from a WUR project focusing on heat stress, idem rumigen. 