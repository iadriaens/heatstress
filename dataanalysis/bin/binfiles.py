# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:15:55 2024

@author: u0084712
"""

print("number of unique weather stations included = " + str(len(data.aws_id.drop_duplicates())))
print("number of weather stations coast < 10kms = " + str(len(data.loc[data["coast"]==2,"aws_id"].drop_duplicates())))
print("number of weather stations coast 10kms - 50 kms = " + str(len(data.loc[data["coast"]==1,"aws_id"].drop_duplicates())))
print("number of weather stations coast > 50kms = " + str(len(data.loc[data["coast"]==0,"aws_id"].drop_duplicates())))

# summarize per AWS id
awssum = (
        data.groupby(by = ["aws_id","year"])
        .agg({
              "coast":"min",
              "HS0":"sum",
              "HS1":"sum",
              "HS2":"sum",
              "HS3":"sum",
              "HS4":"sum",
              })
        ).reset_index()

awssum["sum"] = awssum["HS0"] + awssum["HS1"] + awssum["HS2"] + awssum["HS3"] +awssum["HS4"]
