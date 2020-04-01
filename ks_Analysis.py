# Imports
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# create ks dataframe
ks = pd.DataFrame(pd.read_csv("ks.csv"))
print(list(ks.columns))

print(list(ks.columns))

ks["state_successful"] = (ks["state"] == "successful").astype(int)
ks["state_failed"] = (ks["state"] == "failed").astype(int)
del ks["state"]

#Analyzed total projects and success rate 
ks_success_num = ks["state_successful"].sum()
ks_failed_num = ks["state_failed"].sum()
ks_TotalProjects = ks_success_num + ks_failed_num
ks_SuccessRate = format(ks_success_num / ks_TotalProjects, '.2%')
print("\nKick Starter project success rate: " + str(ks_SuccessRate) + "\nout of " + str(ks_TotalProjects) + " total projects analyzed.\n") 

successful_sum = 0
failed_sum = 0
for project in range(len(ks)):
	if (ks["state_successful"][project] == 1):
		successful_sum += ks["goal"][project]
	if (ks["state_failed"][project] == 1):
		failed_sum += ks["goal"][project]

mean_Success = '${:,.2f}'.format(successful_sum / ks_success_num)
mean_Failed = '${:,.2f}'.format(failed_sum / ks_failed_num)

print('The average successful Kick Starter project had a goal of ' + str(mean_Success) + '.')
print('The average failed Kick Starter project had a goal of ' + str(mean_Failed) + '.')
print('The mean failed Kick Starter project had an average goal 15x that of the mean successful Kick Starter project.\n')

#look at mean funding vs mean goal for successful and failed Kick Starter projects
successful_fund = 0
failed_fund = 0
for project in range(len(ks)):
	if (ks["state_successful"][project] == 1):
		successful_fund += ks["usd_pledged"][project]
	if (ks["state_failed"][project] == 1):
		failed_fund += ks["usd_pledged"][project]

mean_FundSuccess = '${:,.2f}'.format(successful_fund / ks_success_num)
mean_FundFailed = '${:,.2f}'.format(failed_fund / ks_failed_num)

print('The average successful Kick Starter project was funded at ' + str(mean_FundSuccess) + ' vs the mean goal of ' + str(mean_Success) + '.')
print('The average failed Kick Starter project was funded at ' + str(mean_FundFailed) + ' vs the mean goal of ' + str(mean_Failed) + '.')		


"""
# create boolean column for failed/successful
ks['state_bool'] = ks['state'].replace({'failed': False, 'successful': True}).astype(bool)

# correlation of columns
ks_Cor = ks.corr()

# try to find variables with high correlation to state_bool
ks_CorTarget = abs(ks_Cor['state_bool'])
rel_Feat = ks_CorTarget[ks_CorTarget > 0]  # change value to be more selective
print(rel_Feat)
'''
No good variables are found.
Using a boolean (non-continuous) Y means we'll probably need to look at other methods of variable selection
and model building.
'''

'''
CREATE DUMMY VARIABLES (OR RECODE AS INT) FOR OTHER CATEGORICAL COLUMNS TO ALLOW CORRELATION CALCULATION.
'''
"""
#Delete redundant columns

new_ks = ks.drop(['blurb','category','country','creator','currency_symbol',
                  'current_currency','is_backing','is_starred','location',
                  'photo','profile','source_url','urls','name','slug','id',
                  'permissions','friends','created_at','deadline','currency',
                  'currency_trailing_code','state_changed_at','launched_at','fx_rate',
                  'static_usd_rate','pledged','usd_pledged'], axis=1) 

print(list(new_ks.columns))
