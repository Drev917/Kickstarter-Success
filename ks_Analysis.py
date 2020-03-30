# Imports
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# create ks dataframe
ks = pd.DataFrame(pd.read_csv("ks.csv"))
print(list(ks.columns))

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

#Convert successful/failed 'state' into numeric variable: successful = 1 and failed = 0
#Analyzed total projects and success rate 
ks["state_numeric"] = (ks["state"]=="successful").astype(int)
ks_sum = ks["state_numeric"].sum()
ks_TotalProjects = len(ks)
ks_SuccessRate = format(ks_sum / ks_TotalProjects, '.2%')
print("Kick Starter project success rate: " + str(ks_SuccessRate) + "\nout of " + str(ks_TotalProjects) + " total projects analyzed.\n")

#Looked at the averages of the goals for failed Kick Starters vs successful Kick Starters
successful_sum = 0
failed_sum = 0
for project in range(len(ks)):
	if (ks["state"][project] == "successful"):
		successful_sum += ks["goal"][project]
	if (ks["state"][project] == "failed"):
		failed_sum += ks["goal"][project]

mean_Success = '${:,.2f}'.format(successful_sum / ks_sum)
mean_Failed = '${:,.2f}'.format(failed_sum / (ks_TotalProjects - ks_sum))

print('The average "successful" Kick Starter project had a goal of ' + str(mean_Success) + '.')
print('The average "failed" Kick Starter project had a goal of ' + str(mean_Failed) + '.')
print('The average "failed" Kick Starters had an average goal 12x that of the "successful" Kick Starters' + '.')
