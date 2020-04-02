"""
description of the ks data we are analysisng here.
Goal: To be able to analyze data to find out impact/correlation of different factors on the success of 
securing funding.
Dependent variable is State.

This will be achieved in following steps
1. Clean up data
2. Remove unwanted data.
3. Convert categorical values to integer values.
4. 
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import json

# create ks dataframe
ks = pd.read_csv("ks.csv")


#print the list of columns
for colName in ks:
    print(colName)
    

#what are the other states ks["state"].unique()['failed', 'successful', 'live', 'canceled', 'suspended']
#Remove the live, canceled and suspended projects, assuming that failed means no funding and successful means funding
#secured
    
#Remove all the states where we do not know the status of funding

indexNames = ks[ ks['state'] == 'live' ].index
ks.drop(indexNames , inplace=True)

indexNames = ks[ ks['state'] == 'canceled' ].index
ks.drop(indexNames , inplace=True)

indexNames = ks[ ks['state'] == 'suspended' ].index
ks.drop(indexNames , inplace=True)


#now that we have just failed or successfull, we can use the same column to have integer values
ks["is_success"] = (ks["state"] == "successful").astype(int)

#drops uneccessary column data
ks = ks.drop(['blurb','country','creator','currency_symbol',
                  'current_currency','is_backing','is_starred','location',
                  'photo','profile','source_url','urls','name','slug','id',
                  'permissions','friends','created_at','deadline','currency',
                  'currency_trailing_code','state_changed_at','launched_at','fx_rate',
                  'static_usd_rate','pledged'], axis=1) 


#Analyzed total projects and success rate 
ks_success_num = ks["is_success"].sum()

ks_failed_num = len(ks) - ks_success_num

ks_TotalProjects = len(ks)

ks_SuccessRate = format(ks_success_num / ks_TotalProjects, ".2%")
print("\nKick Starter project success rate: " + str(ks_SuccessRate) + "\nout of " + str(ks_TotalProjects) + " total projects analyzed.\n") 


#remove state column as it is not needed
del ks["state"]

#Mean goals of successful vs failed Kick Starter projects
successful_sum = ks[ks['is_success']==1]['goal'].sum()
failed_sum = ks[ks['is_success']==0]['goal'].sum()

mean_Success = "${:,.2f}".format(successful_sum / ks_success_num)
mean_Failed = "${:,.2f}".format(failed_sum / ks_failed_num)


print("The average successful Kick Starter project had a goal of " + str(mean_Success) + ".")
print("The average failed Kick Starter project had a goal of " + str(mean_Failed) + ".")
print("The mean failed Kick Starter project had an average goal 15x that of the mean successful Kick Starter project.\n")
#The above summary states a good relationship beween goal and success rate of a funding


#look at mean funding vs mean goal for successful and failed Kick Starter projects
successful_fund = ks[ks['is_success']==1]['usd_pledged'].sum()
failed_fund = ks[ks['is_success']==0]['usd_pledged'].sum()

mean_FundSuccess = '${:,.2f}'.format(successful_fund / ks_success_num)
mean_FundFailed = '${:,.2f}'.format(failed_fund / ks_failed_num)

print('The average successful Kick Starter project was funded at ' + str(mean_FundSuccess) + ' vs the mean goal of ' + str(mean_Success) + '.')
print('The average failed Kick Starter project was funded at ' + str(mean_FundFailed) + ' vs the mean goal of ' + str(mean_Failed) + '.')

		
"""
Lets clean the data now and keep only those data which are making impact
data clean up, keep only the relevant data. 
Following columns are of interest, 
"backers_count", "category",  "goal","is_backing", "is_starrable", "is_starred",
"staff_pick", "state_successful", "state_failed"

the category field has json which contains slug key which is essentially category and sub category combination.
we will choose only the first path of the slug as category

removed  is_backing , is_starred because it is nan

"""


#show min 7 column data
pd.set_option('display.max_columns', 7)


#take out the category from first path of slug field
cat1 = []
for i in ks['category']:
    c1 = json.loads(i)
    cat1.append(c1["slug"].split('/')[0])

#replace the category colum with the category sliced out ofrom json field
ks["category"] = cat1

selectedCols = ["is_success","backers_count",  "category",  "goal","pledged", "is_starrable","staff_pick"]

#build the relevant data set from the ks dataset
relData = ks.loc[:,selectedCols]

#here is a cleaned data
relData.head()

import matplotlib.pyplot as plt

#check the relation between is_success and backers count
#it shows that distribution of success of scuring funds against the backers_count is not normal
successGroups = relData.groupby(['backers_count'])['is_success']
successGroups.count().plot(kind='bar')

#check the distribution of success agains different categories
#The top two categories which to secure fundint are Film & Video and Art
successGroups = relData.groupby(['category'])['is_success']
successGroups.count().plot(kind='bar')

#box plots, data gives no indications on relation
relData.plot.box(grid=True)

#histograms to be analysed
#success against backers count
relData['backers_count'].hist(by=relData['is_success'])

#success against categories
relData['category'].hist(by=relData['is_success'])


#other columns can be compared as well to find out a relation, however our dependent variable is eithe 0 or 1
#so we can user logistic regression to 


#we can  run regression
#we have total 3346 rows, we will use 2675 rows to train the data and rest to test the model

#convert the categories and staff_picked and is_starrable to integer colums

relData["category"].unique()

relData["is_starrable"] = (relData["is_starrable"] == True).astype(int)

relData["staff_pick"] = (relData["staff_pick"] == True).astype(int)

relData["is_food"] = (relData["category"] == "food").astype(int)

relData["is_film_video"] = (relData["category"] == "film & video").astype(int)

relData["is_photography"] = (relData["category"] == "photography").astype(int)

relData["is_publishing"] = (relData["category"] == "publishing").astype(int)

relData["is_art"] = (relData["category"] == "art").astype(int)

relData["is_music"] = (relData["category"] == "music").astype(int)

relData["is_comics"] = (relData["category"] == "comics").astype(int)

relData["is_games"] = (relData["category"] == "games").astype(int)

relData["is_crafts"] = (relData["category"] == "crafts").astype(int)

relData["is_dance"] = (relData["category"] == "dance").astype(int)

relData["is_tech"] = (relData["category"] == "technology").astype(int)

relData["is_fashion"] = (relData["category"] == "fashion").astype(int)

relData["is_theatre"] = (relData["category"] == "theater").astype(int)

relData["is_journo"] = (relData["category"] == "journalism").astype(int)

relData["is_design"] = (relData["category"] == "design").astype(int)

del relData["category"]

relData["is_food"]


shuRowNum = np.random.permutation(3346)
trainRows = shuRowNum[0:2676]
testRows = shuRowNum[2676:]

xTrain = relData.iloc[trainRows,1:] #take all columns
yTrain = relData.iloc[trainRows,0] #first columns

xTrain.head()

xTest = relData.iloc[testRows,1:] #take all columns
yTest = relData.iloc[testRows,0] #first columns


from sklearn import linear_model

reg = linear_model.LogisticRegression()

#traing data

model = reg.fit(xTrain,yTrain)


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
