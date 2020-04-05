"""
Dataset: Kickstarter campaigns scraped from January 2020 on https://webrobots.io/kickstarter-datasets/
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns

# Create ks dataframe
ks = pd.read_csv("ks.csv")


#Print the list of columns
for colName in ks:
    print(colName)
    

#What are the other states ks["state"].unique()['failed', 'successful', 'live', 'canceled', 'suspended']
#Remove the live, cancelled and suspended projects, assuming that failed means no funding and successful means funding
#secured
    
#Remove all the states where we do not know the status of funding
indexNames = ks[ ks['state'] == 'live' ].index
ks.drop(indexNames , inplace=True)

indexNames = ks[ ks['state'] == 'canceled' ].index
ks.drop(indexNames , inplace=True)

indexNames = ks[ ks['state'] == 'suspended' ].index
ks.drop(indexNames , inplace=True)


#Now that we have just failed or successfull, we can convert the column to have integer values
ks["is_success"] = (ks["state"] == "successful").astype(int)

#Dropping uneccessary column data
ks = ks.drop(['blurb','country','creator','currency_symbol',
                  'current_currency','is_backing','is_starred','location',
                  'photo','profile','source_url','urls','name','slug','id',
                  'permissions','friends','created_at','deadline','currency',
                  'currency_trailing_code','state_changed_at','launched_at',
                  'fx_rate','static_usd_rate','pledged'], axis=1)


#Analyzing total projects and overall success rate 
ks_success_num = ks["is_success"].sum()

ks_failed_num = len(ks) - ks_success_num

ks_TotalProjects = len(ks)

ks_SuccessRate = format(ks_success_num / ks_TotalProjects, ".2%")
print("\nKick Starter project success rate: " + str(ks_SuccessRate) + "\nout of " + str(ks_TotalProjects) + " total projects analyzed.\n") 


#Remove state column as it is not needed
del ks["state"]

#Mean goals of successful vs failed Kickstarter projects
successful_sum = ks[ks['is_success']==1]['goal'].sum()
failed_sum = ks[ks['is_success']==0]['goal'].sum()

mean_Success = "${:,.2f}".format(ks[ks['is_success']==1]['goal'].mean())
mean_Failed = "${:,.2f}".format(ks[ks['is_success']==0]['goal'].mean())

print()
print("The average successful Kickstarter project had a goal of " + str(mean_Success) + ".")
print()
print("The average failed Kickstarter project had a goal of " + str(mean_Failed) + ".")
print()
print("The mean failed Kickstarter project had an average goal 15x that of the mean successful Kickstarter project.\n")
print()
#The above summary states a good relationship beween goal and success rate of a funding


#Look at mean funding vs mean goal for successful and failed Kickstarter projects
successful_fund = ks[ks['is_success']==1]['usd_pledged'].sum()
failed_fund = ks[ks['is_success']==0]['usd_pledged'].sum()

mean_FundSuccess = '${:,.2f}'.format(ks[ks['is_success']==1]['usd_pledged'].mean())
mean_FundFailed = '${:,.2f}'.format(ks[ks['is_success']==0]['usd_pledged'].mean())

print()
print('The average successful Kickstarter project was funded at ' + str(mean_FundSuccess) + ' vs the mean goal of ' + str(mean_Success) + '.')
print()
print('The average failed Kickstarter project was funded at ' + str(mean_FundFailed) + ' vs the mean goal of ' + str(mean_Failed) + '.')
print()
		
"""
Lets clean the data now and keep only those data which are making impact: keep only the relevant data. 
Following columns are of interest, 
"backers_count", "category",  "goal","is_backing", "is_starrable", "is_starred",
"staff_pick", "state_successful", "state_failed"

The category field has json which contains slug key which is essentially category and sub category combination.
we will choose only the first path of the slug as category

Also removed is_backing , is_starred columns because these contain NaN values and are unnecessary to the overall analysis
"""


#Show min 7 columns of data
pd.set_option('display.max_columns', 7)


#Removing the category from first path of slug field
cat1 = []
for i in ks['category']:
    c1 = json.loads(i)
    cat1.append(c1["slug"].split('/')[0])

#Replacing the category column with the category sliced from the json field
ks["category"] = cat1

selectedCols = ["is_success","backers_count",  "category",  "goal", "usd_pledged", "is_starrable","staff_pick"]

#Building the relevant data set from the ks dataset
relData = ks.loc[:,selectedCols]

#Data cleansed preview
relData.head()

#Briefly glimpse linear relationships among numeric variables ('baker count', 'goal', and 'usd pledged')

selectedCols2  = ["backers_count", "goal", "usd_pledged"]
relData2 = ks.loc[:,selectedCols2]
sns.pairplot(relData2)

#Visualizing relationship between is_success and backers count
#Identifies distribution of success of securing funds against the backers_count is not normal
successGroups = relData.groupby(['backers_count'])['is_success']
successGroups.count().plot(kind='hist')
successGroups.count().plot.hist(bins=15)
plt.ylabel('Count of Successful Projects')
plt.xlabel('Backer Count')


#Visualizing the distribution of success against different categories
#The top two categories successfully securing funding are 'Film & Video' and 'Art'
successGroups = relData.groupby(['category'])['is_success']
successGroups.count().plot(kind='bar')
plt.ylabel('Count of Successful Projects')

#Box plots, data gives no indications on relation
relData.plot.box(grid=True)

#Histograms to be analysed
#Visualizing success against backers count
relData['backers_count'].hist(by=relData['is_success'])
plt.ylabel('Count of Projects')
plt.xlabel('Number of Backers')


# Stacked histogram of backers to projects by success
relData.pivot(columns='is_success').backers_count.plot(kind='hist', stacked=True)
plt.title('Stacked Histogram:  Backers to Projects by Success')
plt.xlabel('Number of Backers')
plt.ylabel('Number of Projects')
plt.legend(['Not Success', 'Success'])

#Visualizing success against categories
relData['category'].hist(by=relData['is_success'])


#Other columns can be compared as well to determine correlation, however our dependent variable is either 0 or 1
#so we can use logistic regression


#Regression analysis
#We have a total of 3346 rows; we will use 2675 rows to train the data and rest to test the model

#Converting 'categories' and 'staff_picked' and 'is_starrable' to integer colums

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

for colName in xTrain:
    print(colName)

xTest = relData.iloc[testRows,1:] #take all columns
yTest = relData.iloc[testRows,0] #first columns


from sklearn import linear_model

reg = linear_model.LogisticRegression(solver='lbfgs')

#Training and modeling data

model = reg.fit(xTrain,yTrain)

print('\nBeta predictor values :' + str(reg.coef_)) #prints all beta values
print('Beta0 (y-intercept) :' + str(reg.intercept_)) #print value of beta0 (y-intercept)

from sklearn import metrics
#model_prediction is 1 or 0
model_prediction = reg.predict(xTest)
print('\nModel accuracy: ', metrics.accuracy_score(yTest, model_prediction))

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(yTest,model_prediction)
print('The average error that remains in our model (mse): ' + str(mse))

#Measuring the effectiveness of our model
errors = (model_prediction - yTest)

#Number of wrong predictions
print('\nNumber of wrong predictions: ' + str(sum(abs(errors))))


#K-Nearest neighbors
# Ramdomly choose the values that closed to the means for sample values 
sample_backers_count = 215
sample_goal = 94000
sample_usd_pledged =23500

backers_countDiff = abs(sample_backers_count-relData['backers_count']) #difference for backers_count
goalDiff = abs(sample_goal-relData['goal']) #difference for goal
usd_pledgedDiff = abs(sample_usd_pledged-relData['usd_pledged']) # difference for usd_pledged
relData['dist'] = (backers_countDiff**2+goalDiff**2+usd_pledgedDiff**2)**0.5

sort_data = relData.sort_values(by='dist', ascending=True) #sort data by distance

FiftyNearestDist = sort_data['dist'].head(50) #select 50 nearest neighbor

print('\nFifty K-Nearest neighbors : ' + str(list(FiftyNearestDist)))

# model without "is_starrable"
# is starrable has no impact on the original model, has a beta of 0
# this model has better accuracy, lower error, lower wrong predictions

ksRerun = relData
del ksRerun["is_starrable"]
ksRerun.head()

for colName in ksRerun:
    print(colName)


shuRowNum = np.random.permutation(3346)
trainRows = shuRowNum[0:2676]
testRows = shuRowNum[2676:]

xTrainRerun = ksRerun.iloc[trainRows,1:] #take all columns
yTrainRerun = relData.iloc[trainRows,0] #first columns

xTrainRerun.head()

for colName in xTrainRerun:
    print(colName)

xTestRerun = ksRerun.iloc[testRows,1:] #take all columns
yTestRerun = ksRerun.iloc[testRows,0] #first columns


from sklearn import linear_model

regRerun = linear_model.LogisticRegression(solver='lbfgs')

#Training and modeling data

modelRerun = reg.fit(xTrainRerun,yTrainRerun)

print('\nBeta predictor values :' + str(modelRerun.coef_)) #prints all beta values
print('Beta0 (y-intercept) :' + str(modelRerun.intercept_)) #print value of beta0 (y-intercept)

from sklearn import metrics
#model_prediction is 1 or 0
model_prediction_Rerun = modelRerun.predict(xTestRerun)
print('\nModel accuracy: ', metrics.accuracy_score(yTestRerun, model_prediction_Rerun))

from sklearn.metrics import mean_squared_error
mseRerun = mean_squared_error(yTestRerun,model_prediction_Rerun)
print('The average error that remains in our model (mse): ' + str(mseRerun))

#Measuring the effectiveness of our model
errorsRerun = (model_prediction_Rerun - yTestRerun)

#Number of wrong predictions
print('\nNumber of wrong predictions: ' + str(sum(abs(errorsRerun))))
    
    
    
