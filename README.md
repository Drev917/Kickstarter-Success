# A Data Analysis of Kickstarter Projects 
### Project Group: Habitual Calculators

This repository contains the files used in our team's Kickstarter Data Analysis project.

Our group intends to determine the factors that contribute to a Kickstarter project’s success in several steps:  

1. We will begin by performing exploratory data analysis (EDA) to identify any statistical issues including:  bad or inconsistent data, collinearity, lack of independence of errors, non-normality of errors, and unequal variance of errors. 
2. Next our team will clean the dataset and then look to convert categorical and relational columns into integer pointers in order to best parse the data. We can then look for relationships and correlations between the different Kickstarter campaigns. 

#### Questions that we will be trying to answer include:

- The average pledge amount by Kickerstarter campaign and whether certain categories tend to have larger goals for overall funding.
- Analysis on whether a correlation can be drawn between a campain's number of backers and success chance.
- Whether predictions can be made as to the factors that are most closely correlated with a Kickstarter project's success chances.

#### We will also be looking at hypothesis testing and statistical analysis including significance of the averages and regression analysis:

- If correlation can be determined between pledged amount and backers count by Kickstarter category and whether that leads to a project's ultimate success or failure.
-	Logistic regression to train and test the data in order to form a predictive model of success probability.

After the EDA is complete, using various selection methods (AIC, BIC, ML algorithms), we will attempt to build models using our data to answer the questions we have outlined. 
 
#### How can this data be presented? 

Our group will be using the Numpy and Pandas libaries to cleanse, categorize, and order the dataset. We will then use tools from the scikit ML library to run predictive analysis to guage the factors that contribute to a Kickstarter project's success.  

In order to visualize the data, our team will be using the Matplotlib library.
 
Our group will then present the data in an analytical format incorporating skills and statistical modeling learned through the course and from our own research.




## **A walk-through of the final analysis:**

Data scraped from webrobots on Kickstarter project campains was analyzed in order determine if predictions could be made on which factors more closely influenced whether a campaign was ultimately successful.

[ScreenShot!](https://github.com/CU-tmoney/habitualcalculators/blob/master/Proj1.JPG)

We ran correlation analysis on the columns in the dataset and created an 'is_success' column. Our team was only interested in projects that were either successful or failed and we modified the dataset using Pandas to only include projects with either of these values. This dropped a total of 357 projects from our intial dataset of 3703 rows leaving us with 3346 total rows.

As the 'is_successful' parameter was categorical in nature, we first converted to Boolean and then to integer type. Having our y-predictor as a binary value would allow us to later run a logistic regression model on the data.


[ScreenShot!](https://github.com/CU-tmoney/habitualcalculators/blob/master/Proj2.JPG)

After analyzing the different predictor variables that we could use, our team chose 'backers_count', 'category', 'goal, usd_pledged', and 'staff_pick'. These variables were tied closest to our predictor, 'is_successful', and this allowed us to tune the model in order avoid over-fitting.

[ScreenShot!](https://github.com/CU-tmoney/habitualcalculators/blob/master/Proj3.JPG)

Basic inital analysis led our team to be able to see trends between mean successful projects and mean failed projects.

[ScreenShot!](https://github.com/CU-tmoney/habitualcalculators/blob/master/Proj4.JPG)

We used K-means cluster analysis using the scientific computing libary SciPy on 'backers_count', 'goal', 'usd_pledged', where N=3, in order to segment the data. 

[ScreenShot!](https://github.com/CU-tmoney/habitualcalculators/blob/master/Proj5.JPG)

We turned to Matplotlib in order to visualize successful projects by category, by breaking the 'category' slugs into individual columns. 

[ScreenShot!](https://github.com/CU-tmoney/habitualcalculators/blob/master/Proj6.JPG)

In order to predict the binary dependent variable 'is_successful', we used logistic regression analysis and displayed the log-odds units to determine a prediction equation.

[ScreenShot!](https://github.com/CU-tmoney/habitualcalculators/blob/master/Proj7.JPG)

After re-tuning the model by truncating any unecessary variables that do not point towards a prediction, we were able to determine accuracy with a small mean squared errors. 

[ScreenShot!](https://github.com/CU-tmoney/habitualcalculators/blob/master/Proj8.JPG)

Overall, we were extremely happy to be able to determine which variables weigh heavier in determining a Kickstarter campaign projects success probability.  

[ScreenShot!](https://github.com/CU-tmoney/habitualcalculators/blob/master/Proj9.JPG)

All statistical equations and alogorithms are commented and displayed in the Python code for review, including:
- KNN
- Logistic Regression
    - Predictions
    - MSE
    - ABS of errors
- Mean/Max stats
- Relational analysis (Z-score)





