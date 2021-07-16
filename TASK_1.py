##  Prediction using Supervised ML 

##  ● Predict the percentage of marks of an student based on the number of study hours 
##  ● This is a simple linear regression task as it involves just 2 variables.
##  ● What will be predicted score if a student studies for 9.25 hrs/ day? 



## Importing the required datasets

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## Importing the dataset

student_scores = pd.read_csv('C:\\Users\\Niks PC\\Desktop\\Spark Foundation\\student_scores.csv')
student_scores.shape

## Getting dataset information
student_scores.info()

## Checking for the distribution of data
## Plotting the histograms
## Hours column
plt.hist(student_scores.Hours)
plt.xlabel('Hours')
## Scores column
plt.hist(student_scores.Scores)
plt.xlabel('Scores')

## Checking for the null values
student_scores.isna().sum()        ## NO null values

## checking for the outliers by plotting the boxplot

## Hours Column
sns.boxplot(student_scores['Hours'])  ## NO outliers are present

## Scores column
sns.boxplot(student_scores['Scores']) ## NO outliers are present

## Checking for the linearity between the input and output variables
## plotting the scatter plot

plt.scatter(student_scores['Hours'],student_scores['Scores'])
## From the graph there we can conclude that there is strong linear relation between the 
## input and output variable

## Confirming it with the correlation coefficient
student_scores.corr()   ## corrleation coefficient between Scores and Hours in 0.97

##Checking for normality of the dataset
student_scores.describe()

## Normalizing the data
def norm_func(i):
	x = (i-i.min())	/(i.max()-i.min())
	return(x)

norm_student_scores = norm_func(student_scores)

## Splitting the data into test and train
from sklearn.model_selection import train_test_split

student_train,student_test = train_test_split(student_scores, test_size = 0.2,random_state = 11)

## Building the model
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Scores ~ Hours', data = student_train).fit()
model.summary()  ## R-squared = 0.959  and Adj.R squared = 0.957

## Predicting on the test data
pred_test = model.predict(student_test)


##  Calculating test residuals = fitted value - actual value
test_res = pred_test - student_test['Scores']

## RMSE test
rmse_test = np.sqrt(np.mean(test_res*test_res))

rmse_test    ## 8.869


## Predicting on the train data
pred_train = model.predict(student_train)

## Calculating train residuals 
train_res = pred_train - student_train['Scores']

## RMSE train
rmse_train = np.sqrt(np.mean(train_res*train_res))

rmse_train  ## 4.653


### Importing the new data on which predictions to be made

new_score = pd.read_csv('C:\\Users\\Niks PC\\Desktop\\Book1.csv')

type(new_score) ## Dataframe
len(new_score)  ## len = 9

## Predicting on the new data
pred_new = model.predict(new_score)

pred_new

### So th approximate score for 9.25 hours of study would be 96.93
































