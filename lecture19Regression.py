import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import sklearn
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize']=[10,5]
# Ignore warnings

import warnings
# Set the warning filter to ignore FutureWarning
warnings.simplefilter(action = "ignore", category = FutureWarning)

full_data=pd.read_csv('USA_Housing.csv')

#shape
print('train data:',full_data.shape)
#view first few rows
full_data.head(5)

full_data.info(5)

#heatmap
sns.heatmap(full_data.isnull(),yticklabels=False,cbar=False,cmap='tab20c_r')
plt.title('missing Data: training set')
plt.show()

#remove address
full_data.drop('Address',axis=1,inplace=True)
#remove rows with missing data
full_data.drop(inplace=True)

# full_data

#numeric summary
# full_data.describe()

#shape of train data
# full_data.shape

#split data to be used in the models
#create matrix of features  
x=full_data.drop('Price',axis=1) #grab everything else but 'price'

#create target variable 
y=full_data['Price'] # y is the column i am trying to predict

from sklearn import preprocessing
pre_process=preprocessing.StandardScaler().fit(x)
x_transform=pre_process.fit_transform(x)

# pipe=make_pipeline(StandardScaler(), LogisticRegression())
# pipe.fit(X_train,y_train)

#represents the features
x_transform.shape
x_transform

y # this represents the target

y.shape
# use x and y to split the trainintg data into train and test set 
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test=train_test_split(x_transform,y,test_size=.10, random_state=101)

#fit 
#import model
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#create instance of the model
lin_reg=LinearRegression()
#pass training data into model 
lin_reg.fit(x_train, y_train)
# pipe=make_pipline(StandardScaler(),LinearRegression())
# pipe.fit(x_train,y_train)

#predict
y_pred=lin_reg.predict(x_test)
print(y_pred.shape)
print(y_pred)

sns.scatterplot(x=y_test,y=y_pred, color='blue', label='Actual data points')
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red',label='ideal line')
plt.legend()
plt.show()

#combine actual and predicted values side by side 
results=np.column_stack((y_test,y_pred))
#printing the results 
print(' Actual values | Predicted values')
print("---------------------------------")
for actual, predicted in results:
    print(f"{actual:14.2f} | {predicted:12.2f}")

# RESIDUAL ANALYSIS 
residual=actual-y_pred.reshape(-1)
print(residual)

# Distribution plot for Residual (difference between actual and predicted values)
sns.distplot(residual,kde=True)

#scoring it
from sklearn.metrics import mean_squared_error

print('Linear Regression Model')

#Results
print('--'*30)
#mean squared error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)

# print evaluation metrics
print("Mean Squared Error:",mse)
print("Root Mean Squared Error:",rmse)

# Linear Regression Model
# ------------------------------------------------------------
# Mean Squared Error: 10100187858.864885
# Root Mean Squared Error: 100499.69083964829


# 10170939558

s = 10100187858 - 9839952411
print(s)

y_train.shape

# decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

rf_regressor=DecisionTreeClassifier()
rf_regressor.fit(x_train,y_train)

#predicting the SalePrices using test set 
y_pred_rf=rf_regressor.predict(x_test)

DTr=mean_squared_error(y_pred_rf,y_test)
#regression forest Regression accuracy with test set
print('decision tree regression :',DTr)

# random forest
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor()
rf_regressor.fit(x_train,y_train)

#Predicting the SalePrices using test set
y_pred_rf = rf_regressor.predict(x_test)
RFr = mean_squared_error(y_pred_rf,y_test)
#Random Forest Regression Accuracy with test set
print('Random Forest Regression : ',RFr)

#Gradient Boosting Regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

rf_regressor = GradientBoostingRegressor()
rf_regressor.fit(x_train,y_train)

#Predicting the SalePrices using test set
y_pred_rf = rf_regressor.predict(x_test)

#Random Forest Regression Accuracy with test set
GBr = mean_squared_error(y_pred_rf,y_test)
print('Gradient Boosting Regression : ',GBr)

# Sample model scores (replace these with your actual model scores)
model_scores = {
    "Linear Regression": 9839952411.801708,
    "Descison Tree": 29698988724.82603,
    "Random Forest":14315329749.65445,
    "Gradient Boosting": 12029643835.717766
}

# Sort the model scores in ascending order based on their values (lower values first)
sorted_scores = sorted(model_scores.items(), key=lambda x: x[1])

# Display the ranking of the models
print("Model Rankings (lower values are better):")
for rank, (model_name, score) in enumerate(sorted_scores, start=1):
    print(f"{rank}. {model_name}: {score}")

# end

