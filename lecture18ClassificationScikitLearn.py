import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import sklearn
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"]=[10,5]
# ignore warnings

import warnings
# Set the warning filter to ignore FutureWarning
warnings.simplefilter(action = "ignore", category = FutureWarning)
#1
# current_palette1=sns.color_palette()
# sns.palplot(current_palette1)
# plt.show()
# #2
# current_palette2=sns.color_palette()
# sns.palplot(sns.color_palette("Reds"))
# plt.show()
# #3
# current_palette3 = sns.color_palette()
# sns.palplot(sns.color_palette("BrBG", 20))
# plt.show()

full_data=pd.read_csv('titanic_dataset.csv')
full_data.shape

full_data.head()

# The distplot() function provides the most convenient way to take a quick look at univariate distribution. This function will plot a histogram that fits the kernel density estimation(KDE) of the data.
# Now let's plot the histogram of Number of parents and children of the passenger aboard(parch).
sns.histplot(full_data['Parch'],kde=False)
plt.show()

sns.distplot(full_data['Age'],hist=False)
plt.show()

plt.figure(figsize=(8,8))
sns.distplot(full_data['Age'])
plt.show()

sns.relplot(x="Age", y="Fare", col="Pclass", hue="Sex", style="Sex",kind="line", data=full_data) # scatter can be used instead of "line" plot
plt.show()

plt.figure(figsize=(8,8))
sns.scatterplot(x="Age", y="Fare", hue="Sex", data=full_data)
plt.show()

plt.figure(figsize=(8,8))
sns.lineplot(x="Age", y="Fare", hue="Sex", style="Sex", data=full_data)
plt.show()

plt.figure(figsize=(8,8))
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=full_data)
plt.show()

plt.figure(figsize=(8,8))
sns.stripplot(x="Sex", y="Age",hue='Sex', data=full_data)
plt.show()

plt.figure(figsize=(8,8))
sns.swarmplot(x="Sex", y="Age",hue='Sex', data=full_data)
plt.show()
# box plot important
plt.figure(figsize=(8,8))
sns.boxplot(x="Survived", y="Age", data=full_data)
plt.show()

sns.violinplot(x="Survived", y="Age", hue='Sex', data=full_data)
plt.show()

sns.countplot(x="Survived", data=full_data, palette="Blues");
plt.show()

plt.subplots(figsize=(8, 8))
sns.pointplot(x="Sex", y="Survived", hue="Pclass", data=full_data)
plt.show()

sns.lmplot(x="Age", y="Fare", data=full_data)
plt.show()

plt.subplots(figsize=(10, 10))
sns.heatmap(full_data.corr(), cmap = "YlGnBu", annot=True, fmt=".2f")
plt.show()

# initialize the FacetGrid object
g = sns.FacetGrid(full_data, col='Survived', row='Pclass')

g.map(plt.hist, 'Age')
g.add_legend()
plt.show()


# References:
# https://cmdlinetips.com/2020/01/ especially for heatmap and clustermap.
# https://seaborn.pydata.org/
# https://seaborn.pydata.org/api.html#api-ref
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import sklearn
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = [10,5]
# Ignore warnings

import warnings
# Set the warning filter to ignore FutureWarning
warnings.simplefilter(action = "ignore", category = FutureWarning)

full_data = pd.read_csv('/content/titanic_dataset.csv')

full_data.info()

# Heatmap
sns.heatmap(full_data.isnull(),yticklabels = False, cbar = False,cmap = 'tab20c_r')
plt.title('Missing Data: Training Set')
plt.show()

plt.figure(figsize = (10,7))
sns.boxplot(x = 'Pclass', y = 'Age', data = full_data, palette= 'GnBu_d').set_title('Age by Passenger Class')
plt.show()

def impute_ages(cols):
    Age=cols[0]
    Pclass=cols[1]

    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
    
full_data["Age"]=full_data[['Age','Pclass']].apply(impute_age,axis=1)

# Remove Cabin feature
full_data.drop('Cabin', axis = 1, inplace = True)
# Remove rows with missing data
full_data.dropna(inplace = True)
# Remove unnecessary columns
full_data.drop(['Name','Ticket'], axis = 1, inplace = True)

# Convert objects to category data type
objcat = ['Sex','Embarked']

for colname in objcat:
    full_data[colname] = full_data[colname].astype('category')

# Numeric summary
full_data.describe()
# Remove PassengerId
full_data.drop('PassengerId', inplace = True, axis = 1)

# Shape of train data
full_data.shape
# Identify categorical features
full_data.select_dtypes(['category']).columns
# Convert categorical variables into 'dummy' or indicator variables
sex = pd.get_dummies(full_data['Sex'], drop_first = True) # drop_first prevents multi-collinearity
embarked = pd.get_dummies(full_data['Embarked'], drop_first = True)

full_data.head()
# Add new dummy columns to data frame
full_data = pd.concat([full_data, sex, embarked], axis = 1)
full_data.head(5)

# Drop unecessary columns
full_data.drop(['Sex', 'Embarked'], axis = 1, inplace = True)

# Shape of train data
print('train_data shape',full_data.shape)

# Confirm changes
full_data.head()

# OBJECTIVE 2: MACHINE LEARNING
# Next, I will feed these features into various classification algorithms to determine the best performance using a simple framework: Split, Fit, Predict, Score It.

# Split data to be used in the models
# Create matrix of features
x = full_data.drop('Survived', axis = 1) # grabs everything else but 'Survived'

# Create target variable
y = full_data['Survived'] # y is the column we're trying to predict

x # x Represents the Features
x.shape

y # y represents the Target
y.shape

from sklearn import preprocessing
pre_process=preprocessing.StandardScaler().fit(x)
x_transform=pre_process.fit_transform(x)

# use x and y to split the training data into train and test set 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=.20,random_state=101)

x_train.shape
x_train

y_train

x_test.shape
x_test

# MODEL TRAINING 
from sklearn.linear_model import LogisticRegressionCV

print('Logistic Regression')
# create instance of model
log_reg=LogisticRegression()
# pass training data to the model
log_reg.fit(x_train,y_train)

# MODEL EVALUATION
from sklearn.matrics import accuracy_score
# prediction from the model
y_pred_log_reg=log_reg.predict(x_test)

#score it
print('logistic regression')

# accuracy
print('--'*30)
log_reg_accuracy=round(accuracy_score(y_test, y_pred_log_reg)*100,2)
print('Accuracy',log_reg_accuracy,'%')

# MODEL TRAINING 
from sklearn.tree import DecisionTreeClassifier
print('Decision tree classifier')

#pass the training data to the model
Dtree.fit(x_train, y_train)

# MODEL EVALUATION
from sklearn.metrics import accuracy_score
# prediction from the model
y_pred_Dtree = Dtree.predict(x_test)
# Score It

print('Decision Tree Classifier')
# Accuracy
print('--'*30)
Dtree_accuracy = round(accuracy_score(y_test, y_pred_Dtree) * 100,2)
print('Accuracy', Dtree_accuracy,'%')

# RANDOM FOREST
from sklearn.ensamble import RandomForestClassifier
print('Random Forest Classifier')
# Create instance of model
rfc = RandomForestClassifier()

# Pass training data into model
rfc.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
# prediction from the model
y_pred_rfc = rfc.predict(x_test)
# Score It

print('Random Forest Classifier')
# Accuracy
print('--'*30)
rfc_accuracy = round(accuracy_score(y_test, y_pred_rfc) * 100,2)
print('Accuracy', rfc_accuracy,'%')

# Gradient Bossting Classifier
# model training 
from sklearn.ensemble import GradientBoostingClassifier

print('Gradient Boosting Classifier')
# Create instance of model
gbc = GradientBoostingClassifier()

# Pass training data into model
gbc.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
# prediction from the model
y_pred_gbc = gbc.predict(x_test)
# Score It

print('Gradient Boosting Classifier')
# Accuracy
print('--'*30)
gbc_accuracy = round(accuracy_score(y_test, y_pred_gbc) * 100,2)
print('Accuracy', gbc_accuracy,'%')

# Sample model scores (replace these with your actual model scores)
model_scores = {
    "Logistic Regression": log_reg_accuracy,
    "Decision Tree Classifier": Dtree_accuracy,
    "Random Forest Classifier": rfc_accuracy,
    "Gradient Boosting Classifier": gbc_accuracy
}

# Sort the model scores in descending order based on their values (higher values first)
sorted_scores = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

# Display the ranking of the models
print("Model Rankings (Greater Values are better):")
for rank, (model_name, score) in enumerate(sorted_scores, start=1):
    print(f"{rank}. {model_name}: {score}")


# RESULT 
# Model Rankings (Greater Values are better):
# 1. Gradient Boosting Classifier: 84.27
# 2. Random Forest Classifier: 82.58
# 3. Logistic Regression: 82.02
# 4. Decision Tree Classifier: 78.09