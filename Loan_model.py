
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from sklearn import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import random
import os
import warnings
warnings.filterwarnings('ignore') # if there are any warning due to version mismatch, it will be ignored


# In[2]:

loan = pd.read_csv("loan_table.csv")
borrow = pd.read_csv("borrower_table.csv")


# In[3]:

joined_df = pd.merge(loan,borrow, on= 'loan_id', how ='inner')
joined_df.head()


# In[4]:

train = joined_df[joined_df.loan_granted == 1]
test = joined_df[joined_df.loan_granted == 0]


# In[5]:

# Remove columns where number of unique value is only 1 because that will not make any sense in the analysis
unique_train = train.nunique()
unique_train = unique_train[unique_train.values == 1]
unique_test = test.nunique()
unique_test = unique_test[unique_test.values == 1]
train.drop(labels = list(unique_train.index), axis =1, inplace=True)
print("So now we are left with",train.shape ,"rows & columns.")
test.drop(labels = list(unique_test.index), axis =1, inplace=True)
test.drop('loan_repaid',axis=1,inplace = True)
print("So now we are left with",test.shape ,"rows & columns.")


# In[6]:

# Derive some new columns based on business understanding that will be helpful in our analysis.
# replace null values of fully_repaid column with '2' (just random segregration) becasue here null value represent some useful meaning. Similar reason for currently_repaying_other_loans
# Square root transformation of saving amount and checking amount because after transdormation they follows normal distribution and since these two are most correlated with our target variable.
# replace null values of avg_percentage_credit_used by mean. Here we can also replace via median because standard deviation is very less for this features.
# Converting categorical column (loan_purpose) to numerical one (encoding.) I am not using any library like labelencoder or one hot encoding. 
# assuming checking_amount is loan_term_amount.


# In[7]:

train['fully_repaid_previous_loans'].replace(np.NaN, 2 ,inplace=True)
train['currently_repaying_other_loans'].replace(np.NaN, 2 ,inplace=True)
train['sqrt_saving_amt'] = np.sqrt(train.saving_amount)
train['sqrt_checking_amt'] = np.sqrt(train.checking_amount)
train["avg_percentage_credit_card_limit_used_last_year"].fillna(train.avg_percentage_credit_card_limit_used_last_year.mean(), inplace = True)
train['loan_purpose'] = train['loan_purpose'].apply({'home':0, 'business':1,'investment':2, 'emergency_funds':3, 'other':4}.get)


# In[8]:

# Creating a bin for dependent_number and encoding them into numerical by mapping.
train['dependent_number']= train['dependent_number'].map({0:0, 1:0,2:1,3:1,4:2,5:2,6:2,7:3,8:3})
test['dependent_number']= test['dependent_number'].map({0:0, 1:0,2:1,3:1,4:2,5:2,6:2,7:3,8:3})


# In[9]:

# Creating a bin for age features for train and test data.
bins = [0, 30, 50, 70, 100]
labels = [1,2,3,4]
train['binned_age'] = pd.cut(train['age'], bins=bins, labels=labels)


# In[10]:

bins = [0, 30, 50, 70, 100]
labels = [1,2,3,4]
test['binned_age'] = pd.cut(test['age'], bins=bins, labels=labels)


# In[11]:

test['fully_repaid_previous_loans'].replace(np.NaN, 2 ,inplace=True)
test['currently_repaying_other_loans'].replace(np.NaN, 2 ,inplace=True)
test['sqrt_saving_amt'] = np.sqrt(test.saving_amount)
test['sqrt_checking_amt'] = np.sqrt(test.checking_amount)
test["avg_percentage_credit_card_limit_used_last_year"].fillna(test.avg_percentage_credit_card_limit_used_last_year.mean(), inplace = True)
test['loan_purpose'] = test['loan_purpose'].apply({'home':0, 'business':1,'investment':2, 'emergency_funds':3, 'other':4}.get)


# In[12]:

# Taking useful features in count for creation of models
ID_col=['loan_id']
target_col=['loan_repaid']
not_useful_col=['date','is_employed','age','checking_amount','saving_amount']
features=list(set(list(train.columns))-set(ID_col)-set(target_col)-set(not_useful_col))
print(features)


# In[13]:

#GBM Model
cv=[]
lst=[]
kf=cross_validation.KFold(len(train),n_folds=5,random_state=0)
for idx1,idx2 in kf:
    x_train,x_cv=train[features].iloc[idx1],train[features].iloc[idx2]
    y_train,y_cv=train.loan_repaid.iloc[idx1],train.loan_repaid.iloc[idx2]
    random.seed(100)
    gbm = GradientBoostingClassifier()
    gbm.fit(x_train, y_train)
    cv.extend(gbm.predict(x_cv))
    lst.append(gbm.predict(test[features]))
    print(metrics.f1_score(gbm.predict(x_cv),y_cv))


# In[14]:

importances = gbm.feature_importances_
std = np.std([gbm.feature_importances_ for tree in gbm.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
names = [features[i] for i in indices]
# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, names[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), indices)
plt.xlim([-1, x_train.shape[1]])
plt.show()


# In[15]:

#Random Forest Model
cv2=[]
lst2=[]
kf=cross_validation.KFold(len(train),n_folds=5,random_state=0)
for idx1,idx2 in kf:
    x_train,x_cv=train[features].iloc[idx1],train[features].iloc[idx2]
    y_train,y_cv=train.loan_repaid.iloc[idx1],train.loan_repaid.iloc[idx2]
    random.seed(100)
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    cv2.extend(rf.predict(x_cv))
    lst2.append(rf.predict(test[features]))
    print(metrics.f1_score(rf.predict(x_cv),y_cv))


# In[16]:

importances = rf.feature_importances_
std = np.std([rf.feature_importances_ for tree in rf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
names = [features[i] for i in indices]
# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, names[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), names)
plt.xlim([-1, x_train.shape[1]])
plt.show()

