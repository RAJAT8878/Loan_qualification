
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from scipy import stats
import re
import seaborn as sns

pd.options.mode.chained_assignment = None #set it to None to remove SettingWithCopyWarning
pd.options.display.float_format = '{:.4f}'.format #set it to convert scientific noations such as 4.225108e+11 to 422510842796.00
pd.set_option('display.max_columns', 25) # to display all the columns

import os
import warnings
warnings.filterwarnings('ignore') # if there are any warning due to version mismatch, it will be ignored


# In[2]:

loan = pd.read_csv("loan_table.csv")
borrow = pd.read_csv("borrower_table.csv")


# In[3]:

loan.info()


# In[4]:

loan.head(4)


# In[5]:

borrow.info()


# In[6]:

borrow.head(4)


# In[7]:

joined_df = pd.merge(loan,borrow, on= 'loan_id', how ='inner')
joined_df.head()


# In[8]:

train = joined_df[joined_df.loan_granted == 1]
test = joined_df[joined_df.loan_granted == 0]


# In[9]:

train.info()


# In[10]:

test.info()


# In[11]:

# List of Columns & NA counts where NA values are more than 30%
NA_col = train.isnull().sum()
NA_col = NA_col[NA_col.values >(0.3*len(train))]
plt.figure(figsize=(20,4))
NA_col.plot(kind='bar')
plt.title('List of Columns & NA counts where NA values are more than 30%')
plt.show()


# In[12]:

#  Remove columns where number of unique value is only 1 because that will not make any sense in the analysis
unique_train = train.nunique()
unique_train = unique_train[unique_train.values == 1]
unique_test = test.nunique()
unique_test = unique_test[unique_test.values == 1]
train.drop(labels = list(unique_train.index), axis =1, inplace=True)
train.drop('loan_id',axis=1,inplace = True)
print("So now we are left with",train.shape ,"rows & columns.")
test.drop(labels = list(unique_test.index), axis =1, inplace=True)
test.drop('loan_repaid',axis=1,inplace = True)
print("So now we are left with",test.shape ,"rows & columns.")


# In[13]:

# Data validation
# For first loan (train), previous_loan_repaid should be none. So nan means its not exist.
train.loc[train['is_first_loan'] == 1, 'fully_repaid_previous_loans'].unique()


# In[14]:

# Data validation
# For first loan (test), previous_loan_repaid should be none. So nan means its not exist.
test.loc[test['is_first_loan'] == 1, 'fully_repaid_previous_loans'].unique()


# In[15]:

# Data validation
# For first loan (train), currently_repaying_other_loans should be none. So nan means its not exist.
train.loc[train['is_first_loan'] == 1, 'currently_repaying_other_loans'].unique()


# In[16]:

# Data validation
# For first loan (test), currently_repaying_other_loans should be none. So nan means its not exist.
test.loc[test['is_first_loan'] == 1, 'currently_repaying_other_loans'].unique()


# In[17]:

# Data validation
# For not employed applicant (train), yearly_salary should be zero.
train.loc[train['is_employed'] == 0, 'yearly_salary'].unique()


# In[18]:

# Data validation
# For not employed applicant (test), yearly_salary should be zero.
test.loc[test['is_employed'] == 0, 'yearly_salary'].unique()


# In[19]:

# Analysing the distribution behaviour of loan_purpose and check whether the distribtion are same for test or not.
(train.loan_purpose.value_counts()*100)/len(train)


# In[20]:

(test.loan_purpose.value_counts()*100)/len(test)


# In[21]:

def univariate(df,col,vartype):
    
    '''
    Univariate function will plot the graphs based on the parameters.
    df      : dataframe name
    col     : Column name
    vartype : Continuos(0)   : Distribution, Violin & Boxplot will be plotted.
    
    '''
    sns.set(style="darkgrid")
    
    if vartype == 0:
        fig, ax=plt.subplots(nrows =1,ncols=3,figsize=(20,8))
        ax[0].set_title("Distribution Plot")
        sns.distplot(df[col],ax=ax[0])
        ax[1].set_title("Violin Plot")
        sns.violinplot(data =df, x=col,ax=ax[1], inner="quartile")
        ax[2].set_title("Box Plot")
        sns.boxplot(data =df, x=col,ax=ax[2],orient='v')
    else:
        exit
        
    plt.show()


# In[22]:

univariate(df=train,col='checking_amount',vartype=0)
# Insights: Most of the checking_amount are distributed between 2000 to 5000.


# In[23]:

univariate(df=train,col='yearly_salary',vartype=0)
# Insights: Most of the checking_amount are distributed between 20000 to 40000.


# In[24]:

univariate(df=train,col='saving_amount',vartype=0)
# Insights: Most of the checking_amount are distributed between 1000 to 3000.


# In[25]:

plt.figure(figsize=(10,8))
sns.boxplot(data =train, x='loan_purpose', y='checking_amount', hue ='loan_repaid')
plt.title('loan_purpose v/s checking_amount')
plt.show()


# In[26]:

plt.figure(figsize=(10,8))
sns.boxplot(data =train, x='loan_purpose', y='saving_amount', hue ='loan_repaid')
plt.title('loan_purpose v/s saving_amount')
plt.show()


# In[27]:

plt.figure(figsize=(10,8))
sns.boxplot(data =train, x='loan_purpose', y='yearly_salary', hue ='loan_repaid')
plt.title('loan_purpose v/s yearly_salary')
plt.show()


# In[28]:

plt.figure(figsize=(10,8))
sns.boxplot(data =train, x='loan_purpose', y='total_credit_card_limit', hue ='loan_repaid')
plt.title('loan_purpose v/s total_credit_card_limit')
plt.show()


# In[29]:

train_correlation = train.corr()
train_correlation


# In[30]:

f, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(train_correlation, 
            xticklabels=train_correlation.columns.values,
            yticklabels=train_correlation.columns.values,annot= True)
plt.show()


# In[31]:

list(set(train.dtypes.tolist()))


# In[32]:

train_num = train.select_dtypes(include = ['float64', 'int64'])
train_num.head()


# In[33]:

# Checking the distribution of numerical columns
train_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)


# In[34]:

train_num_corr = train_num.corr()['loan_repaid'][1:]
train_num_corr 


# In[35]:

train_num_corr = train_num.corr()['loan_repaid'][1:]
golden_features_list = train_num_corr[train_num_corr > 0.4].sort_values(ascending=False)
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))

