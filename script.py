#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[75]:


df = pd.read_csv('data/data.csv')
df


# In[76]:


df.isnull().sum()


# In[77]:


df = df.dropna()
df.isnull().sum()


# In[78]:


# drop id column
df = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status', 'stroke']]
df


# In[79]:


# encode
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encode_col = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

def encode(columns, df):
    for column in columns:
        df[column] = encoder.fit_transform(df[column])
    return df

df = encode(encode_col, df)
df.head()


# In[80]:


fig = (18,8)
plt.figure(figsize=fig)
heatmap = sns.heatmap(df.corr(method='pearson'), annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, cmap='inferno', linewidths=1, linecolor='Black')
heatmap.set_title('correlation heatmap between variables')
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)


# In[81]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pickle


# In[82]:


columns = list(df.columns)
columns


# In[96]:


x_col = columns[0:len(columns)-1]
y_col = columns[len(columns)-1]
y_col, x_col


# In[97]:


x = df[x_col]
y = df[y_col]


# In[98]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,random_state=42)


# In[99]:


# models
models = []


# In[100]:


# gaussiannb
pipe = Pipeline([('gnb',GaussianNB())])
pipe.fit(xtrain, ytrain)
ypred = pipe.predict(xtest)
gnb_score = accuracy_score(ytest,ypred)

location = 'models/gnb.pkl'
pickle.dump(pipe, open(location,'wb'))
models.append(
    {
        "model":"gnb",
        "accuracy":gnb_score,
        "location":location
    }
)

gnb_score


# In[101]:


# decisiontree
pipe = Pipeline([('dt',DecisionTreeClassifier())])
pipe.fit(xtrain, ytrain)
ypred = pipe.predict(xtest)
dt_score = accuracy_score(ytest,ypred)

location = 'models/dt.pkl'
pickle.dump(pipe, open(location,'wb'))
models.append(
    {
        "model":"decision tree",
        "accuracy":dt_score,
        "location":location
    }
)

dt_score


# In[102]:


# svm
pipe = Pipeline([('svm',svm.SVC(kernel='linear'))])
pipe.fit(xtrain, ytrain)
ypred = pipe.predict(xtest)
svm_score = accuracy_score(ytest,ypred)

location = 'models/svm.pkl'
pickle.dump(pipe, open(location,'wb'))
models.append(
    {
        "model":"svm",
        "accuracy":svm_score,
        "location":location
    }
)

svm_score


# In[103]:


# knn
pipe = Pipeline([('knn',KNeighborsClassifier(n_neighbors=5))])
pipe.fit(xtrain, ytrain)
ypred = pipe.predict(xtest)
knn_score = accuracy_score(ytest,ypred)

location = 'models/knn.pkl'
pickle.dump(pipe, open(location,'wb'))
models.append(
    {
        "model":"knn",
        "accuracy":knn_score,
        "location":location
    }
)

knn_score


# In[104]:


# lr
pipe = Pipeline([('lr',LogisticRegression(random_state=42))])
pipe.fit(xtrain, ytrain)
ypred = pipe.predict(xtest)
lr_score = accuracy_score(ytest,ypred)

location = 'models/lr.pkl'
pickle.dump(pipe, open(location,'wb'))
models.append(
    {
        "model":"linear regression",
        "accuracy":lr_score,
        "location":location
    }
)

lr_score


# In[105]:


import json

with open("results/models.json", "w") as outfile:
    json.dump(models, outfile)

