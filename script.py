#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[18]:


df = pd.read_csv('data/data.csv')
df


# In[19]:


df.isnull().sum()


# In[20]:


df = df.dropna()
df.isnull().sum()


# In[21]:


# drop id column
df = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status', 'stroke']]
df


# In[22]:


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


# In[23]:


fig = (18,8)
plt.figure(figsize=fig)
heatmap = sns.heatmap(df.corr(method='pearson'), annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, cmap='inferno', linewidths=1, linecolor='Black')
heatmap.set_title('correlation heatmap between variables')
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)


# In[76]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import pickle


# In[25]:


columns = list(df.columns)
columns


# In[26]:


x_col = columns[0:len(columns)-1]
y_col = columns[len(columns)-1]
y_col, x_col


# In[27]:


x = df[x_col]
y = df[y_col]


# In[54]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(df[x_col], df[y_col])


# In[55]:


xtrain, xtest, ytrain, ytest = train_test_split(X_resampled,y_resampled,test_size=0.2,random_state=0)


# In[118]:


# models
models = []


# In[119]:


# gaussiannb
pipe = Pipeline([('scaler',StandardScaler()),('gnb',GaussianNB())])
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

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(ytest,  ypred)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

gnb_score


# In[120]:


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

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(ytest,  ypred)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

dt_score


# In[121]:


# random forest
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([('scaler',StandardScaler()),('rfc',RandomForestClassifier(random_state=0,))])
pipe.fit(xtrain, ytrain)
ypred = pipe.predict(xtest)
rfc_score = accuracy_score(ytest,ypred)

location = 'models/rfc.pkl'
pickle.dump(pipe, open(location,'wb'))
models.append(
    {
        "model":"random forest classifier",
        "accuracy":rfc_score,
        "location":location
    }
)

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(ytest,  ypred)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

rfc_score


# In[122]:


# svm
pipe = Pipeline([('scaler',StandardScaler()),('svm',svm.SVC(kernel='rbf'))])
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

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(ytest,  ypred)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

svm_score


# In[123]:


# knn
pipe = Pipeline([('scaler',StandardScaler()),('knn',KNeighborsClassifier(n_neighbors=5))])
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

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(ytest,  ypred)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

knn_score


# In[124]:


# lr
pipe = Pipeline([('scaler',MinMaxScaler()),('lr',LogisticRegression(random_state=42))])
pipe.fit(xtrain, ytrain)
ypred = pipe.predict(xtest)
lr_score = accuracy_score(ytest,ypred)

location = 'models/lr.pkl'
pickle.dump(pipe, open(location,'wb'))
models.append(
    {
        "model":"logistic regression",
        "accuracy":lr_score,
        "location":location
    }
)

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(ytest,  ypred)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

lr_score


# In[125]:


import json

with open("results/models.json", "w") as outfile:
    json.dump(models, outfile)


# In[37]:


import pandas as pd
import sqlite3

dataset = pd.read_csv('data/data.csv')

conn = sqlite3.connect('stroke_database.db')
c = conn.cursor()

c.execute('CREATE TABLE IF NOT EXISTS stroke (id, gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke)')
conn.commit()

dataset.to_sql('stroke', conn, if_exists='replace', index=False)

conn.close()


# In[38]:


df.columns


# In[39]:


import sqlite3
conn = sqlite3.connect('stroke_database.db')
c = conn.cursor()

c.execute('''SELECT * FROM stroke''')
print(c.fetchall())

