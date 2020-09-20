#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install xgboost


# In[2]:


from numpy import loadtxt
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[3]:


# load data
dataset = loadtxt('RF3.csv', delimiter=",")


# In[4]:


# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]


# In[5]:


# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# In[6]:


# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)


# In[7]:


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[8]:


# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[17]:


import os
os.environ['PATH'] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'


# In[18]:


# plot decision tree
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
# load data
dataset = loadtxt('RF3.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
y = dataset[:,8]
# fit model no training data
model = XGBClassifier()
model.fit(X, y)
# plot single tree
plot_tree(model)
plt.show()


# In[19]:


plot_tree(model, num_trees=4)
plot_tree(model, num_trees=0, rankdir='LR')


# In[ ]:




