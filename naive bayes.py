#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd 
import numpy as np 


# In[2]:


data = pd.read_csv("Breast_cancer_data.csv")
data.head(10)


# In[3]:


X=data.drop(['diagnosis'],axis=1).values
y=data['diagnosis'].values


# In[57]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.35,random_state=12)


# In[58]:


# Create a Gaussian Classifier
gnb=GaussianNB()
y_pred=gnb.fit(X_train,y_train)
# Predict Output
pred= gnb.predict(X_test) # 0:Overcast, 2:Mild
print ("Predicted Value:", pred)


# In[59]:


print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))


# In[60]:


import sklearn.metrics as metrics
accuracy = metrics.accuracy_score(y_test, pred)
print(accuracy)


# In[ ]:




