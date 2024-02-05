#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('iris.csv')


# In[4]:


df.head(5)


# In[5]:


df = df.drop(columns = ['Id'])


# In[6]:


df


# In[7]:


df.Species.value_counts()


# In[8]:


df.shape


# In[9]:


df["Species"].replace({"Iris-setosa": 2, "Iris-versicolor": 3, "Iris-virginica": 4}, inplace = True)


# In[15]:


df.head(5)


# In[16]:


df.tail(5)


# In[17]:


x=pd.DataFrame(df,columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]).values
y=df.Species.values.reshape(-1,1)


# In[18]:


x


# In[19]:


y


# In[20]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings


# In[21]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=42) 


# In[22]:


k=7

clf=KNeighborsClassifier(k)

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)


# In[23]:


metrics.accuracy_score(y_test,y_pred)*100


# In[ ]:




