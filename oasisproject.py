#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Reading the iris dataset
df = pd.read_csv(r"D:\oasis project\IRIS.csv")


# Displaying the dataset
print(df)

# Checking the information of the dataset
print(df.info())

df = df.drop(columns="Id")


# Checking for any null values
print(df.isnull().sum())


# Checking the count of each species in the dataset
print(df['Species'].value_counts())

# Separating the features and target variable
x = df.iloc[:,:4]
y = df.iloc[:,4]

# Visualizing the count of each species using a countplot
sns.countplot(df['Species']);


# Splitting the dataset into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Printing the shapes of training and testing data
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:





# In[ ]:





# In[ ]:




