#used jupyter notebooks and exported as .py file as matplotlib was unaccessible from spyder

#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[58]:


df = pd.read_csv('iris.csv')
df = df.drop(columns=['Id'])


# In[64]:


#scatterplot
colors = ['red','orange','blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']

for i in range (3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'], c=colors[i],label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[63]:


for i in range (3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'], c=colors[i],label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[62]:


for i in range (3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'],x['PetalWidthCm'], c=colors[i],label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[61]:


for i in range (3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'],x['PetalWidthCm'], c=colors[i],label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# In[ ]:


df1 = df.drop(columns=['Species'])
corr = df1.corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, ax=ax)
plt.show()


# In[52]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
df.head()


# In[103]:


X1 = df.drop(columns=['Species'])
Y1 = df['Species']
xtrain1,xtest1,ytrain1,ytest1 = train_test_split(X1,Y1,test_size=0.30)


# In[108]:


model1i = LogisticRegression()
model1i.fit(xtrain1,ytrain1)
print("Accuracy:",model1i.score(xtest1,ytest1)*100)


# In[110]:


model1ii =  KNeighborsClassifier()
model1ii.fit(xtrain1,ytrain1)
print("Accuracy:",model1ii.score(xtest1,ytest1)*100)


# In[116]:


model1iii = DecisionTreeClassifier()
model1iii.fit(xtest1,ytest1)
print("Accuracy:",model1iii.score(xtest1,ytest1)*100)


# In[118]:


X2 = df.drop(columns=['Species'])
Y2 = df['Species']
xtrain2,xtest2,ytrain2,ytest2 = train_test_split(X2,Y2,test_size=0.50)


# In[119]:


model2i = LogisticRegression()
model2i.fit(xtrain2,ytrain2)
print("Accuracy:",model2i.score(xtest2,ytest2)*100)


# In[120]:


model2ii = KNeighborsClassifier()
model2ii.fit(xtrain2,ytrain2)
print("Accuracy:",model2ii.score(xtest2,ytest2)*100)


# In[121]:


model2iii = DecisionTreeClassifier()
model2iii.fit(xtest2,ytest2)
print("Accuracy:",model2iii.score(xtest2,ytest2)*100)


# In[122]:


X3 = df.drop(columns=['Species'])
Y3 = df['Species']
xtrain3,xtest3,ytrain3,ytest3 = train_test_split(X3,Y3,test_size=0.70)


# In[123]:


model3i = LogisticRegression()
model3i.fit(xtrain3,ytrain3)
print("Accuracy:",model3i.score(xtest3,ytest3)*100)


# In[124]:


model3ii = KNeighborsClassifier()
model3ii.fit(xtrain3,ytrain3)
print("Accuracy:",model3ii.score(xtest3,ytest3)*100)


# In[125]:


model3iii = DecisionTreeClassifier()
model3iii.fit(xtest3,ytest3)
print("Accuracy:",model3iii.score(xtest3,ytest3)*100)


# In[ ]:




