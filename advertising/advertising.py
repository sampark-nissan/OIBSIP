#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


df = pd.read_csv('Advertising.csv')


# In[26]:


df.head()


# In[27]:


df.tail()


# In[28]:


df = df.drop(columns=['Unnamed: 0'])


# In[29]:


df.shape



# In[30]:


df.info()


# In[31]:


df.describe()


# In[32]:


a = df.iloc[:,0:-1]


# In[33]:


a


# In[34]:


b = df.iloc[:,-1]


# In[35]:


b


# In[36]:


a.iloc[:,0]


# In[37]:


df.head()


# In[38]:


df.corr()


# In[39]:


import seaborn as sns


# In[40]:


sns.pairplot(data=df)
plt.show()


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


a_train, a_test, b_train, b_test = train_test_split(a,b,test_size=0.2, random_state=49)


# In[43]:


a_train


# In[44]:


a_test


# In[45]:


b_train


# In[46]:


b_test


# In[47]:


a_train = a_train.astype(int)
a_test = a_test.astype(int)
b_train = b_train.astype(int)
b_test = b_test.astype(int)


# In[51]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[56]:


a_train_scaled = sc.fit_transform(a_train)
a_test_scaled = sc.fit_transform(a_test)


# In[57]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[58]:


lr.fit(a_train_scaled,b_train)


# In[59]:


b_pred = lr.predict(a_test_scaled)


# In[60]:


b_pred


# In[61]:


b_test


# In[62]:


import matplotlib.pyplot as plt


# In[68]:


plt.scatter(b_test, b_pred, c='r')


# In[69]:


plt.scatter(b_test,b_pred,c='r',label='Predicted',marker='x')
plt.scatter(b_test,b_test,c='b',label='True',marker='o')
plt.xlabel('true valuess')
plt.ylabel('predicted values')
plt.title('true vs predicted')
plt.legend()
plt.show()


# In[70]:


from sklearn.metrics import mean_squared_error
m_sq_err = mean_squared_error(b_test,b_pred)
m_sq_err


# In[72]:


from sklearn.metrics import mean_absolute_error
m_abs_err = mean_absolute_error(b_test,b_pred)
m_abs_err


# In[73]:


from sklearn.metrics import r2_score
r2 = r2_score(b_test,b_pred)
r2


# In[74]:


#bottom two regression models show more accuracy


# In[ ]:




