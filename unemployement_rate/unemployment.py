#!/usr/bin/env python
# coding: utf-8

# In[200]:


import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[201]:


df_full = pd.read_csv('Unemployment in India.csv')
df_2020 = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')


# In[202]:


df_full.head()


# In[203]:


df_2020.head()


# In[132]:


df_full.info()


# In[133]:


df_2020.info()


# In[134]:


df_full.shape


# In[135]:


df_2020.shape


# In[136]:


df_full.describe()


# In[137]:


df_2020.describe()


# In[138]:


df_full.isnull().sum()


# In[139]:


df_2020.isnull().sum()


# In[140]:


df_full = df_full.dropna()


# In[141]:


df_full.isnull().sum()


# In[142]:


df_full.shape


# In[143]:


df_2020.shape


# In[144]:


df_full.columns = ['State','Date',' Frequency',' Estimated Unemployment Rate',' Estimated Employed',' Estimated Labour Participation Rate','Area']

df_2020.columns = ['State','Date',' Frequency',' Estimated Unemployment Rate',' Estimated Employed',' Estimated Labour Participation Rate','Region','Longitude','Latitude']


# In[145]:


df_full.columns


# In[146]:


df_2020.columns


# In[147]:


df_full.head()


# In[148]:


df_2020.head()


# In[149]:


df_full['State'].value_counts()


# In[150]:


df_full['State'].value_counts().idxmax()


# In[151]:


df_full['State'].value_counts().idxmin()


# In[152]:


df_2020['State'].value_counts()


# In[153]:


df_2020['State'].value_counts().idxmax()


# In[154]:


df_2020['State'].value_counts().idxmin()


# In[155]:


import datetime as dt
import calendar as cal


# In[156]:


df_full['Date'] = pd.to_datetime(df_full['Date'],dayfirst=True)
df_full['Month_int'] = df_full['Date'].dt.month
df_full['Month'] = df_full['Month_int'].apply(lambda x: cal.month_abbr[x])


# In[157]:


df_2020['Date'] = pd.to_datetime(df_2020['Date'],dayfirst=True)
df_2020['Month_int'] = df_2020['Date'].dt.month
df_2020['Month'] = df_2020['Month_int'].apply(lambda x: cal.month_abbr[x])


# In[158]:


df_full['Month'].value_counts().idxmax()


# In[159]:


df_full['Month'].value_counts().idxmin()


# In[160]:


df_2020['Month'].value_counts().idxmax()


# In[161]:


df_2020['Month'].value_counts().idxmin()


# In[162]:


df_full.head()


# In[163]:


df_2020.head()


# In[164]:


df_full.drop(columns=[' Frequency','Month_int'])


# In[165]:


df_2020.drop(columns=[' Frequency','Month_int'])


# In[166]:


df_full.head(1)


# In[167]:


df_2020.head(1)


# In[168]:


df_temp = df_full[['State',' Estimated Unemployment Rate']].groupby('State').sum().sort_values(' Estimated Unemployment Rate', ascending=False)


# In[169]:


df_temp.head()


# In[170]:


df_temp2 = df_2020[['State',' Estimated Unemployment Rate']].groupby('State').sum().sort_values(' Estimated Unemployment Rate', ascending=False)


# In[171]:


df_temp2.head()


# In[175]:


fig = plt.figure()
ax0 = fig.add_subplot(1,2,1)
df_temp[:10].plot(kind='bar',color='red',figsize=(30,5),ax=ax0)
ax0.set_title('Top 10 states with highest')
ax0.set_xlabel('State')
ax0.set_ylabel('Number of people unemployed %')


# In[176]:


fig2 = plt.figure()
ax0 = fig2.add_subplot(1,2,1)
df_temp2[:10].plot(kind='bar',color='blue',figsize=(30,5),ax=ax0)
ax0.set_title('Top 10 states with highest')
ax0.set_xlabel('State')
ax0.set_ylabel('Number of people unemployed %')


# In[177]:


df_temp3 = df_full[['Month',' Estimated Unemployment Rate']].groupby('Month').sum().sort_values(' Estimated Unemployment Rate', ascending=False)


# In[178]:


df_temp3.head()


# In[179]:


df_temp4 = df_2020[['Month',' Estimated Unemployment Rate']].groupby('Month').sum().sort_values(' Estimated Unemployment Rate', ascending=False)


# In[180]:


df_temp4.head()


# In[181]:


fig3 = plt.figure()
ax0 = fig3.add_subplot(1,2,1)
df_temp3[:12].plot(kind='bar',color='yellow',figsize=(30,5),ax=ax0)
ax0.set_title('Top 10 states with highest')
ax0.set_xlabel('State')
ax0.set_ylabel('Number of unemployment %')


# In[182]:


fig4 = plt.figure()
ax0 = fig4.add_subplot(1,2,1)
df_temp4[:12].plot(kind='bar',color='green',figsize=(30,5),ax=ax0)
ax0.set_title('Top 10 states with highest')
ax0.set_xlabel('State')
ax0.set_ylabel('Number of unemployment %')


# In[183]:


df_temp5 = df_full.groupby(['Month'])[[' Estimated Unemployment Rate',' Estimated Employed',' Estimated Labour Participation Rate']].mean()
df_temp5 = pd.DataFrame(df_temp5).reset_index()
Month_temp = df_temp5.Month
Unemp_rate_temp = df_temp5[' Estimated Unemployment Rate']
labour_ptp_rate_temp = df_temp5[' Estimated Labour Participation Rate']
fig5 = go.Figure()
fig5.add_trace(go.Bar(x=Month_temp,y=Unemp_rate_temp,name='unemployment rate'))
fig5.add_trace(go.Bar(x=Month_temp,y=labour_ptp_rate_temp,name='labour participation rate'))
fig5.show()


# In[184]:


df_temp6 = df_2020.groupby(['Month'])[[' Estimated Unemployment Rate',' Estimated Employed',' Estimated Labour Participation Rate']].mean()
df_temp6 = pd.DataFrame(df_temp6).reset_index()
Month_temp2 = df_temp6.Month
Unemp_rate_temp2 = df_temp6[' Estimated Unemployment Rate']
labour_ptp_rate_temp2 = df_temp6[' Estimated Labour Participation Rate']
fig6 = go.Figure()
fig6.add_trace(go.Bar(x=Month_temp2,y=Unemp_rate_temp2,name='unemployment rate'))
fig6.add_trace(go.Bar(x=Month_temp2,y=labour_ptp_rate_temp2,name='labour participation rate'))
fig6.show()


# In[185]:


df_temp7 = df_full[['State',' Estimated Employed']].groupby('State').sum().sort_values(' Estimated Employed', ascending=False)
df_temp7


# In[187]:


df_temp7 = df_full[['State',' Estimated Employed']].groupby('State').sum().sort_values(' Estimated Employed',ascending=False)
fig7=plt.figure()
ax1=fig7.add_subplot(1,2,2)
df_temp7[:10].plot(kind='bar',color='green',figsize=(15,6),ax=ax1)
ax1.set_title(' Estimated Employed people in each state')
ax1.set_title('State')
ax1.set_ylabel('Number of estimated employed ')
fig7.show()


# In[188]:


df_temp8 = df_2020[['State',' Estimated Employed']].groupby('State').sum().sort_values(' Estimated Employed', ascending=False)
df_temp8


# In[189]:


df_temp8 = df_2020[['State',' Estimated Employed']].groupby('State').sum().sort_values(' Estimated Employed',ascending=False)
fig8=plt.figure()
ax1=fig8.add_subplot(1,2,2)
df_temp8[:10].plot(kind='bar',color='green',figsize=(15,6),ax=ax1)
ax1.set_title(' Estimated Employed people in each state')
ax1.set_title('State')
ax1.set_ylabel('Number of estimated employed ')
fig8.show()


# In[191]:


df_temp9 = df_full[['Month',' Estimated Employed']].groupby('Month').sum().sort_values(' Estimated Employed', ascending=False)
df_temp9


# In[192]:


df_temp9 = df_full[['Month',' Estimated Employed']].groupby('Month').sum().sort_values(' Estimated Employed',ascending=False)
fig9=plt.figure()
ax1=fig9.add_subplot(1,2,2)
df_temp9[:10].plot(kind='bar',color='green',figsize=(15,6),ax=ax1)
ax1.set_title(' Estimated Employed people in each state')
ax1.set_title('Month')
ax1.set_ylabel('Number of estimated employed ')
fig9.show()


# In[193]:


df_temp10 = df_2020[['Month',' Estimated Employed']].groupby('Month').sum().sort_values(' Estimated Employed', ascending=False)
df_temp10


# In[195]:


df_temp10 = df_2020[['Month',' Estimated Employed']].groupby('Month').sum().sort_values(' Estimated Employed',ascending=False)
fig10=plt.figure()
ax1=fig10.add_subplot(1,2,2)
df_temp10[:10].plot(kind='bar',color='green',figsize=(15,6),ax=ax1)
ax1.set_title(' Estimated Employed people in each state')
ax1.set_title('Month')
ax1.set_ylabel('Number of estimated employed ')
fig10.show()


# In[198]:


fig11 = px.bar(df_full, x='State', y=' Estimated Unemployment Rate', animation_frame='Month', color='State', title='Unemployment rate month wise')
fig11.update_layout(xaxis={'categoryorder': 'total ascending'})
fig11.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 2000
fig11.show()


# In[ ]:





# In[199]:


fig12 = px.bar(df_2020, x='State', y=' Estimated Unemployment Rate', animation_frame='Month', color='State', title='Unemployment rate month wise')
fig12.update_layout(xaxis={'categoryorder': 'total ascending'}) 
fig12.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 2000
fig12.show()

