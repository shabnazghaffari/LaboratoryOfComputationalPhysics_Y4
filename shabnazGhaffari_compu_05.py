#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt


# In[5]:


filename = r"C:\Users\Shabnaz\Downloads\data_000637.txt"
n_rows = np.random.randint(low=10000, high=1310719)
print("rows: ", n_rows)
data = pd.read_csv(filename, nrows=n_rows)

# create dataframe
df = pd.DataFrame(data)
print(df)


# In[6]:


#2
x_BX = np.max(data['BX_COUNTER'])+1
# the min value is 0 so the difference is equal to the max value
print("x value: ", x_BX)


# In[7]:


#3
data = pd.read_csv(filename)
last_index = len(data)-1

time_ns = np.array((data['ORBIT_CNT']*(x_BX)*25 + data['BX_COUNTER']*25 + data['TDC_MEAS']*25/30))

estimated_time = time_ns[last_index] - time_ns[0]
print("Estiamted total time in ns: ", estimated_time)
print("Estiamted total time in s: ", estimated_time*(10**(-9)))


# In[8]:


#4
import time
stattime=0
endtime=0
starttime=time.time_ns()
df=pd.read_csv( r"C:\Users\Shabnaz\Downloads\data_000637.txt",delimiter=',')
endtime=time.time_ns()
print(endtime-starttime)


# In[ ]:


#5
import numpy as np
for i in range(len(df['HEAD'])):
    df['HEAD'][i]=np.random.randint(2)
print(df['HEAD'].iloc[0:12000])


# In[ ]:


#6
df['newDataFrame']=df['HEAD']==1
print(df)


# In[ ]:


#7
import matplotlib.pyplot as plt
for i in range(20):
    plt.bar(i,df['FPGA'][i])
plt.figure()
for i in range(20):
    plt.bar(i,df['TDC_CHANNEL'][i])


# In[ ]:


#8
new_df=pd.value_counts(df['TDC_CHANNEL'])
new_df.iloc[0:3]


# In[ ]:


#9
print(len(df[df['TDC_CHANNEL'] == 139]))
print(df['TDC_CHANNEL'].unique())


# In[ ]:





# In[ ]:




