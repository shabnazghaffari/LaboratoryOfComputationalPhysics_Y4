#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
get_ipython().run_line_magic('matplotlib', 'inline')


import numpy as np


# In[2]:


#1
n_bins = 20
sigma = 1
mu = 2
x = sigma**(2)*np.random.randn(1000)+mu


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
y, bins, patches = ax1.hist(x, 50)
bincenters = 0.5*(bins[1:]+bins[:-1])
ax1.set_ylabel("")
ax1.errorbar(bincenters, y, yerr=np.sqrt(y))
#fig.tight_layout()

gaussians = []
mixture = None
space = np.linspace(mu-4*sigma, mu+4*sigma, 100)
for i in range(len(x)):
    gaussian = norm(loc=x[i],scale=1.06*sigma**(2)*(len(x)**(-0.2)))
    gaussians.append(gaussian)
    ax2.plot(space, gaussian.pdf(space))
    try:
        mixture +=gaussian.pdf(space)
    except:
        mixture =gaussian.pdf(space)
    
ax2.plot(space, mixture*(scipy.integrate.trapz(y)/scipy.integrate.trapz(mixture)))


# In[6]:


#2
get_ipython().system(' wget https://www.dropbox.com/s/u4y3k4kk5tc7j46/two_categories_scatter_plot.png')
from IPython.display import Image
Image('two_categories_scatter_plot.png')


# In[7]:


import pandas as pd 
import seaborn as sns

mean = [(1, 2),(6, 12),(-6, -12)]
cov = [[[1, 0], [0, 1]], [[1, 0], [0, 1]],[[1, 0], [0, 1]]]
def gendata(mean, cov):
    data = np.random.multivariate_normal(mean[0], cov[0],200)
    df2 = pd.DataFrame(data, columns=["x","y"])
    df2['label'] = np.zeros(len(df2))
    df = pd.DataFrame(df2,columns=["x","y","label"])
    dataframes = []
    for i in range(1,len(mean)):
        data = np.random.multivariate_normal(mean[i], cov[i],200)
        df3 = pd.DataFrame(data, columns=["x","y"])
        df3['label'] = i*np.ones(len(df3))
        dataframes.append(df3)
        
    lista = []
    lista.append(df)
    for i in range(len(dataframes)):
        lista.append(dataframes[i])
    dataframe = pd.concat(lista,ignore_index=True)
    return dataframe
df = gendata(mean,cov).astype({'label': 'int32'})
print(df['label'].unique())
sns.relplot(x="x", y="y", hue="label", data=df);


# In[8]:


#3
get_ipython().system('wget https://www.dropbox.com/s/hgnvyj9abatk8g6/residuals_261.npy')


# In[9]:


from scipy import stats

numpy_data = np.load("residuals_261.npy",allow_pickle=True)
df = pd.DataFrame(numpy_data.item())
df = df.drop(df.index[np.where(np.abs(df['residuals'])>2)])
df


# In[10]:


slope, intercept, r_value, p_value, std_err = stats.linregress(df['residuals'],df['distances'])
print("slope: %f    intercept: %f" % (slope, intercept))
plt.plot(df['residuals'],df['distances'],'o', label='original data')
plt.plot(df['residuals'], intercept + slope*df['residuals'], 'r', label='fitted line')
plt.legend()
plt.show()
g = sns.jointplot("residuals", "distances", data=df, kind="reg")
nrbins = 20
_,bins = np.histogram(df['distances'],nrbins, range=[0,20])
x = 0.5*(bins[1:]+bins[:-1])
y = []
erry = []
bound = x[0]

for i in x:
    y.append(np.mean(df[(df['distances']<i+bound) & (df['distances']>i-bound)]['residuals']))
    erry.append(np.std(df[(df['distances']<i+bound) & (df['distances']>i-bound)]['residuals']))
plt.errorbar(y, x,0,erry, fmt='r', capsize=2)


# In[ ]:





# In[ ]:





# In[ ]:




