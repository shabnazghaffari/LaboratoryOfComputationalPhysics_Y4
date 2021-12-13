#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
#import numpy.random as nprd
from scipy import linalg as la
import matplotlib.pyplot as plt


# In[2]:


N = 1000
x1 = np.random.normal(0,1,N)
x2 = x1 + np.random.normal(0, 3, N)
x3 = 2*x1 + x2

M = np.array([x1,x2,x3])
print(M)

cov = np.cov(M)
print(cov)

Valu, Vctr = la.eig(cov)
print(Vctr)
print(Valu)


# In[3]:


A, spectrum, Vt = la.svd(cov)

print("shapes:", A.shape,  spectrum.shape, Vt.shape)

#print (spectrum,'\n')
#print (A,'\n')
print (Vt,'\n')

D = np.zeros((3, 3))
for i in range(min(3, 3)):
    D[i, i] = spectrum[i]
    
SVD = np.dot(A, np.dot(D, Vt))
np.allclose(SVD, cov) 

l_SVD, V_SVD = la.eig(SVD)
print(l_SVD)


# In[4]:


np.allclose(l_SVD, Valu) 
np.allclose(V_SVD, Vctr)


# In[5]:


Lambda=np.diag(Valu)
#print (Lambda)
#print ("cov.trace():", np.cov(cov).trace())
print ("Lambda.trace():", Lambda.trace())

Lambda = Lambda/Lambda.trace()

l_L, V_L = la.eig(Lambda)

l_L = np.sort(l_L,)
print( l_L.argmax() )
print( l_L.max() )

variability = l_L[1] + l_L[2]
print(variability)


# In[6]:


plt.scatter(M[0,:], M[2,:], alpha=0.2)

l, V = la.eig(cov)

for li, vi in zip(l, V.T):
    plt.plot([0, li*vi[0]], [0, li*vi[2]], '-',color="orange", lw=2)
plt.title('Eigenvectors of covariance matrix scaled by eigenvalue.');


# In[7]:


M_r = np.dot(V.T, M)

plt.scatter(M_r[0,:], M_r[2,:], alpha=0.2)

for li, vi in zip(Valu, np.diag([1]*2)):
    plt.plot([0, li*vi[0]], [0, li*vi[1]], 'r-', lw=2)


# In[8]:


l, V = la.eig(cov)
M_r = np.dot(V.T, M)

fig = plt.figure(figsize = (18,12))

plt.subplot(2,3,1)
plt.scatter(M[0,:], M[1,:], alpha=0.2)

plt.subplot(2,3,2)
plt.scatter(M[0,:], M[2,:], alpha=0.2)

plt.subplot(2,3,3)
plt.scatter(M[1,:], M[2,:], alpha=0.2)

plt.subplot(2,3,4)
plt.scatter(M_r[0,:], M_r[1,:], alpha=0.2)

plt.subplot(2,3,5)
plt.scatter(M_r[0,:], M_r[2,:], alpha=0.2)

plt.subplot(2,3,6)
plt.scatter(M_r[1,:], M_r[2,:] ,alpha=0.2)


# In[9]:


import numpy.random as npr
from scipy import linalg as la


N = 1000

x_1 = npr.normal(loc = 0, scale = 1, size = N)

x_2 = x_1 + npr.normal(loc = 0, scale = 3, size = N)

x_3 = 2*x_1 + x_2

#ADD NOISE
noise = np.array( [npr.normal(loc = 0, scale = 1/50, size = N) for i in range(10)])

noise = np.sum(noise, axis=0)

x = np.array([x_1,x_2,x_3]) + noise

#PCA
cov = np.cov(x)
l, V = la.eig(cov)

Lambda=np.diag(l)
print ("Lambda.trace():", Lambda.trace())

Lambda = Lambda/Lambda.trace()

l_L, V_L = la.eig(Lambda)

l_L = np.sort(l_L,)
print( l_L.argmax() )
print( l_L.max() )

variability = l_L[1] + l_L[2]
print(variability)


# In[10]:


plt.scatter(x[0,:], x[2,:], alpha=0.2)

l, V = la.eig(cov)

for li, vi in zip(l, V.T):
    plt.plot([0, li*vi[0]], [0, li*vi[2]], 'r-', lw=2)
plt.title('Eigenvectors of covariance matrix scaled by eigenvalue.');

#rotation
x_r = np.dot(V.T, x)

plt.scatter(x_r[0,:], x_r[2,:], alpha=0.2)

for li, vi in zip(l, np.diag([1]*2)):
    plt.plot([0, li*vi[0]], [0, li*vi[1]], 'r-', lw=2)
    
    
fig = plt.figure(figsize = (18,12))

plt.subplot(2,3,1)
plt.scatter(x[0,:], x[1,:], alpha=0.2)

plt.subplot(2,3,2)
plt.scatter(x[0,:], x[2,:], alpha=0.2)

plt.subplot(2,3,3)
plt.scatter(x[1,:], x[2,:], alpha=0.2)
plt.subplot(2,3,4)
plt.scatter(x_r[0,:], x_r[1,:], alpha=0.2)

plt.subplot(2,3,5)
plt.scatter(x_r[0,:], x_r[2,:], alpha=0.2)

plt.subplot(2,3,6)
plt.scatter(x_r[1,:], x_r[2,:], alpha=0.2)

plt.legend()


# In[ ]:




