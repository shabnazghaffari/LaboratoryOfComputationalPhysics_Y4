#!/usr/bin/env python
# coding: utf-8

# The exercise goal is to predict the maximum wind speed occurring every 50 years even if no measure exists for such a period. The available data are only measured over 21 years at the Sprogø meteorological station located in Denmark.
# 
# The annual maxima are supposed to fit a normal probability density function. However such function is not going to be estimated because it gives a probability from a wind speed maxima. Finding the maximum wind speed occurring every 50 years requires the opposite approach, the result needs to be found from a defined probability. That is the quantile function role and the exercise goal will be to find it. In the current model, it is supposed that the maximum wind speed occurring every 50 years is defined as the upper 2% quantile.

# By definition, the quantile function is the inverse of the cumulative distribution function. The latter describes the probability distribution of an annual maxima. In the exercise, the cumulative probability pi  for a given year i is defined as pi=i/N+1  with N=21, the number of measured years. Thus it will be possible to calculate the cumulative probability of every measured wind speed maxima. From those experimental points, the scipy.interpolate module will be very useful for fitting the quantile function. Finally the 50 years maxima is going to be evaluated from the cumulative probability of the 2% quantile.
# 
# 

# Practically, load the dataset:
# 
# import numpy as np
# max_speeds = np.load('max-speeds.npy')
# years_nb = max_speeds.shape[0]
# Compute then the cumulative probability pi (cprob) and sort the maximum speeds from the data. Use then the UnivariateSpline from scipy.interpolate to define a quantile function and thus estimate the probabilities.
# 
# In the current model, the maximum wind speed occurring every 50 years is defined as the upper 2% quantile. As a result, the cumulative probability value will be:
# 
# fifty_prob = 1. - 0.02
# So the storm wind speed occurring every 50 years can be guessed as:
# 
# fifty_wind = quantile_func(fifty_prob)

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


max_speeds = np.load('max-speeds.npy')
years_nb = max_speeds.shape[0]


# In[3]:


from scipy.interpolate import UnivariateSpline


# In[4]:


print(max_speeds)
print(years_nb)
np.max(max_speeds)


# In[5]:


max_speeds = np.sort(max_speeds)
cprob=np.zeros(years_nb)
for i in range(0,years_nb):
    cprob[i] =  i/(years_nb+1)

quantile_func = UnivariateSpline(x=cprob, y=max_speeds)


fifty_prob = 1 - 0.02
fifty_wind = quantile_func(fifty_prob)
print("Maximum wind speed occurring every 50 years:", fifty_wind)


# Curve fitting of temperature in Alaska
# 
# The temperature extremes in Alaska for each month, starting in January, are given by (in degrees Celcius):
# 
# max: 17, 19, 21, 28, 33, 38, 37, 37, 31, 23, 19, 18
# 
# min: -62, -59, -56, -46, -32, -18, -9, -13, -25, -46, -52, -58

# Plot these temperature extremes.
# Define a function that can describe min and max temperatures.
# Fit this function to the data with scipy.optimize.curve_fit().
# Plot the result. Is the fit reasonable? If not, why?
# Is the time offset for min and max temperatures the same within the fit accuracy?

# In[6]:


import scipy.optimize as scp


# In[7]:


maxT=[ 17, 19, 21, 28, 33, 38, 37, 37, 31, 23, 19, 18]
minT=[-62, -59, -56, -46, -32, -18, -9, -13, -25, -46, -52, -58]
ll=len(maxT)
month=np.zeros(ll)
for i in range(0,ll):
    month[i] =  i
    
plt.plot(month,minT,"bo")    
plt.plot(month,maxT,"ro")

def f(x,omega,phi,a,b):
    return b*np.sin(x*omega+phi)+a


val,err=scp.curve_fit(f,month,maxT)
val1,err1=scp.curve_fit(f,month,minT,p0=val)
plt.plot(month,f(month,val1[0],val1[1],val1[2],val1[3]))


plt.plot(f(month,val[0],val[1],val[2],val[3]))
plt.grid(True)
plt.xlabel("month")
plt.ylabel("temperature")
plt.show()

dif1=(val1[1]-val[1])/val1[1]
dif=(val1[1]-val[1])/val[1]
print(np.sqrt(err[1,1]))
print(np.sqrt(err1[1,1]))


# 2D minimization of a six-hump camelback function
# 
#  
# has multiple global and local minima. Find the global minima of this function.
# 
# Hints:
# 
# Variables can be restricted to  -2<x<2 and -1<y<1 .
# Use numpy.meshgrid() and pylab.imshow() to find visually the regions.
# Use scipy.optimize.minimize(), optionally trying out several of its methods.
# How many global minima are there, and what is the function value at those points? What happens for an initial guess of (x,y)=(0,0) ?

# In[8]:


X=np.linspace(-2,2,num=50)
Y=np.linspace(-1,1,num=50)
l=np.meshgrid(X,Y)

def func(z):
    return (4-2.1*z[0]**2+(z[0]**4)/3)*z[0]**2 +z[0]*z[1] + (4*z[1]**2 -4)*z[1]**2
plt.contourf(Y,X,func(l),levels=15)
plt.show()
k=scp.minimize(func,x0=[0,0])


for a in range(-2,3):
    for b in range(-1,2):
        k=scp.minimize(func,x0=[a,b])
        print(k.x,k.fun)


# FFT of a simple dataset
# 
# Performe a periodicity analysis on the lynxs-hares population

# In[9]:


data = np.loadtxt("populations.txt")
years = data[:, 0]
populations = data[:, 1:]
hares = data[:,1]
lynxes = data[:,2]


from scipy import fftpack
from matplotlib import pyplot as plt

ft_hares = fftpack.fft(hares, axis=0)
ft_lynxes = fftpack.fft(lynxes, axis=0)
freq_hares = fftpack.fftfreq(hares.shape[0], years[1] - years[0])
freq_lynxes = fftpack.fftfreq(lynxes.shape[0], years[1] - years[0])
print(freq_hares)
periods_hares = 1 / freq_hares
periods_lynxes = 1 / freq_lynxes
p=periods_hares[1:]
q=periods_lynxes[1:]
print("the period of hares population is: ", abs(p).argmin()+1)
print("the period of lynxes population is: ", abs(q).argmin()+1)


# FFT of an image
# 
# Examine the provided image moonlanding.png, which is heavily contaminated with periodic noise. In this exercise, we aim to clean up the noise using the Fast Fourier Transform.
# Load the image using pylab.imread().
# Find and use the 2-D FFT function in scipy.fftpack, and plot the spectrum (Fourier transform of) the image. Do you have any trouble visualising the spectrum? If so, why?
# The spectrum consists of high and low frequency components. The noise is contained in the high-frequency part of the spectrum, so set some of those components to zero (use array slicing).
# Apply the inverse Fourier transform to see the resulting image.

# In[10]:


im = plt.imread('moonlanding.png')


# In[11]:


from scipy import fftpack
from matplotlib.colors import LogNorm

im_fft = fftpack.fft2(im)


def plot_spectrum(im_fft):
    plt.imshow(im_fft.real, norm=LogNorm(vmin=10))


keep_fraction = 0.15

im_fft2 = im_fft.copy()

rows, columns = im_fft2.shape

im_fft2[int(rows*keep_fraction):] = 0
im_fft2[:, int(columns*keep_fraction):] = 0


# In[12]:


im_new = fftpack.ifft2(im_fft2).real

fig = plt.figure( )
plt.imshow(im_new)
plt.title('Reconstructed moonlanding')
plt.show()
plt.imshow(im)

plt.title('Original moonlanding')


# In[ ]:





# In[ ]:




