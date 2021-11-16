#!/usr/bin/env python
# coding: utf-8

# In[1]:


#3.1
import numpy as np
m = np.arange(12).reshape((3,4))
for i in range(len(m)):
                print("rows are like=",m[i])


# In[2]:


#3.1.2
import numpy as np
m = np.arange(12).reshape((3,4))
for i in range(len(m[0])):
                     print("colmns are like=",m[:,i])


# In[3]:


#3.2
import numpy as np
u = np.array([1,3,5,7])
v = np.array([2,4,6,8])
print(u[:, np.newaxis] + v)


# In[1]:


# outer list comprehension
out2 = np.array([i*j for i in u for j in v])
out2 = out2.reshape(4,4)

# outer broadcasting
u = u[np.newaxis]
out3 = (u.T)*v

print(out1)
print(out2)
print(out3)
print(u,v)


# In[4]:


#3.3

matrix = np.random.random(60).reshape(10, 6)
print("Original matrix:\n", matrix)


matrix[matrix < 0.1] = 0
print("\n Modified matrix:\n", matrix)


# In[13]:


#3.4
import numpy as np
a=np.linspace(0, 2*np.pi,100)
print(a)


# In[14]:


#3,4.1
list1 = slice(0,10)
list2 = slice(10,20)
list3 = slice(20,30)
list4 = slice(30,40)
list5 = slice(40,50)
list6 = slice(50,60)
list7 = slice(60,70)
list8 = slice(70,80)
list9 = slice(80,90)
list10 = slice(90,100)


    
print(a[list1])

print(a[list2])

print(a[list3])

print(a[list4])

print(a[list5])

print(a[list6])

print(a[list7])

print(a[list8])

print(a[list9])

print(a[list10])


# In[7]:


#3.4.2
b=list(a)
b.reverse()
b


# In[15]:


#3.4.3
import math
for i in a:
     if (math.sin(i))-(math.cos(i)) <0.1:
             print(i)
            
  
        


# In[17]:


#3.4.4
import matplotlib.pyplot as plt
diff = a[abs(np.sin(a)-np.cos(a))<0.1]       
plt.plot(a, np.sin(a), label="Sin")
plt.plot(a, np.cos(a), label="Cos")
plt.plot(diff, np.cos(diff), 'ro', label="Close points")
plt.plot(diff, np.sin(diff), 'ro')
plt.legend()    

 




# In[18]:


#5
a = np.arange(1,11)
matrix = a.reshape(10,1) * a
trace = matrix.trace()
antodiagonal = np.diagonal(np.fliplr(matrix))
diagonal_1off = np.diagonal(matrix, 1)
print(matrix)
print("Trace:", trace)
print("Anto-diagonal:", antodiagonal)
print("Diagonal offset by +1:", diagonal_1off)



# In[ ]:





# In[ ]:





# In[ ]:


#6
city =np.array(['Chicago', 'Springfield', 'St.Louis', 'Tulsa', 'Oklahoma City', 'Amarillo', 'Santa Fe', 'Albuquerque', 'Flagstaff', 'Los Angeles'])
distgrid = np.array([0,198,303,736,871,1175,1475,1544,1913,2448])
distgrid = distgrid[:,np.newaxis]

recipr_dist=abs(distgrid-distgrid.T)
for i in range(distgrid.shape[0]):
    print('The %i coloumn rappresents -->'%i, city[i], "<--  compared with the other cities' distances (in miles)\n")
print(recipr_dist, '\n\n')



print('In km the previous matrix became:\n\n',np.abs(distgrid-distgrid.T)*1.61)


# In[ ]:


#7
import numpy as np
def prime(n) :
    array = np.arange(1, 100)
    mask = np.array([True for i in array])
    for i in array:
        count = 0
        for j in range(2, (i//2 + 1)):
            if(i % j == 0):
                count = count + 1
                mask=True
                break

        if (count == 0 and i!=1):
            print(i)

    primenumbers = array[mask]
    return(prime)
  
print(prime(100))


# In[ ]:


#8
import numpy.random as npr
import matplotlib.pyplot as plt
npr.seed(1234) 
walkers = np.array([npr.choice([-1,1]) for i in range(0,200000)]).reshape(1000,200)


walksum = np.array([npr.choice([-1,1]) for i in range(0,200000)]).reshape(1000,200)
for i in range(1,200) : 
    walksum[:,i] += walksum[:,i-1] 

    
means = np.mean(walksum**2, axis = 0)
get_ipython().run_line_magic('matplotlib', 'inline')
t = np.linspace(0,200,200)

plt.plot(t,np.sqrt(means))


# In[ ]:


#9
data = np.loadtxt("./populations.txt")
#print(data)
year = data[:, 0]
hares = data[:, 1]
lynxes = data[:, 2]
carrots = data[:, 3]
#plt.axes([0.2, 0.1, 0.5, 0.8])
plt.xticks(year[::2])
plt.plot(year, hares, label="hares")
plt.plot(year, lynxes, label="lynxes")
plt.plot(year, carrots, label="carrots")
plt.legend()

print("\n***HARES***")
for function in ['sum','min','argmin','mean','median','std']:
    print (function, getattr(np, function)(hares))
print("\n***LYNXES***")
for function in ['sum','min','argmin','mean','median','std']:
    print (function, getattr(np, function)(lynxes))
print("\n***CARROTS***")
for function in ['sum','min','argmin','mean','median','std']:
    print (function, getattr(np, function)(carrots))


# In[ ]:




