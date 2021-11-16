#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#1.1
for i in range(0,100): 
  if i%3==0: 
    print("Mickey") 
  if i%5==0: 
    print("Mouse") 
  if i%3==0 and i%5==0: 
    print("MickeyMouse") 
  else: 
      print(i)


# In[ ]:


#1.2
for i in range(0,100): 
  if i%3==0: 
    b=["Mickey" , "Donald"] 
    c=tuple(b) 
    print(c) 
  if i%5==0: 
    d=[] 
    print("Mouse", "Duck") 
  if i%3==0 and i%5==0: 
    print("mickeymouse") 
  else: 
      print(i)


# In[ ]:


#2.1
x=int(input()) 
y=int(input()) 
def swap(x,y): 
    a=x 
    b=y
    y=a
    x=b
    return(x,y) 
swap(x,y)


# In[1]:


a = [2, 4, 10, 6, 8, 4, 10, 12, 18, 19, 25, 5, 7]
print("Before normalization : ", a)

amin, amax = min(a), max(a)
for i, val in enumerate(a):
    a[i] = (val-amin) / (amax-amin)

print("After normalization : ", a)


# In[ ]:


#4
count = {}
check_string="Write a program that prints the numbers from 1 to 100. But for multiples of three print Mickey instead of the number and for the multiples of five print Mouse. For numbers which are multiples of both three and five print MickeyMouse"
for s in check_string:
        if s in count:
             count[s] += 1
        else:
             count[s] = 1

for key in count:
    if count[key] > 1:
                print (key, count[key])


# In[ ]:


#5
check_string= [36, 45, 58, 3, 74, 96, 64, 45, 31, 10, 24, 19, 33, 86, 99, 18, 63, 70, 85,
 85, 63, 47, 56, 42, 70, 84, 88, 55, 20, 54, 8, 56, 51, 79, 81, 57, 37, 91,
 1, 84, 84, 36, 66, 9, 89, 50, 42, 91, 50, 95, 90, 98, 39, 16, 82, 31, 92, 41,
 45, 30, 66, 70, 34, 85, 94, 5, 3, 36, 72, 91, 84, 34, 87, 75, 53, 51, 20, 89, 51, 20]
count = {}

for s in check_string:
        if s in count:
             count[s] += 1
        else:
             count[s] = 1

for key in count:
    if count[key] > 1:
                print (key, count[key])


# In[ ]:


#6
def square(num):
    return (num*num)

# User defind method to find cube

def cube(num):
    return (num*num*num)

def function6th(num):
    return cube(num)*square(num)*num

function6th(2)
print(function6th(2))


# In[ ]:


#7
for i in range(0,10):
    print(i*i*i)




#7.2
cube = [i*i*i for i in range(10)]
cube


# In[ ]:


#8
import numpy as np
a=(3,4,5)
for i in range(0,21):
    b=list(a)
    c=np.array(b)
    print(c*i)


# In[2]:



#9

a = [2, 4, 10, 6, 8, 4, 10, 12, 18, 19, 25, 5, 7]
print("Before normalization : ", a)

amin, amax = min(a), max(a)
for i, val in enumerate(a):
    a[i] = (val-amin) / (amax-amin)

print("After normalization : ", a)


# In[ ]:




