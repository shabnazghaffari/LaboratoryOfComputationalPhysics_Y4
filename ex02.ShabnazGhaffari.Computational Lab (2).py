#!/usr/bin/env python
# coding: utf-8

# In[35]:


#1
def conversion(number, base_1, base_2):
    if base_1 == 10:
        if base_2 == 2:
            return bin(number)
        if base_2 == 16:
            return hex(number)
    else:
        s_n = str(number)
        if base_2 == 10:
                return int(s_n,base_1)
        else:
            if base_1==2:
                if base_2 == 16:
                    return hex(int(s_n,base_1))
            if base_1==16:
                if base_2 == 2:
                    return bin(int(s_n,base_1))
    
print(conversion(1100, 2, 16))
print(conversion("c", 16, 2))
print(conversion(15, 10, 16))


# In[ ]:


import numpy as np
m = np.arange(12).reshape((3,4))
for i in range(len(m)):
                print("rows are like=",m[i])


# In[ ]:


import numpy as np
m = np.arange(12).reshape((3,4))
for i in range(len(m[0])):
                     print("colmns are like=",m[:,i])


# In[36]:


#2 
def s_conversion(word):
    print(word)
    word=word[::-1]
    print(len(word))
    segno=word[31]
    e=int(word[24:31],2)
    bias=127
    esp=e-bias
    mantissa=int(word[0:24],2)
    number = (1+ mantissa*0.1)*2**esp
    if segno=='0':
        return '+'+str(number)
    else:
        return '-'+str(number)

s_conversion('11000000101100000000000000000000')


# In[37]:


#3
detect_under=float(1)
detect_over=float(1)
i=0
m=0
while detect_under>0:
    i +=1
    detect_under=detect_under/2
print(detect_under)
print(i)

while detect_over<float('Inf'):
    m+=1
    detect_over=detect_over*2
    
print(detect_over)
print(m)
print(2**m)


# In[4]:


import struct

def floatToBinary32(value):
    return ''.join(f'{c:0>8b}' for c in struct.pack('!f', value))

def binaryToFloat(value):
    hx = hex(int(value, 2))   
    return struct.unpack("f", struct.pack("l", int(hx, 16)))[0]

# float to binary
fl0 = 19.5
binstr = floatToBinary32(fl0)
print(f'Binary equivalent of {fl0}: {binstr}')

# binary to float
fl1 = binaryToFloat(binstr)
print(f'Decimal equivalent of      {binstr}: {fl1}')

print(f'\nSign     ( 1 bit ) = {binstr[0]}\nExponent ( 8 bits) = {binstr[1:9]}\nMantissa (23 bits) = {binstr[9:]}')

assert fl0 == fl1


# In[38]:


#4
var=float(1)
eps=float(1)
it=0
while (var + eps) > var:
    eps=eps/2
    it +=1
print(eps)
print(it)
print(var+eps)


# In[8]:



N=100
Underflow=1
Overflow=1
factor=2
for n in range(N):
    Underflow=Underflow/2
    Overflow=Overflow*2
    print("|%2d"%n,"\t\t","|2.5e"%Underflow,"|","\t\t","|2.5e" %Overflow,"|")
    
    


# In[ ]:





# In[4]:


#5
import math

a,b,c = input("Enter the coefficients of a, b and c separated by commas: ")

d = b**2-4*a*c # discriminant

if d < 0:
    print ("This equation has no real solution")
elif d == 0:
    x = (-b+math.sqrt(b**2-4*a*c))/2*a
    print ("This equation has one solutions: ", x)
else:
    x1 = (-b+math.sqrt(b**2-4*a*c))/2*a
    x2 = (-b-math.sqrt(b**2-4*a*c))/2*a
    print ("This equation has two solutions: ", x1, " and", x2)


# In[14]:


#5
# import complex math module
import math

a = 0.001
b = 1000
c = 0.001

# calculating the discriminant
dis = (b**2) - (4 * a*c)

# find two results
ans1 = (-b - (math.sqrt(dis))/(2 * a))
ans2 = (-b + (math.sqrt(dis))/(2 * a))

# printing the results
print('The roots are')
print(ans1)
print(ans2)


# In[42]:


def sol_eq_2(a,b,c):
    d=math.sqrt(b**2-4*a*c)
    x1=4*a*c/((2*a)*(-b-d))
    x2=4*a*c/((2*a)*(-b+d))
    return x1, x2

print(sol_eq_2(0.001,1000,0.001))


# In[44]:


for i in range(2):
    print(sol_eq_2(0.001,1000,0.001)[i]-sol_eq_2(0.001,1000,0.001)[i])


# In[45]:


#6
def f(x):
    return x*(x-1)


# In[28]:


from sympy import limit, Symbol

x = Symbol('x')
deltas=[10**-2,10**-4,10**-6,10**-8,10**-10,10**-12,10**-14]
for i in  deltas:
                     y=(f(x+i) -f(x))/10**-2
                     print(limit(y, x,1) )


# In[ ]:


#7
import math
def semicircle(x):
    return math.sqrt(1-x**2)

def integral(n):
    N = n
    h = 2/N
    I=0
    true_I=1.57079632679
    value=-1
    for i in range(1,N+1):
        I+=h*semicircle(value)
        value+=h
    return (I,true_I-I)

N=100

I, I_comparison = integral(N)
print("integral computation: ",I)
print("inegral compared: ",true_I-I)


# In[ ]:


N=100
for i in range(7):
    print('N:',N,'result:',integral(N))
    get_ipython().run_line_magic('timeit', 'integral(N)')
    N = N*10
    print()


# In[ ]:





# In[ ]:




