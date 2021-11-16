#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1.1
ans = []
[  ans.append((i, j)) for i in range(3) for j in range(4) ]
ans








# In[ ]:


# 1.2
ans = [x*x for x in range(5) if x%2 == 0]
print(ans)


# In[2]:


#2
x = 5
def f(alist):
    for i in range(x):
         alist.append(i)
    return alist

alist = [1,2,3]
ans = f(alist)
print (ans)
print (alist) # alist has been changed!

# New function
def f2(alist):
    blist = alist.copy()
    for i in range(x):
         blist.append(i)
    return blist

alist = [1,2,3]
ans = f2(alist)
print (ans)
print (alist)


# In[ ]:


#3
def hello(x):
    return "hello"
hello(x)


# In[ ]:


#4.1
n=int(input())
def factorial(n):
    fact = 1
    for num in range(2, n + 1):
        fact *= num
    return fact
factorial(n)


# In[ ]:


4.2#
# Python 3 program to find
# factorial of given number
import math

def factorial(n):
     return(math.factorial(n))


# Driver Code
num = 3
print("Factorial of", num, "is",
factorial(num))


# In[ ]:


#5.1
import math
densities = {"Al":[0.5,1,2],"Fe":[3,4,5],"Pb": [15,20,30]}
radii =[1,2,3]
import math
for al, fe, pb in zip(densities['Al'], densities['Fe'], densities['Pb']):
    if radii==1 :
        for i in map (lambda r : r*2*math.pi*al, radii):print ('Al circle:',i)
        for i in map (lambda r : r*r*math.pi*al, radii):print ('Al disk:',i)
        for i in map (lambda r : 4/3*r*r*r*math.pi*al, radii):print ('Al sphere:',i)
        
    if radii==2 :
        
        for i in map (lambda r : r*2*math.pi*fe, radii):print ('Fe circle:',i)
        for i in map (lambda r : r*r*math.pi*fe, radii):print ('Fe disk:',i)
        for i in map (lambda r : 4/3*r*r*r*math.pi*fe, radii):print ('Fe sphere:',i)
        
        
    if radii==3:
        for i in map (lambda r : r*2*math.pi*pb, radii):print ('Pb circle:',i)
        for i in map (lambda r : r*r*math.pi*pb, radii):print ('Pb disk:',i)
        for i in map (lambda x : 4/3*r*r*r*math.pi*pb, radii):print ('Pb sphere:',i)

#5.2

radii =[1,2,3]
for i in map (lambda r : r*2*math.pi, radii) : print (i)
for i in map (lambda r : r*r*math.pi, radii) : print (i)
for i in map (lambda r : 4/3*r*r*r*math.pi, radii) : print (i)


# In[5]:


#6
dogs=[]
class Dog:

    # Class attribute
    species = 'mammal'
    _counter = 0
    # Initializer / Instance attributes
    def init(self, name, age):
        dogs.append(self)
        self.name = name
        self.age = age
        Dog._counter += 1
        self.id = Dog._counter
        self.is_hungry=True
        
    def eat(self):
        self.is_hungry=False
        
    def inf(self):
        if self.is_hungry==True:
            print("{}".format(self.name),'is hungry')
        else:
            print("{}".format(self.name),'is not hungry')
        return "{}".format(self.id)
    def hung(self):
        return self.is_hungry
    def num(self):
        print(self)

    # instance method
    def description(self):
        return "{} is {} years old".format(self.name, self.age)

    # instance method
    def speak(self, sound):
        return "{} says {}".format(self.name, sound)

# Child class (inherits from Dog class)
class RussellTerrier(Dog):
    def run(self, speed):
        return "{} runs {}".format(self.name, speed)

# Child class (inherits from Dog class)
class Bulldog(Dog):
    def run(self, speed):
        return "{} runs {}".format(self.name, speed)
Tom=Dog('Tom',10)
bob=Dog('bob',2)
Fletcher=Dog('Fletcher',7)

Tom.eat()
bob.eat()
Fletcher.eat()

def information():
    print('I have',len(dogs),'dogs.')
    hung=False
    for i in dogs:
        print(i.name,'is',i.age,'.')
        
        if i.species=='mammal':
            mammal=True
        else:
            mammal=False
            
        if i.hung()==True:
            hung=True
            print("{}".format(i.name),'is hungry')
            
    if mammal==True:
        print('And they are all mammals, of course.' )
    if hung==True:
        print('My dogs are hungry.')
    else:
        print('My dogs are not hungry.')
        
information()


# In[3]:


class Dog:
    
    # Class Attribute
    species = 'mammal'
    
    # Initializer / Intance attributes
    def __init__(self, name, age, is_hungry):
        self.name = name
        self.age = age
        self.is_hungry = is_hungry
    
        # instance method
    def description(self):
        return "{} is {} years old".format(self.name, self.age)

    # instance method
    def speak(self, sound):
        return "{} says {}".format(self.name, sound)
    
    # Eat method
    def eat(self):
        self.is_hungry = False
    
# Child class (inherits from Dog class)
class RussellTerrier(Dog):
    def run(self, speed):
        return "{} runs {}".format(self.name, speed)

# Child class (inherits from Dog class)
class Bulldog(Dog):
    def run(self, speed):
        return "{} runs {}".format(self.name, speed)

# Method to feed all dogs
def feedall(dogs):
    count = 0
    for dog in dogs:
        if dog.is_hungry == True:
            dog.eat()
        else:
            count += 1
    if count == len(dogs):
        print("My dogs are not hungry")
    elif count == 0:
        print("My dogs are hungry")


# In[4]:


mydogs = []
mydogs.append(Dog("Tom",6,False))
mydogs.append(Dog("Fletcher",7,False))
mydogs.append(Dog("Larry",9,False))

print("I have", len(mydogs), "dogs")
for dog in mydogs: print(dog.description())
print("And they are all", mydogs[0].species + "s,", "of course")
feedall(mydogs)


# In[ ]:




