#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import requests
import numpy as np
import random
import csv
import json
from os import path
import io


# In[3]:


#1
n = open("simple_data.txt", "w")
num = [random.randrange(0,100) for i in range(10)]

n.write(str(num))

n.close()

n = open("simple_data.txt", "r")
print(n.read())
n.close()


# In[4]:


#2
b = open("data.txt", "w")
matrix = np.random.random(25).reshape(5,5).round(2)

b.write(str(matrix))

b.close()

b = open("data.txt", "r")
print(b.read())
b.close()


# In[7]:


#3
with open('data.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('data.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(lines)
writer


# In[8]:


#4
f = open("credit_card.dat", "r")

read = True

while(read):
    credit_card = ""
    for i in range(19):
        bit_group = f.readline(6)
        if(len(bit_group)==6): credit_card = credit_card+chr(int(bit_group,2))        
        if(i == 18): f.readline()
    print(credit_card)
    if len(bit_group) == 0:
        read = False

f.close()


# In[9]:


#5
data = json.load(open('user_data.json'))

filtered_data = []

for user in data:
        if(user['CreditCardType'] == "American Express"):
            filtered_data.append(user)
            

out_file = open(path.expanduser("filtered_credit_cart.json"), "w")

json.dump(filtered_data, out_file)

out_file.close()  

df = pd.read_json(path.expanduser("filtered_credit_cart.json"))
df.to_csv(path.expanduser("filtered_credit_cart.csv"), index = None)


# In[10]:


#6
url = 'https://www.dropbox.com/s/7u3lm737ogbqsg8/mushrooms_categorized.csv?dl=1'
s=requests.get(url).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))
print(c.info)

c.hist("class")


# In[11]:


#7
url = 'https://www.dropbox.com/s/vkl89yce7xjdq4n/regression_generated.csv?dl=1'
r=requests.get(url).content
c=pd.read_csv(io.StringIO(r.decode('utf-8')))

for i in range(3):
    for j in range(3):
        if(i!=j and i<j): c.plot.scatter("features_"+str(i+1),"features_"+str(j+1))


# In[12]:


#8
url = 'https://www.dropbox.com/s/7u3lm737ogbqsg8/mushrooms_categorized.csv?dl=1'
s=requests.get(url).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))

c.to_json("mushrooms_categorized.json")

