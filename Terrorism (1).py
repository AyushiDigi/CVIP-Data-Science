#!/usr/bin/env python
# coding: utf-8

# ![images%20%2812%29.jpeg](attachment:images%20%2812%29.jpeg)

# #                    #                          Task : Exploratory Data Analysis - Terrorism

# # Objective :
#     
#     1. Perform Exploratory Data Analysis on dataset : "Global Terrorsim"
#     2. As a Security Analyst, try to find ou the hot zone of terrorism.
#     3. What all security issues and Insights you can derive by EDA?

# # Importing Required Libraries

# In[59]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy 
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')


# # Importing Required Data Set

# In[25]:


df=pd.read_csv("Terrorism.csv",encoding='ISO-8859-1')
print(df)


# # Understanding the Data Set

# In[4]:


df.head()


# In[5]:


df.tail()


# # Dataset stats

# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.info()


# # Finding Null Values

# In[9]:


df.isnull()


# In[10]:


df.isnull().sum()


# In[11]:


df=df[['eventid', 'iyear', 'imonth', 'country', 'region','provstate','city','crit1', 'crit2', 'crit3','success', 'suicide', 'attacktype1','targtype1','natlty1','gname','guncertain1','claimed','weaptype1','nkill','nwound']]
df.head()


# # Removing Null Values

# In[12]:


df.isnull().any()


# In[13]:


df = df.dropna()


# In[14]:


df.isnull().any()


# In[15]:


df.info()


# # Now that there are no null values so we can start the analysis

# In[16]:


df.head()


# In[17]:


df['attacktype1'].value_counts()


# In[18]:


df['provstate'].value_counts()


# In[19]:


df['iyear'].value_counts()


# # Performing EDA- EXPLORATORY DATA ANALYSIS

# # Type of Attacks

# In[26]:


plt.figure(figsize=(15, 6))
sns.countplot(x='attacktype1_txt', data=df, palette="rocket")
plt.xlabel('Attack type')
plt.ylabel('No. of attacks')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# Most number of type of attacks was done by Bombing/Explosion


# # Region Wise Attacks

# In[29]:


ra = df["region_txt"].value_counts().head(10)
print(ra)

# Created the bar plot using a gradient color palette
sns.barplot(x=ra.index, y=ra.values, palette="viridis")

plt.xlabel('Region')
plt.ylabel('Number of attacks')
plt.title('Attacks Region Wise')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[30]:


pd.crosstab(df.iyear,df.region_txt).plot(kind="area",stacked=False,figsize=(18,6))
plt.xlabel('year')
plt.ylabel('number of attacks')
plt.title('Terrorist Attacks Region Wise')


# In[ ]:


# Most number of attacks happened after 2010
# Most of attacks took place in Middle East and North Africa


# # Country Wise Attacks

# In[28]:


ca = df["country_txt"].value_counts().head(10)
print(ca)

sns.barplot(x=ca.index, y=ca.values, palette="coolwarm")

plt.xlabel('Country')
plt.ylabel('Number of attacks')
plt.title('Attacks Country Wise')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# # Top 15 Areas that are Attacked

# In[31]:


plt.figure(figsize=(10, 6), dpi=100)

sns.barplot(
    x=df['provstate'].value_counts()[:15].index,
    y=df['provstate'].value_counts()[:15].values,
    palette="icefire")

plt.title("Top 15 areas that are attacked")
plt.xlabel("Area")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()

plt.show()


# # Attacks Every Year

# In[32]:


plt.figure(figsize=(31,20),dpi=400)
sns.countplot(x='iyear',data=df)
plt.xlabel('year')
plt.ylabel('number of attacks')
plt.title('Terrorist Attacks Every Year')


# In[ ]:


ay=df["iyear"].value_counts(dropna=False)
print(ay)


# In[21]:


#from the above we can conclude that the least number of attacks were made in 1971 and most number of attacks in 2014


# # Attacks in TOP  10 States

# In[66]:


plt.figure(figsize=(4, 4))

provstate_counts = df['provstate'].value_counts().nsmallest(10)
pie_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#ff6666', '#c2f0c2', '#d9b3ff', '#ffdb4d']

wedges, texts, autotexts = plt.pie(provstate_counts, labels=provstate_counts.index, colors=pie_colors, autopct='%1.1f%%')


for text in texts + autotexts:
    text.set_fontsize(6) 

plt.title("Distribution of Attacks in Top 10 States")

plt.show()


# # Attacks in TOP 10 Cities

# In[23]:


plt.figure(figsize=(7, 5))
top_cities = df['city'].value_counts().nsmallest(10).index
subset = df[df['city'].isin(top_cities)]
heatmap_data = subset.pivot_table(index='city', columns='iyear', values='attacktype1', aggfunc='sum')

sns.heatmap(heatmap_data, cmap="YlOrRd")

plt.title("Attack Counts in Top 10 Cities with Fewest Attacks")
plt.xlabel("Year")
plt.ylabel("City")
plt.tight_layout()

plt.show()


# # Weapon Wise Attacks

# In[39]:


wt=df["weaptype1_txt"].value_counts()
print(wt)
plt.figure(figsize=(15, 6))
sns.countplot(x='weaptype1_txt', data=df, palette="Paired")
plt.xlabel('Weapon Types')
plt.ylabel('No. of attacks')
plt.title('Weapon Wise Attacks')
plt.xticks(rotation=90)
plt.show()


# # Target Type Attacks

# In[49]:


tt= df["targtype1_txt"].value_counts()
print(tt)
plt.figure(figsize=(15, 6))
sns.countplot(x='targtype1_txt', data=df, palette="magma")
plt.xlabel('Target Types')
plt.ylabel('No. of attacks')
plt.title('Target Wise Attacks')
plt.xticks(rotation=90)
plt.show()


# # Conclusion:
# 
# From the given analysis, we can conclude that :
# 
#     1. Terrorist attacks increased after 2010.
#     2. 2014 had the largest amount of terrorist attacks.
#     3. In cases of hotzones for terrorism, Baghdad,Karachi,Lima and Mosul rank the highest. Bagdad was the highest.
#     4. The safest zones were Equator, Altai, burgas etc.
#     5. Boftari, Dorgali are some of the safest cities.
