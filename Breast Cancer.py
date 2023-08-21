#!/usr/bin/env python
# coding: utf-8

# ![Breast%20Cancer%20Prediction%20%282%29.png](attachment:Breast%20Cancer%20Prediction%20%282%29.png)

# # Step 1: We import various module required for the objectives of this work. 
# We also read and load the breast cancer data. df is the name of our dataset.

# In[163]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')


# In[162]:


df=pd.read_csv("Breast_Cancer.csv")
print(df)


# In[3]:


df.head(10)


# In[4]:


df.tail(10)


# # Step 2: We will obtain a list of all the columns of our dataset

# In[6]:


df.shape


# In[7]:


df.columns


# # Step 3: We obtained information about our dataset showing their dtypes and non_null counts

# In[8]:


df.info()


# # Step 4: Data Stats

# In[9]:


df.describe()


# # Step 4: Checking Null Values

# In[10]:


df.isnull()


# In[11]:


df.isnull().sum()


# We will be exploring the Age variable of our data

# In[12]:


df['Age'].value_counts()


# # Step 5 : Performing EDA
# We will obtained the counts of ages in the dataset and show the visual

# In[168]:


plt.figure(figsize=(15,6))
ax = sns.countplot(data=df, x=df["Age"])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
plt.title("Value counts of Ages")
plt.xlabel('Age of Breast-Cancer pateints')


# Cleaning Spaces, white spaces and checking unique values for Marital status column

# In[23]:


unique_marital_status = df['Marital Status'].unique()
print("Unique Marital Status Values:", unique_marital_status)

# Cleaning up white spaces and remove any leading/trailing spaces
df['Marital Status'] = df['Marital Status'].str.strip()

# Updating incorrect values, if any
df['Marital Status'].replace({'Single ': 'Single'}, inplace=True)

# Checking unique values again to confirm cleaning
unique_marital_status_cleaned = df['Marital Status'].unique()
print("Unique Marital Status Values (Cleaned):", unique_marital_status_cleaned)

df.to_csv('Breast_Cancer.csv', index=False)


# Correlation Heat map
# 
# Selected only numerical columns from the dataset

# In[87]:


numerical_columns = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']

numerical_data = df[numerical_columns]

correlation = numerical_data.corr().round(2)
plt.figure(figsize=(14, 7))
sns.heatmap(correlation, annot=True, cmap='RdBu')
plt.title('Correlation Heatmap')
plt.show()


# Distribution of Races

# In[127]:


labels = {
    0: "White Race people",
    1: "Black Race people",
    2: "Other Race"
}

values = {
    0: 3413,
    2: 320,
    1: 291
}

# Creating a DataFrame
race_data = pd.DataFrame(values.values(), index=values.keys(), columns=['Count'])

# Ploting the pie chart
colors = ['coral', 'tan', 'wheat']  # Use specific colors for each label
explode = (0.1, 0, 0)  # Explode the first slice for emphasis

plt.pie(race_data['Count'], labels=[labels[key] for key in race_data.index], colors=colors, autopct='%1.1f%%', startangle=360, explode=explode)
plt.title("Distribution of Races")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# From the above pie chart we can conclude that white race people are more affected

# Now, lets understand counts of different count of status i.e. counts of patients that are alive or dead based on their differentiation, grades, Race, Marital Status T-Stage and N-Stage cancer

# In[140]:


df['Race'].value_counts()


# In[141]:


df['T Stage '].value_counts()


# In[142]:


df['N Stage'].value_counts()


# In[149]:


df['Grade'].value_counts()


# In[144]:


df['differentiate'].value_counts()


# Understanding the same through graph plots-

# In[119]:


plt.figure(figsize = (20,20))

plt.subplot(3,2,1)
sns.countplot(x = 'Status', hue= 'Race', palette='Reds', data = df)

plt.subplot(3,2,2)
sns.countplot(x = 'Status', hue= 'Marital Status', palette='Reds', data = df)

plt.subplot(3,2,3)
sns.countplot(x = 'Status', hue= 'differentiate', palette='Reds', data = df)

plt.subplot(3,2,4)
sns.countplot(x = 'Status', hue= 'Grade', palette='Reds', data = df)

plt.subplot(3,2,5)
sns.countplot(x = 'Status', hue= 'T Stage ', palette='Reds', data = df)

plt.subplot(3,2,6)
sns.countplot(x = 'Status', hue= 'N Stage', palette='Reds', data = df)


# Now, lets understand counts of different count of status i.e. counts of patients that are alive or dead based on their Estrogen Status, Progesterone Status, 6th Stage Cancer 

# In[147]:


df['6th Stage'].value_counts()


# In[150]:


df['Estrogen Status'].value_counts()


# In[151]:


df['Progesterone Status'].value_counts()


# Understanding the same through graph plots-

# In[146]:


plt.figure(figsize = (15,10))

plt.subplot(2,2,1)
sns.countplot(x = 'Status', hue= '6th Stage', palette='RdGy', data = df)

plt.subplot(2,2,2)
sns.countplot(x = 'Status', hue= 'Estrogen Status', palette='RdGy', data = df)

plt.subplot(2,2,3)
sns.countplot(x = 'Status', hue= 'Progesterone Status', palette='RdGy', data = df)


# In[81]:


plt.figure(figsize=(15,8))
x=df['Tumor Size']
sns.displot(x,kde=True,color='#e74c3c')
plt.show()


# In[96]:


plt.figure(figsize=(12,7))
sns.histplot(data=df, x='Age', hue='Status',palette="dark:salmon_r",kde=True)


# In[110]:


plt.rcParams['font.size']= 10
sns.pairplot(df,hue='Status', palette='hls')


# In[180]:


sns.displot(data["Regional Node Examined"], kde=True, color=("skyblue"),height=8,aspect=1,facet_kws=None)
plt.title("Histogram plot of Regional Node Examined (RNE)")
plt.show()


# In[191]:


a4_dims = (7, 5)
fig, ax = plt.subplots(figsize=a4_dims) # 0 = T1 , 1 = T2 , 2 = T3, 3 = T4
sns.barplot(x=data['T Stage '] ,y=data['Tumor Size'],palette='vlag').set(xlabel='Tumor Stage', ylabel='Tumor Size')
plt.title("Bar plot of Tumor Stage vs Tumor Size")
plt.show()


# In[193]:


k = sns.displot(data['Reginol Node Positive'], kde=True, color=("red"),height=5,aspect=1,facet_kws=None)
plt.title("Histogram plot of Regional Node Positive (RNP)")
plt.show()


# In[ ]:




