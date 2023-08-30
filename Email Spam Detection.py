#!/usr/bin/env python
# coding: utf-8

# ![Email%20Spam%20Detection.png](attachment:Email%20Spam%20Detection.png)

# # IMPORTING REQUIRED LIBRARIES

# In[31]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import nltk
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # LOADING AND UNDERSTANDING THE DATA

# In[3]:


df= pd.read_csv('spam.csv', encoding='latin-1')
print(df)


# In[6]:


df.head(10)


# In[7]:


df.tail(10)


# # Obtaining a list of all the columns of our dataset

# In[8]:


df.shape


# In[9]:


df.columns


# # Data Stats

# In[11]:


df.describe()


# # Checking Null Values

# In[10]:


df.info()


# In[12]:


df.isnull()


# In[13]:


df.isnull().sum()


# In[ ]:


# Here Columns - Unnamed 2, Unnamed 3, Unnamed 4 has null values .


# # CLEANING AND PROCESSING DATA

# In[15]:


df = df.where((pd.notnull(df)),'')
print(df)


# In[16]:


df.isna().sum()


# In[17]:


# Now we can see there are no null values. But now columns- Unnamed 2, Unnamed 3, Unnamed is of no use now. So we need to remove them .


# In[18]:


df.drop(['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)


# In[19]:


df.shape


# In[20]:


df.head()


# In[ ]:


# Now we can see the unnecessary columns are removed from our data


# # Renaming Columns

# In[21]:


df.rename(columns={'v1':'Category','v2':'Message'}, inplace=True)


# In[22]:


df.head()


# In[32]:


df.columns


# In[ ]:


#Column names are changed


# # Performing EDA

# In[29]:


plt.figure(figsize=(4, 4))
plt.pie(df['Category'].value_counts(), labels=['ham', 'spam'], autopct='%0.2f%%', explode=[0.1, 0])
plt.show()


# In[33]:


df['Message Length'] = df['Message'].apply(len)

# Create a histogram to show the distribution of message lengths
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='Message Length', hue='Category', bins=20, alpha=0.7)
plt.title('Message Length Distribution by Category')
plt.xlabel('Message Length')
plt.ylabel('Count')
plt.legend(title='Category')
plt.show()


# Message Word Cloud

# In[37]:


from wordcloud import WordCloud

# Combine all messages into a single string
all_messages = ' '.join(df['Message'])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_messages)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Message Word Cloud')
plt.axis('off')
plt.show()


# Top Words in Spam Messages

# In[38]:


from collections import Counter

# Filter only spam messages
spam_messages = ' '.join(df[df['Category'] == 'spam']['Message'])

# Tokenize and count word frequencies
words = spam_messages.split()
word_counts = Counter(words)

# Create a bar plot for top N words in spam messages
plt.figure(figsize=(10, 6))
sns.barplot(x=list(word_counts.keys())[:10], y=list(word_counts.values())[:10])
plt.title('Top Words in Spam Messages')
plt.xlabel('Word')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# Message Length Analysis:

# In[39]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Category', y='Message Length', palette='Set3')
plt.title('Message Length Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Message Length')
plt.show()


# Pairplot by Category

# In[40]:


df['Category'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Create a pair plot
sns.pairplot(df, hue='Category', markers=["o", "s"], diag_kind='kde')
plt.show()


# Sentiment Analysis of Messages

# In[43]:


from textblob import TextBlob

df['Sentiment'] = df['Message'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Create a scatter plot to visualize sentiment
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Sentiment', y=df.index, hue='Sentiment', size='Sentiment', sizes=(50, 200))
plt.title('Sentiment Analysis of Messages')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Message')
plt.legend(title='Sentiment')
plt.show()


# Message Category Distribution

# In[49]:


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Category', palette='Set2')
plt.title('Message Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()


# Most Common Words:

# In[50]:


from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

# Tokenize and count word frequencies
all_words = [word.lower() for message in df['Message'] for word in word_tokenize(message) if word.isalpha() and word.lower() not in stop_words]
word_counts = Counter(all_words)

# Get top N most common words
top_words = word_counts.most_common(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=[word[0] for word in top_words], y=[word[1] for word in top_words])
plt.title('Top 10 Most Common Words')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()


# Combined Analysis Results

# In[59]:


df['Contains_Urgent'] = df['Message'].apply(lambda x: 'urgent' in x.lower())
df['Is_Question'] = df['Message'].apply(lambda x: x.endswith('?'))

# Message Content Analysis
df['Is_Meeting'] = df['Message'].apply(lambda x: 'meeting' in x.lower())

# Combine all analysis categories
df['Analysis_Type'] = 'None'
df.loc[df['Contains_Urgent'], 'Analysis_Type'] = 'Keyword: Urgent'
df.loc[df['Is_Question'], 'Analysis_Type'] = 'Question'
df.loc[df['Is_Meeting'], 'Analysis_Type'] = 'Meeting'

# Create a stacked bar chart
analysis_counts = df['Analysis_Type'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=analysis_counts.index, y=analysis_counts.values, palette='Set2')
plt.title('Combined Analysis Results')
plt.xlabel('Analysis Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[ ]:




