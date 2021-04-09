#!/usr/bin/env python
# coding: utf-8

# https://towardsdatascience.com/a-beginners-guide-to-sentiment-analysis-in-python-95e354ea84f6

# In[1]:


import pandas as pd 
from pywebio.input import *
from pywebio.output import *


# API KEY 
# 0cZbnYBjL5gXfK3nQAtaTBhCB
# 
# API SECRET KEY 
# AwcNDR48iGOc7Uf2eA9h4Dz7nGPlABAp4scMsIQsJrqPzeWFs0
# 
# BEARER TOKEN 
# AAAAAAAAAAAAAAAAAAAAAPXEMQEAAAAAqYYfNHG7Gsofn7Td8RaINw1QyIU%3DzefvGumyno5WMLwrGRIQJs0S10r2UCYuwk3NJDLN8A790g2PVR
# 
# 
# 
# Access token:
# 1065450120433553410-JM1bl1KXMCZa9yrjw6K2IokKGwQ4GV
# 
# 
# Access token secret:
# RyO2JjxFOZgf5HWFKo0BQHAQLyCdWspbQjv7MjCPPLzYe

# In[2]:


import numpy as np 


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


import tweepy
import csv
import re
import json
import pandas as pd
import csv
import re
from textblob import TextBlob
import string
#import preprocessor as p
import os
import time
from tweepy import OAuthHandler


# In[5]:




consumer_key = '0cZbnYBjL5gXfK3nQAtaTBhCB'
consumer_secret = 'AwcNDR48iGOc7Uf2eA9h4Dz7nGPlABAp4scMsIQsJrqPzeWFs0'
access_key = '1065450120433553410-JM1bl1KXMCZa9yrjw6K2IokKGwQ4GV'
access_secret = 'RyO2JjxFOZgf5HWFKo0BQHAQLyCdWspbQjv7MjCPPLzYe'


# Pass your twitter credentials to tweepy via its OAuthHandler
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


# In[6]:



# 3.To get the tweets in a Proper format, first lets create a Dataframe to store the extracted data.

df = pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT",'User_location'])
print(df)


# In[7]:



# We will use api as api.search inside this tweepy cursor.


# In[8]:


# 4.Write a Function to extract tweets:

# We will Use **tweepy.cursor()** because we want to extract a larger number of tweets i.e over 100,500 etc


def get_tweets(Topic,Count):    
    i=0
    for tweet in tweepy.Cursor(api.search, q=Topic,count=100, lang="en",exclude='retweets').items():
        print(i, end='\r')
        df.loc[i,"Date"] = tweet.created_at
        df.loc[i,"User"] = tweet.user.name
        df.loc[i,"IsVerified"] = tweet.user.verified
        df.loc[i,"Tweet"] = tweet.text
        df.loc[i,"Likes"] = tweet.favorite_count
        df.loc[i,"RT"] = tweet.retweet_count
        df.loc[i,"User_location"] = tweet.user.location
        #df.to_csv("TweetDataset.csv",index=False)
        df.to_excel('{}.xlsx'.format("TweetDataset"),index=False)   ## Save as Excel
        i=i+1
        if i>Count:
            break
        else:
            pass


# In[9]:



# Call the function to extract the data. pass the topic and filename you want the data to be stored in.
Topic=["modi"]
get_tweets(Topic , Count=100)


# In[10]:


df.head(8)


# In[11]:



# Function to Clean the Tweet.

import re
def clean_tweet(tweet):
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', str(tweet).lower()).split())

# We only want the Text so :

# (@[A-Za-z0-9]+)   : Delete Anything like @hello @Letsupgrade etc
# ([^0-9A-Za-z \t]) : Delete everything other than text,number,space,tabspace
# (\w+:\/\/\S+)     : Delete https://
# ([RT]) : Remove "RT" from the tweet


# In[12]:



# Funciton to analyze Sentiment

from textblob import TextBlob
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'


# In[13]:



#Function to Pre-process data for Worlcloud:here we are removing the words present in Topic from the Corpus so they dont come in WordCloud.
# Ex : Topic is "Arsenal vs United", we want to remove "Arsenal" "vs" "United" from the WordCloud.

def prepCloud(Topic_text,Topic):
    Topic = str(Topic).lower()
    Topic=' '.join(re.sub('([^0-9A-Za-z \t])', ' ', Topic).split())
    Topic = re.split("\s+",str(Topic))
    stopwords = set(STOPWORDS)
    stopwords.update(Topic) ### Add our topic in Stopwords, so it doesnt appear in wordClous
    ###
    text_new = " ".join([txt for txt in Topic_text.split() if txt not in stopwords])
    return text_new


# In[14]:



# Call function to get Clean tweets

df['clean_tweet'] = df['Tweet'].apply(lambda x : clean_tweet(x))
df.head(5)


# In[15]:



# Call function to get the Sentiments

df["Sentiment"] = df["Tweet"].apply(lambda x : analyze_sentiment(x))
df.head(5)


# In[16]:


# Check Summary of Random Record
n = 15
print("Original tweet:\n",df['Tweet'][n])
print()
print("Clean tweet:\n",df['clean_tweet'][n])
print()
print("Sentiment of the tweet:\n",df['Sentiment'][n])


# In[17]:


# Overall Summary

print("Total Tweets Extracted for Topic : {} are : {}".format(Topic,len(df.Tweet)))
print("Total Positive Tweets are : {}".format(len(df[df["Sentiment"]=="Positive"])))
print("Total Negative Tweets are : {}".format(len(df[df["Sentiment"]=="Negative"])))
print("Total Neutral Tweets are : {}".format(len(df[df["Sentiment"]=="Neutral"])))


# In[ ]:





# In[ ]:




