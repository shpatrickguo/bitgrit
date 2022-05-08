#!/usr/bin/env python
# coding: utf-8

# # Viral Tweets Prediction Challenge
# Develop a machine learning model to predict the virality level of each tweet based on attributes such as tweet content, media attached to the tweet, and date/time published.
# 
# In this notebook:
# - Data processing: one-hot encoding + cyclical encoding for categorical features. Normalization.
# - LASSO regression for feature selection
# - Memory footprint reduction of data
# - Hyper-parameter tuning with RandomizedSearchCV
# - Building LightGBM classifier model for prediction
# - Feature importance visualization

# # Import libraries

# In[1]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import time
import timeit
import collections
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

# Preprocessing + Feature Selection
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

# Model Building
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Hyperparameter tuning 
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# Metrics
from sklearn.metrics import accuracy_score


# In[2]:


# Function takes the minimum and the maximum of each column and changes the data type to what is optimal for the column.
def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# # Data retrieval

# In[3]:


# Kagge storage path
path = "../input/viral-tweets/Dataset/"

# Training datasets
train_tweets = pd.read_csv(path + 'Tweets/train_tweets.csv')
train_tweets_vectorized_media = pd.read_csv(path + 'Tweets/train_tweets_vectorized_media.csv')
train_tweets_vectorized_text = pd.read_csv(path + 'Tweets/train_tweets_vectorized_text.csv')

# Test dataset
test_tweets = pd.read_csv(path + 'Tweets/test_tweets.csv')
test_tweets_vectorized_media = pd.read_csv(path + 'Tweets/test_tweets_vectorized_media.csv')
test_tweets_vectorized_text = pd.read_csv(path + 'Tweets/test_tweets_vectorized_text.csv')

# User dataset
users = pd.read_csv(path + 'Users/users.csv')
user_vectorized_descriptions = pd.read_csv(path + 'Users/user_vectorized_descriptions.csv')
user_vectorized_profile_images = pd.read_csv(path + 'Users/user_vectorized_profile_images.csv')

# Solutions format
solutions_format = pd.read_csv(path + "solution_format.csv")


# # Dimensions of Data

# In[4]:


# print dimensions of data
print('Dimensions:')
print('Train tweets:', train_tweets.shape)
print('Train tweets vectorized media:', train_tweets_vectorized_media.shape)
print('Train tweets vectorized text:', train_tweets_vectorized_text.shape)
print()

print('Test tweets:', test_tweets.shape)
print('Test tweets vectorized media:', test_tweets_vectorized_media.shape)
print('Test tweets vectorized text:', test_tweets_vectorized_text.shape)
print()

print('Users:', users.shape)
print('User vectorized descriptions:', user_vectorized_descriptions.shape)
print('User vectorized profile images:', user_vectorized_profile_images.shape)


# The dimensions for ```Users``` are smaller than ```Tweets```, which indicate that the users in the dataset may have multiple tweets.    
# Vectorized text has the same number of rows as tweets, meaning that all tweets have text.  
# Vectorized media has fewer rows than tweets, indicating that not all tweets have media or that some tweets have multiple media.  
# All ```Users``` have descriptions and profile images.

# # Exploratory Data Analysis
# ## Train Tweets

# In[5]:


train_tweets.head()


# Primary Keys: ```tweet_id```, ```tweet_user_id```. There are 11 Features. Target variable: ```virality```. Tweet data are connected thorugh ```tweet_id```.

# In[6]:


train_tweets.info()


# ### Tweet Creation Date
# tweet_created_at_year  
# tweet_created_at_day  
# tweet_created_at_month  
# tweet_created_at_hour

# In[7]:


fig, axs = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(train_tweets, x = 'tweet_created_at_year', discrete = True, ax = axs[0,0])
sns.histplot(train_tweets, x = 'tweet_created_at_day', discrete = True, ax = axs[0,1])
sns.histplot(train_tweets, x = 'tweet_created_at_month', discrete = True, ax = axs[1,0])
sns.histplot(train_tweets, x = 'tweet_created_at_hour', discrete = True, ax = axs[1,1])
plt.show()


# - The histplot for ```tweet_created_at_year``` shows a left skeweed distribution between 2013-2020 where each subsequent year has more tweets created. Note that this data was produced during 2021, so the count for ```tweet_created_at_year``` for 2021 does not account for the full year unlike others.
# - The histplot for ```tweet_created_at_month``` show that December is the month with the highest number of tweets created. The lowest being March.
# - The histplot for ```tweet_created_at_day``` generally has a uniform distribution. The highest being 27th, perhaps because February have 28 days. The 31st is an outlier because not all months have 31 days. 
# - The histplot for ```tweet_created_at_hour``` show a cyclical distribution where most tweets are created during the afternoon/evening, the highest being 4pm. The least amount where created late at night/early in the morning.

# ### Tweet Message Content
# tweet_hashtag_count  
# tweet_url_count  
# tweet_mention_count

# In[8]:


fig, axs = plt.subplots(3, 1, figsize=(12, 12))
sns.histplot(x = 'tweet_hashtag_count', data = train_tweets, discrete = True, ax = axs[0])
sns.histplot(x = 'tweet_url_count', data = train_tweets, discrete = True, ax = axs[1])
sns.histplot(x = 'tweet_mention_count', data = train_tweets, discrete = True, ax = axs[2])
plt.show()


# - The histplot for ```tweet_hashtag_count``` is right skewed where most tweets have zero hashtags and less tweets have more hashtags.
# - The histplot for ```tweet_url_count``` shows that most tweets have one URL, and not many tweets have a high number of tweets.
# - The histplot for ```tweet_mention_count``` is right skewed where most tweets have zero mentions and less tweets have multiple hashtags.

# ### Tweet Attatchment
# tweet_has_attachment   
# tweet_attachment_class

# In[9]:


fig, axs = plt.subplots(2, 1, figsize=(10, 10))
sns.countplot(x = 'tweet_has_attachment', data = train_tweets, ax = axs[0])
sns.countplot(x = 'tweet_attachment_class', data = train_tweets, ax = axs[1])
plt.show()


# - The countplot for ```tweet_has_attachment``` shows that more tweets have an attachment, such as media.
# - The countplot for ```tweet_attachment_class``` shows that most tweets have an attachment class A, and very few tweets have attachment class B.

# ### Tweet Language

# In[10]:


fig, axs = plt.subplots(1, 1, figsize=(8, 3))
sns.countplot(x = 'tweet_language_id', data = train_tweets, ax = axs)
plt.show()


# - The countplot for ```tweet_language_id``` shows a high amount of tweets in language_id 0, which is presumed to be english. Very few tweets in this datset are in other languages.

# ### Tweet Virality

# In[11]:


sns.countplot(x = 'virality', data = train_tweets)
plt.show()


# - The countplot for ```virality``` shows the virality of tweets where 1 is low whereas 5 is high. Most tweets have a virality of 1.
# 
# Since there are 5 values in ```virality```, this means that this is a multi-class classification problem.

# ### Correlation Matrix

# In[12]:


corrmat = train_tweets.corr()[2:] 
sns.heatmap(corrmat, square=True);


# The heatmap shows that some features have correlation with each other. ```tweet_url_count``` and ```tweet_has_attachment``` has the highest correlation with each other.

# In[13]:


df_corr = train_tweets.corr()['virality'][2:-1]
top_features = df_corr.sort_values(ascending=False, key=abs)
top_features


# The correlation numbers show a low correlation between virality and features, meaning they cannot be used linearly to predict virality.

# ## Train Tweets Vecotrized Media

# In[14]:


train_tweets_vectorized_media.info()


# Primary Keys: ```media_id```, ```tweet_id```. There are 2048 Features. Tweet data are connected thorugh ```tweet_id```.

# ## Train Tweets Vectorized Text

# In[15]:


train_tweets_vectorized_text.info()


# Primary Keys: ```tweet_id```. There are 768 Features. Tweet data are connected thorugh ```tweet_id```.
# 
# Each column in Vectorized Text/Media represents one coordinate in the numeric feature space

# ## Users

# In[16]:


users.info()


# Primary Keys: ```user_id```. There are 10 Features. User data are connected thorugh ```user_id```.

# ### User Count
# user_like_count  
# user_followers_count  
# user_following_count  
# user_listed_on_count  
# user_tweet_count

# In[17]:


fig, axs = plt.subplots(2, 3, figsize=(18, 8))
sns.histplot(users, x = 'user_like_count', ax = axs[0,0])
sns.histplot(users, x = 'user_followers_count', ax = axs[0,1])
sns.histplot(users, x = 'user_following_count', ax = axs[0,2])
sns.histplot(users, x = 'user_listed_on_count', ax = axs[1,0])
sns.histplot(users, x = 'user_tweet_count', ax = axs[1,1])
axs[1][2].set_visible(False)
plt.show()


# - The histplot for ```user_like_count``` is right skewed. A large propotion of users have between 0-2500 likes.
# - The histplot for ```user_follower_count``` is right skewed. A large propotion of users have between 0-10000 followers.
# - The histplot for ```user_following_count``` is right skewed. A large propotion of users follow between 0-1000 accounts.
# - The histplot for ```user_listed_on_count``` is right skewed. A large propotion of users are listed on between 0-5000 lists.
# - The histplot for ```user_tweet_count``` is right skewed. A large propotion of users have between 0-10000 tweeets.

# ### User Creation Date

# In[18]:


fig, axs = plt.subplots(2, 1, figsize=(12, 8))
sns.histplot(users, x = 'user_created_at_year', discrete = True, ax = axs[0])
sns.histplot(users, x = 'user_created_at_month', discrete = True, ax = axs[1])
plt.show()


# - The histplot for ```user_created_at_year``` shows that most users were created in 2011.
# - The histplot for ```user_created_at_month``` shows that most users were created in August. 0 users were creaed in March, which may explain why March has the lowest tweets created.

# ### User Has
# user_has_location  
# user_has_url  
# user_verified

# In[19]:


fig, axs = plt.subplots(1, 3, figsize=(16, 6))
sns.countplot(x = 'user_has_location', data = users, ax = axs[0])
sns.countplot(x = 'user_has_url', data = users, ax = axs[1])
sns.countplot(x = 'user_verified', data = users, ax = axs[2])
plt.show()


# For the binary data: most of the users have their location and url listed on their accounts. Most of them are not verified.

# ## User Vectorized Descriptions

# In[20]:


user_vectorized_descriptions.info()


# Primary Keys: ```user_id```. There are 768 Features. User data are connected thorugh ```user_id```.
# 
# Vectorized descriptions and vectorized text have the same number of features.

# ## User Vectorized Profile Images

# In[21]:


user_vectorized_profile_images.info()


# Primary Keys: ```user_id```. There are 2048 Features. User data are connected thorugh ```user_id```.
# 
# Vectorized media and vectorized profile images have the same number of features.

# # Data Preprocessing & Wrangling

# In[22]:


train_tweets.isnull().sum()


# Only ```tweet_topic_ids``` have null values. These will treated as another tweet_topic_id by filling them with another id such as ```["0"]```. (The number does not matter as long as it is distinct from other values).

# In[23]:


train_tweets.fillna({'tweet_topic_ids':"['0']"}, inplace=True)


# The rest of the data files do not have non-null values.

# ## Categorical Variables
# ### Train Tweets
# #### One-hot encoding

# In[24]:


# Split topic ids
topic_ids = (
    train_tweets.tweet_topic_ids.str.strip('[]').str.split('\s*,\s*').explode().str.get_dummies().sum(level=0).add_prefix('topic_id_')
) 
topic_ids.rename(columns = lambda x: x.replace("'", ""), inplace=True)


# In[25]:


year = pd.get_dummies(train_tweets.tweet_created_at_year, prefix='year')
month = pd.get_dummies(train_tweets.tweet_created_at_month , prefix='month')
day = pd.get_dummies(train_tweets.tweet_created_at_day, prefix='day')
attachment = pd.get_dummies(train_tweets.tweet_attachment_class, prefix='attatchment')
language = pd.get_dummies(train_tweets.tweet_language_id, prefix='language')


# #### Cyclical Encoding
# From histplot we saw that hours have a cyclical distribution so we will us cyclical encoding.

# In[26]:


hour_sin = np.sin(2 * np.pi * train_tweets['tweet_created_at_hour']/24.0)
hour_sin.name = 'hour_sin'
hour_cos = np.cos(2 * np.pi * train_tweets['tweet_created_at_hour']/24.0)
hour_cos.name = 'hour_cos'


# In[27]:


# Join encoded data to train data.
columns_drop = [
                "tweet_topic_ids",
                "tweet_created_at_year",
                "tweet_created_at_month",
                "tweet_created_at_day",
                "tweet_attachment_class",
                "tweet_language_id",
                "tweet_created_at_hour",
               ]
encoded = [topic_ids, year, month, day, attachment, language, hour_sin, hour_cos]

train_tweets_final = train_tweets.drop(columns_drop, 1).join(encoded)
train_tweets_final.head()


# ### Users
# #### One-hot encoding

# In[28]:


year = pd.get_dummies(users.user_created_at_year, prefix='year')
month = pd.get_dummies(users.user_created_at_month , prefix='month')


# In[29]:


# Join encoded data to train data.
columns_drop = [
                "user_created_at_year",
                "user_created_at_month",
               ]
dfs = [year, month]

users_final = users.drop(columns_drop, 1).join(dfs)
users_final.head()


# ## Normalize Data
# Machine learning algorithms perform better or converage faster when the features are on a small scale. Let's normalize the counts.
# ### Train Tweets

# In[30]:


# Normalize using reprocessing.normalize
scaled_tweet_hashtag_count = preprocessing.normalize([train_tweets_final["tweet_hashtag_count"]])
train_tweets_final["tweet_hashtag_count"] = scaled_tweet_hashtag_count[0]

scaled_tweet_url_count = preprocessing.normalize([train_tweets_final["tweet_url_count"]])
train_tweets_final["tweet_url_count"] = scaled_tweet_url_count[0]

scaled_tweet_mention_count = preprocessing.normalize([train_tweets_final["tweet_mention_count"]])
train_tweets_final["tweet_mention_count"] = scaled_tweet_mention_count[0]
train_tweets_final.head()


# ### User

# In[31]:


users_final["user_like_count"] = preprocessing.normalize([users_final["user_like_count"]])[0]
users_final["user_followers_count"] = preprocessing.normalize([users_final["user_followers_count"]])[0]
users_final["user_following_count"] = preprocessing.normalize([users_final["user_following_count"]])[0]
users_final["user_listed_on_count"] = preprocessing.normalize([users_final["user_listed_on_count"]])[0]
users_final["user_tweet_count"] = preprocessing.normalize([users_final["user_tweet_count"]])[0]
users_final.head()


# # Feature Selection
# Fit a LASSO regression on our dataset and only consider features that have a conefficient different from 0. This reduce the number of features and helps the model generalize better for future datasets.
# ## Train Tweets Media

# In[32]:


print("train_tweets shape:", train_tweets.shape)
print("train_tweets_vectorized_media shape:", train_tweets_vectorized_media.shape)

# Match row number between train tweets and vectorized media
vectorized_media_df = pd.merge(train_tweets, train_tweets_vectorized_media, on='tweet_id', how='right')
# Drop extra columns
vectorized_media_df.drop(train_tweets.columns.difference(['virality']), axis=1, inplace=True)
vectorized_media_df.head()


# In[33]:


# Set the target as well as dependent variables from image data.
y = vectorized_media_df['virality']
x = vectorized_media_df.loc[:, vectorized_media_df.columns.str.contains("img_")] 

# Run Lasso regression for feature selection.
sel_model = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))

# time the model fitting
start = timeit.default_timer()

# Fit the trained model on our data
sel_model.fit(x, y)

stop = timeit.default_timer()
print('Time: ', stop - start) 

# get index of good features
sel_index = sel_model.get_support()

# count the no of columns selected
counter = collections.Counter(sel_model.get_support())
counter


# In[34]:


media_ind_df = pd.DataFrame(x[x.columns[(sel_index)]])
train_tweets_media_final = pd.concat([train_tweets_vectorized_media[['media_id', 'tweet_id']], media_ind_df], axis=1)
train_tweets_media_final.head()


# ## Train Tweets Text

# In[35]:


print("train_tweets shape:", train_tweets.shape)
print("train_tweets_vectorized_text:", train_tweets_vectorized_media.shape)

# Match row number between train tweets and vectorized text
vectorized_text_df = pd.merge(train_tweets, train_tweets_vectorized_text, on='tweet_id', how='right')
# Drop extra columns
vectorized_text_df.drop(train_tweets.columns.difference(['virality']), axis=1, inplace=True)
vectorized_text_df.head()


# In[36]:


# Set the target as well as dependent variables from image data.
y = vectorized_text_df['virality']
x = vectorized_text_df.loc[:, train_tweets_vectorized_text.columns.str.contains("feature_")] 

# time the model fitting
start = timeit.default_timer()

# Fit the trained model on our data
sel_model.fit(x, y)

stop = timeit.default_timer()
print('Time: ', stop - start) 

# get index of good features
sel_index = sel_model.get_support()

# count the no of columns selected
counter = collections.Counter(sel_model.get_support())
counter


# In[37]:


text_ind_df = pd.DataFrame(x[x.columns[(sel_index)]])
train_tweets_text_final = pd.concat([train_tweets_vectorized_text[['tweet_id']], text_ind_df], axis=1)
train_tweets_text_final.head()


# ## User Descriptions

# In[38]:


# Find the median virality for each user to reduce features
average_virality_df = train_tweets.groupby('tweet_user_id').agg(pd.Series.median)['virality']


# Obtain median of virality since each user may have multiple tweets.

# In[39]:


descriptions_df = pd.merge(average_virality_df, user_vectorized_descriptions, left_on='tweet_user_id', right_on='user_id', how='right')
descriptions_df.head()


# In[40]:


# Set the target as well as dependent variables from image data.
y = descriptions_df['virality']
x = descriptions_df.loc[:, descriptions_df.columns.str.contains("feature_")] 

# time the model fitting
start = timeit.default_timer()

# Fit the trained model on our data
sel_model.fit(x, y)

stop = timeit.default_timer()
print('Time: ', stop - start) 

# get index of good features
sel_index = sel_model.get_support()

# count the no of columns selected
counter = collections.Counter(sel_model.get_support())
counter


# In[41]:


desc_ind_df = pd.DataFrame(x[x.columns[(sel_index)]])
user_descriptions_final = pd.concat([user_vectorized_descriptions[['user_id']], desc_ind_df], axis=1)
user_descriptions_final.head()


# ## User Profile Images

# In[42]:


profile_images_df = pd.merge(average_virality_df, user_vectorized_profile_images, left_on='tweet_user_id', right_on='user_id', how='right')
profile_images_df.head()


# In[43]:


# Set the target as well as dependent variables from image data.
y = profile_images_df['virality']
x = profile_images_df.loc[:, profile_images_df.columns.str.contains("feature_")] 

# time the model fitting
start = timeit.default_timer()

# Fit the trained model on our data
sel_model.fit(x, y)

stop = timeit.default_timer()
print('Time: ', stop - start) 

# get index of good features
sel_index = sel_model.get_support()

# count the no of columns selected
counter = collections.Counter(sel_model.get_support())
counter


# In[44]:


user_prof_ind_df = pd.DataFrame(x[x.columns[(sel_index)]])
user_profile_images_final = pd.concat([user_vectorized_profile_images[['user_id']], user_prof_ind_df], axis=1)
user_profile_images_final.head()


# ## Join all tables together

# In[45]:


print("Shape:")
print("train_tweets:", train_tweets_final.shape)
print("train_tweets_media:", train_tweets_media_final.shape) # join on tweet id
print("train_tweets_text:", train_tweets_text_final.shape) # join on tweet id
print("")
print("user", users_final.shape) 
print("user_description", user_descriptions_final.shape) # join on user id
print("user_profile", user_profile_images_final.shape) # join on user id


# In[46]:


# tweets_vectorized_text and user_vectorized_profile_images has same column names. 
# rename columns in tweets_vectorized_text
cols = train_tweets_text_final.columns[train_tweets_text_final.columns.str.contains('feature_')]
train_tweets_text_final.rename(columns = dict(zip(cols, 'text_' + cols)), inplace=True)
train_tweets_text_final.head()


# In[47]:


# Group media by tweet_id (since there are multiple media id for a single tweet)
media_df = train_tweets_media_final.groupby('tweet_id').mean()


# In[48]:


# tweets_vectorized_text and user_vectorized_profile_images has same column names. 
# rename columns in tweets_vectorized_text
cols = train_tweets_text_final.columns[train_tweets_text_final.columns.str.contains('feature_')]
train_tweets_text_final.rename(columns = dict(zip(cols, 'text_' + cols)), inplace=True)
train_tweets_text_final.head()


# In[49]:


# Merge all tables on the column 'user_id' for user data and tweet_id for tweet data

# Join tweets data
tweet_df = pd.merge(media_df, train_tweets_text_final, on = 'tweet_id', how = 'right')
tweet_df.fillna(0, inplace=True)

# Join users data
user_df = pd.merge(users_final, user_profile_images_final, on='user_id')

# Join tweets data on train_tweets
tweet_df_final = pd.merge(train_tweets_final, tweet_df, on = 'tweet_id')

# Join with the users data
final_df = pd.merge(tweet_df_final, user_df, left_on = 'tweet_user_id', right_on='user_id')

final_df.shape


# # Preprocessing Test Data
# The preprocessing done on the train data is replicated on the test data, so that our model we train using our train data is usable for our test data.
# ## Test Tweets
# ### Missing Values

# In[50]:


test_tweets.isnull().sum()


# In[51]:


# Fill missing values as done in Train Tweets
test_tweets.fillna({'tweet_topic_ids':"['0']"}, inplace=True)


# ### Encoding

# In[52]:


# One hot Encoding
topic_ids = (
    test_tweets['tweet_topic_ids'].str.strip('[]').str.split('\s*,\s*').explode()
    .str.get_dummies().sum(level=0).add_prefix('topic_id_')
) 
topic_ids.rename(columns = lambda x: x.replace("'", ""), inplace=True)


# In[53]:


year = pd.get_dummies(test_tweets.tweet_created_at_year, prefix='year')
month = pd.get_dummies(test_tweets.tweet_created_at_month , prefix='month')
day = pd.get_dummies(test_tweets.tweet_created_at_day, prefix='day')
attachment = pd.get_dummies(test_tweets.tweet_attachment_class, prefix='attatchment')
language = pd.get_dummies(test_tweets.tweet_language_id, prefix='language')


# In[54]:


# Cyclical encoding
hour_sin = np.sin(2*np.pi*test_tweets['tweet_created_at_hour']/24.0)
hour_sin.name = 'hour_sin'
hour_cos = np.cos(2*np.pi*test_tweets['tweet_created_at_hour']/24.0)
hour_cos.name = 'hour_cos'


# In[55]:


columns_drop = [
                "tweet_topic_ids",
                "tweet_created_at_year",
                "tweet_created_at_month",
                "tweet_created_at_day",
                "tweet_attachment_class",
                "tweet_language_id",
                "tweet_created_at_hour",
              ]
dfs = [
        topic_ids,
        year,
        month,
        day,
        attachment,
        language,
        hour_sin,
        hour_cos,
      ]

test_tweets_final = test_tweets.drop(columns_drop, 1).join(dfs)
test_tweets_final.head()


# ### Missing Columns

# In[56]:


# Columns missing in train from test
cols_test = set(test_tweets_final.columns) - set(train_tweets_final.columns)
cols_test


# In[57]:


for col in cols_test:
    final_df[col] = 0


# In[58]:


# Columns missing in test from train
cols_train = set(train_tweets_final.columns) - set(test_tweets_final.columns)
cols_train.remove('virality') # remove virality from columns to add to test
cols_train


# In[59]:


for col in cols_train:
    test_tweets_final[col] = 0


# ### Join data

# In[60]:


test_tweets_media_final = pd.concat([test_tweets_vectorized_media[['media_id', 'tweet_id']], media_ind_df], axis=1)
test_tweets_text_final = pd.concat([test_tweets_vectorized_text[['tweet_id']], text_ind_df], axis=1)

media_df = test_tweets_media_final.groupby('tweet_id').mean()

cols = test_tweets_text_final.columns[test_tweets_text_final.columns.str.contains('feature_')]
test_tweets_text_final.rename(columns = dict(zip(cols, 'text_' + cols)), inplace=True)

# Join tweets data
tweet_df = pd.merge(media_df, test_tweets_text_final, on = 'tweet_id', how = 'right')
tweet_df.fillna(0, inplace=True)

# Join users data
user_df = pd.merge(users_final, user_profile_images_final, on='user_id')

# Join tweets data on train_tweets
tweet_df_final = pd.merge(test_tweets_final, tweet_df, on = 'tweet_id')

# Join with user data
p_final_df = pd.merge(tweet_df_final, user_df, left_on = 'tweet_user_id', right_on='user_id')

p_final_df.shape


# In[61]:


final_df.shape 


# Train has one more column than test because of virality column

# # Memory Footprint reduction.
# Function takes the minimum and the maximum of each column and changes the data type to what is optimal for the column. Implementation copied from [Eryk Lewson](https://towardsdatascience.com/make-working-with-large-dataframes-easier-at-least-for-your-memory-6f52b5f4b5c4)

# In[62]:


get_ipython().run_cell_magic('time', '', 'final_df = reduce_mem_usage(pd.read_csv("../input/temp-twitter-virality/final_df.csv"))\np_final_df = reduce_mem_usage(pd.read_csv("../input/temp-twitter-virality/p_final_df.csv"))\nprint("Shape of train set: ", final_df.shape)\nprint("Shape of test set: ", p_final_df.shape)')


# # Model fitting
# ## Split the full sample into train/test (70/30)

# In[63]:


X = final_df.drop(['virality', 'tweet_user_id', 'tweet_id', 'user_id'], axis=1)
y = final_df['virality']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=314, stratify=y)
print('Training set shape ', X_train.shape)
print('Test set shape ', X_test.shape)


# ## Hyperparameter tuning

# In[64]:


# param_test = {'num_leaves': sp_randint(6, 50), 
#             'min_child_samples': sp_randint(100, 500), 
#             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
#             'subsample': sp_uniform(loc=0.2, scale=0.8), 
#             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
#             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
#             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

#clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)
#gs = RandomizedSearchCV(
#    estimator=clf, param_distributions=param_test, 
#    n_iter=100,
#    scoring= 'f1_macro',
#    cv=3,
#    refit=True,
#    random_state=314,
#    verbose=True)

#gs.fit(X_train, y_train, **fit_params)
#print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))


# Best score reached: 0.48236216974224616)  
# with params: {  
# 'colsample_bytree': 0.7076074093370144,   
# 'min_child_samples': 105,   
# 'min_child_weight': 1e-05,   
# 'num_leaves': 26,   
# 'reg_alpha': 5,   
# 'reg_lambda': 5,   
# 'subsample': 0.7468773130235173  
# } 

# In[65]:


opt_params = {'num_leaves': 26,
             'min_child_samples': 105,
             'min_child_weight': 1e-05,
             'subsample': 0.7468773130235173,
             'colsample_bytree': 0.7076074093370144,
             'reg_alpha': 5,
             'reg_lambda': 5
             }


# In[66]:


clf = lgb.LGBMClassifier(**opt_params)
clf.fit(
    X_train, y_train, 
    eval_set=[(X_train, y_train), (X_test, y_test)],
    early_stopping_rounds=10
)


# In[67]:


# Prediction on the test dataset
y_pred = clf.predict(X_test)

# Base accuracy 66.45%
# 0.6849 LGBMClassifier(max_depth=12, num_leaves=300)
print('Accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))


# # Identify feature importance

# In[68]:


feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,X.columns)), columns=['Value','Feature'])
plt.figure(figsize=(10, 5))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:10], color='blue')
plt.show()


# # Fit model to Test data

# In[69]:


X = p_final_df.drop(['tweet_user_id', 'tweet_id', 'user_id'], axis=1)

solution = clf.predict(X)
solution_df = pd.concat([p_final_df[['tweet_id']], pd.DataFrame(solution, columns = ['virality'])], axis=1)
solution_df.head()


# In[70]:


#solutions_format = pd.read_csv("../input/viral-tweets/Dataset/solution_format.csv")
solutions_format = solutions_format.drop(["virality"], axis=1)
final_solution = solutions_format.merge(solution_df, left_on='tweet_id', right_on='tweet_id')
final_solution


# In[71]:


final_solution.to_csv("final_solution.csv", index=False)


# # Next Steps
# - More feature engineering
# - Further parameter tuning
# - Stacking ensemble ML models
# - Learning rate decay in LightGBM model training to improve convergence to the minimum
