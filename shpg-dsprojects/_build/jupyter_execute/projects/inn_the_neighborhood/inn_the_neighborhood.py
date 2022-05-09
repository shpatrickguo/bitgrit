#!/usr/bin/env python
# coding: utf-8

# # DataCamp Certification Case Study
# 
# ## Project Brief
# 
# You have been hired by Inn the Neighborhood, an online platform that allows people to rent out their properties for short stays. Currently, the webpage for renters has a conversion rate of 2%. This means that most people leave the platform without signing up. 
# 
# The product manager would like to increase this conversion rate. They are interested in developing an application to help people estimate the money they could earn renting out their living space. They hope that this would make people more likely to sign up.
# 
# The company has provided you with a dataset that includes details about each property rented, as well as the price charged per night. They want to avoid estimating prices that are more than 25 dollars off of the actual price, as this may discourage people.
# 
# You will need to present your findings in two formats:
# - You must submit a written report summarising your analysis to your manager. As a data science manager, your manager has a strong technical background and wants to understand what you have done and why. 
# - You will then need to share your findings with the product manager in a 10 minute presentation. The product manager has no data science background but is familiar with basic data related terminology. 
# 
# The data you will use for this analysis can be accessed here: `"data/rentals.csv"`

# ## The Data
# - `id`: Numeric, the unique identification number of the property
# - `latitude`: Numeric, the latitude of the property
# - `longitude`: Numeric, the longitude of the property
# - `property_type`: Character, the type of property (e.g., apartment, house, etc)
# - `room_type`: Character, the type of room (e.g., private room, entire home, etc)
# - `bathrooms`: Numeric, the number of bathrooms
# - `bedrooms`: Numeric, the number of bedrooms
# - `minimum_nights`: Numeric, the minimum number of nights someone can book
# - `price`: Character, the dollars per night charged

# ## Import Libraries

# In[1]:


# Installations
#!pip install geopy lightgbm xgboost verstack --upgrade


# In[2]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from verstack import LGBMTuner
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ## Load the data

# In[4]:


rentals_csv = pd.read_csv('data/rentals.csv')
df = rentals_csv.copy()
df.head()


# ## Data Size and Structure
# - Dataset comprises of 8111 observations and 7 characteristics.
# - `id` can be treated as the index
# - 7 independent variables: `latitude`, `longitude`, `property_type`, `room_type`, `bathrooms`, `bedrooms`, `minimum_nights`
#     * 5 numerical features, 2 categorical features
# - 1 dependent variable: `price`
# - `price` is treated as `object` type because of the non-numeric characters in cell, e.g."$", ",". 
# - 2 variable columns has null/missing values.

# In[5]:


df.info()


# In[6]:


# Set id as index
df.set_index('id', inplace=True)


# In[7]:


def remove_nonnumeric(string):
   """Remove non-numeric characters from string using regex.
   
   INPUTS:
   string - str
   
   OUTPUT
   output - str
   """
   output = re.sub("[^0-9]", "", string)
   return output

# Remove non-numeric characters
df.price = df.price.apply(lambda x: remove_nonnumeric(x))

# Change dtype of price from str to float64 
df.price = df.price.astype('float64') / 100


# ### Summary Statistics
# - Features have different scales and units.
# - There is a narrow distribution of latitude and longitude values, which suggest the dataset contains properties in the same region.
# - There are extreme outliers in `minimum_nights`, with a max value of `100000000` nights, which would be over 273972 years.

# In[8]:


df.describe()


# In[9]:


# Remove outliers in columns minimum_nights and price using IQR
quant_cols = ["minimum_nights", "price"]

for col_name in quant_cols:
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    df = df[df[col_name] < high]
    df = df[df[col_name] > low]
df.describe()


# ### Completeness of Data
# - Less than 0.2% of missing values present in columns bathroom and bedrooms
# - Since, the missing observations only account for a small amount of the dataset, we can drop these observations.

# In[10]:


# Percentage of dataset is null
df.isnull().sum() / len(df) * 100


# In[11]:


# Drop null values
df = df.dropna()


# ## Data Cleaning Summary
# - Set `id` as index.
# - Removed non-numeric characters from `price`.
# - Dropped extreme outliers in `minimum_nights` and `price`.
# - Drop rows with null values.

# ## Exploratory Data Analysis

# ### Correlation
# - There is a positive correlation between number of bathrooms and bedrooms.
# - Price has the highest positive correlation with the number of bedrooms.
# - Price has a negative correlation with the number of bathrooms and minimum nights.
# - There is no strong correlation between variables. 

# In[12]:


# Visualizing the correlations between numerical variables
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="RdBu", annot=True)
plt.title("Correlations Between Variables", size=15)
plt.show()


# ### Price
# Our target variable `price` is a continuous numerical variable. <br>
# The distribution is right skewed, with most properties costing around $100.00 per night. <br>
# Recall that outliers have already been removed from `price`.

# In[13]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,8))
sns.boxplot(x='price', data=df, ax=ax1)
sns.histplot(x='price', data=df, ax=ax2, kde=True)
ax1.set_title('Box-plot of Price')
ax2.set_title('Histplot of Price')
ax1.set_xlabel('Price ($) per night')
ax2.set_xlabel('Price ($) per night')
plt.show()


# ### Bedrooms
# `bedrooms` is a discrete numerical variable. It has the highest positive correlation with `price`. <br>
# The distribution is right skewed, with most properties having 1 bathroom.
# Different variability of the price is present for each number of bedrooms.

# In[14]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,8))
sns.boxplot(x='bedrooms', data=df, ax=ax1)
sns.countplot(x='bedrooms', data=df, ax=ax2)
ax1.set_title('Box-plot of no. of Bedrooms')
ax2.set_title('Countplot of no. of Bedrooms')
ax1.set_xlabel('Bedrooms')
ax2.set_xlabel('Bedrooms')
plt.show()


# In[15]:


# Price corresponding to no. of Bedrooms
plt.figure(figsize=(10,8))
ax = sns.barplot(y='price', x='bedrooms', data=df)
ax.set_title('Price ($) vs no. of Bedrooms')
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price ($) per night')
plt.show()


# ### Bathrooms
# `bathrooms` is a discrete numerical variable. It has a weak positive correlation with bedrooms.<br>
# **Contextual information:** A full bathroom has a sink, toilet, and either a tub/shower combo or separate tub and shower. Whereas, a half-bath has a sink and a toilet - no tub or shower.<br>
# The distribution is right skewed, with most properties having 1 bathroom. Different variability of the price is present for each number of bathrooms.

# In[16]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,8))
sns.boxplot(x='bathrooms', data=df, ax=ax1)
sns.countplot(x='bathrooms', data=df, ax=ax2)
ax1.set_title('Box-plot of no. of Bathrooms')
ax2.set_title('Countplot of no. of Bathrooms')
ax1.set_xlabel('Bathrooms')
ax2.set_xlabel('Bathrooms')
plt.show()


# In[17]:


# Price corresponding to no. of Bathrooms
plt.figure(figsize=(10,8))
ax = sns.barplot(y='price', x='bathrooms', data=df)
ax.set_title('Price ($) vs no. of Bathrooms')
ax.set_xlabel('Bathrooms')
ax.set_ylabel('Price ($) per night')
plt.show()


# ### Minimum Nights
# `minimum_nights` is a discrete numerical variable. <br>
# The distribution is right skewed and bimodal, with most properties requiring a minimum of 30 or 1-3 nights of stay. <br>
# Recall that outliers have already been removed from `Minimum Nights`.

# In[18]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,8))
sns.boxplot(x='minimum_nights', data=df, ax=ax1)
sns.countplot(x='minimum_nights', data=df, ax=ax2)
ax1.set_title('Box-plot of no. of Minimum Nights')
ax2.set_title('Countplot of no. of Minimum Nights')
ax1.set_xlabel('Minimum Nights')
ax2.set_xlabel('Minimum Nights')
plt.show()


# ### Property Type
# `property_type` is a categorical variable. There are 26 property types.<br>
# Most properties are Apartments. The most expensive type being resorts, and the cheapest being Camper/RV. <br>
# Different variability of the price is present for each property type. For example, cottages has a wide range of prices, whereas apartments have a narrow range.

# In[19]:


print(f"No. of property types: {len(df['property_type'].unique())}")

# Frequency of each property type
plt.figure(figsize=(10,8))
ax = sns.countplot(y='property_type', data=df, order = df['property_type'].value_counts().index)
ax.set_title('Frequency of Property Types')
ax.set_ylabel('Property Type')
plt.show()


# In[20]:


# Price of each property type
plt.figure(figsize=(10,8))
ax = sns.barplot(x='price', y='property_type', data=df, order = df['property_type'].value_counts().index)
ax.set_title('Price vs Property Types')
ax.set_ylabel('Property Type')
ax.set_xlabel('Price ($) per night')
plt.show()


# ### Room Type
# `room_type` is a categorical variable. There are 4 room types. <br>
# Most rooms are Entire home/apt. The most expensive type being Entire home/apt, and the cheapest being Shared Room.

# In[21]:


print(f"No. of room types: {len(df['room_type'].unique())}")

# Frequency of each room type
plt.figure(figsize=(10,8))
ax = sns.countplot(y='room_type', data=df, order = df['room_type'].value_counts().index)
ax.set_title('Frequency of Room Types')
ax.set_ylabel('Room Type')
plt.show()


# In[22]:


# Price of each room type
plt.figure(figsize=(10,8))
ax = sns.barplot(x='price', y='room_type', data=df, order = df['room_type'].value_counts().index)
ax.set_title('Price vs Room Types')
ax.set_ylabel('Room Type')
ax.set_xlabel('Price ($) per night')
plt.show()


# ### Latitude and Longitude
# Properties are located near San Francisco. <br>
# Scatterplot does not show distinct price clusters based on latitude and longitude.

# In[23]:


# Get address from Latitude and Longitude
geolocator = Nominatim(user_agent="geoapi")
sample_df = df.sample(1)
Latitude, Longitude = sample_df['latitude'].astype(str), sample_df['longitude'].astype(str)
location = geolocator.reverse(Latitude + "," + Longitude)
print(location)

# Visualization of latitude and longitude
plt.figure(figsize=(10,8))
sns.scatterplot(data=df, x='longitude', y='latitude', hue='price')
plt.title('Scatterplot of Latitude and Longitude, colored by Price')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# ## Train Test Split
# Split the data to train (80%) and test (20%) to estimate the performance of machine learning algorithms when they are used to make predictions on data not used to train the model.

# In[24]:


X = df.iloc[:, :-1] # Features
y = df['price'] # Target


# In[25]:


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022)

print('X train data {}'.format(X_train.shape))
print('y train data {}'.format(y_train.shape))
print('X test data  {}'.format(X_test.shape))
print('y test data  {}'.format(y_test.shape))


# ## Feature Engineering
# Performed after Train Test split to prevent information leakage from the test set.
# - Convert categorical variables into a format that can be readily used by machine learning algorithms using One-Hot-Encoding.
# - Cartesian coordinates into polar coordinates
# - Scale input variables to improve the convergence of algorithms that do not possess the property of scale invariance.

# ### One-Hot-Encoding

# In[26]:


# Categorical Features
cat_cols = ['property_type', 'room_type']

# Transformer Object
ohe = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), 
     cat_cols))

def encode(transformer, df):
    """ Apply transformer columns of a pandas DataFrame
    
    INPUTS:
    transformer: transformer object
    df: pandas DataFrame
    
    OUTPUT:
    transformed_df: transformed pandas DataFrame
    
    """
    transformed = transformer.fit_transform(df[cat_cols]).toarray()
    transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names())
    return transformed_df

# Apply transformation on both Train and Test set
encoded_train = encode(ohe, X_train)
encoded_train.index = X_train.index
encoded_test = encode(ohe, X_test)
encoded_test.index = X_test.index

print(f"encoded_train shape: {encoded_train.shape}")
print(f"encoded_test shape: {encoded_test.shape}")

# Join encoded cols to Train and Test set, drop original column
X_train = X_train.join(encoded_train)
X_train.drop(cat_cols, axis=1, inplace=True)
X_test = X_test.join(encoded_test)
X_test.drop(cat_cols, axis=1, inplace=True)

# Ensure Train and Test have the same no. of columns
train_cols = list(X_train.columns)
test_cols = list(X_test.columns)
cols_not_in_test = {c:0 for c in train_cols if c not in test_cols}
X_test = X_test.assign(**cols_not_in_test)
cols_not_in_train = {c:0 for c in test_cols if c not in train_cols}
X_train = X_train.assign(**cols_not_in_train)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# ### Geospaital Features
# Transform Cartesian coordinates into polar coordinates.

# In[27]:


# Geospatial Features
geo_cols = ['latitude', 'longitude']

def cart2pol(x, y):
    """ Convert cartesian coordinates into polar coordinates
    
    INPUT
    x, y: Cartesian coorindates
    
    OUTPUT
    rho, phi: Polar coordinates
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

rho, phi = cart2pol(X_train['longitude'], X_train['latitude'])
X_train['rho'] = rho
X_train['phi'] = phi
rho, phi = cart2pol(X_test['longitude'], X_test['latitude'])
X_test['rho'] = rho
X_test['phi'] = phi

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# ## Variable Importance with Random Forest
# Using a simple random forest with 100 trees to get a quick overview of the top 5 relative permutation importance for each variable. <br>
# According to the random forest, `bedrooms` is the most important feature. This is expected because it has the highest correlation to the `price`.

# In[28]:


rf = RandomForestRegressor(n_estimators=100, random_state=2022)
rf.fit(X_train, y_train)
perm_importance = permutation_importance(rf, X_test, y_test)


# In[29]:


plt.figure(figsize=(10,8))
sorted_idx = perm_importance.importances_mean.argsort()[-5:]
plt.barh(X_train.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.title("Random Forest Feature Importance")
plt.show()


# ## Feature Selection
# Recognize that many new features were generated during feature engineering. <br>
# Using Lasso Regularization to identify and select a subset of input variables that are most relevant to the target variable. <br>
# Non-important features are removed.

# In[30]:


# Lasso Regularization
sel_lasso = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear', random_state=2022))
sel_lasso.fit(X_train, y_train)
sel_lasso.get_support()


# In[31]:


# make a list with the selected features and print the outputs
selected_feat = X_train.columns[(sel_lasso.get_support())]

print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(np.sum(sel_lasso.estimator_.coef_ == 0)))


# In[32]:


X_train = X_train[selected_feat]
X_test = X_test[selected_feat]
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# ## Model Selection
# Explore different supervised regression models since the target variable is labeled and continuous. <br>
# Using k-folds cross validation to estimate and compare the performance of models on out-of-sample data. Identify, which model is worth improving upon. <br>
# LGBMRegressor - (Microsoft’s implementation of gradient boosted machines) - gives the best results out of models.

# In[33]:


def neural_network():
    # No. of neuron based on the number of available features
    model = Sequential()
    model.add(Dense(26,activation='relu'))
    model.add(Dense(26,activation='relu'))
    model.add(Dense(26,activation='relu'))
    model.add(Dense(26,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='Adam',loss='mse')
    return model


# In[34]:


# Pipelines for Machine Learning models
pipelines = []
pipelines.append(('Linear Regression', Pipeline([('scaler', MinMaxScaler()), ('LR', LinearRegression())])))
pipelines.append(('KNN Regressor', Pipeline([('scaler', MinMaxScaler()), ('KNNR', KNeighborsRegressor())])))
pipelines.append(('SupportVectorRegressor', Pipeline([('scaler', MinMaxScaler()), ('SVR', SVR())])))
pipelines.append(('DecisionTreeRegressor', Pipeline([('scaler', MinMaxScaler()), ('DTR', DecisionTreeRegressor())])))
pipelines.append(('AdaboostRegressor', Pipeline([('scaler', MinMaxScaler()), ('ABR', AdaBoostRegressor())])))
pipelines.append(('RandomForestRegressor', Pipeline([('scaler', MinMaxScaler()), ('RBR', RandomForestRegressor())])))
pipelines.append(('BaggingRegressor', Pipeline([('scaler', MinMaxScaler()), ('BGR', BaggingRegressor())])))
pipelines.append(('GradientBoostRegressor', Pipeline([('scaler', MinMaxScaler()), ('GBR', GradientBoostingRegressor())])))
pipelines.append(('LGBMRegressor', Pipeline([('scaler', MinMaxScaler()), ('lightGBM', LGBMRegressor())])))
pipelines.append(('XGBRegressor', Pipeline([('scaler', MinMaxScaler()), ('XGB', XGBRegressor())])))
pipelines.append(('Neural Network', Pipeline([('scaler', MinMaxScaler()), ('NN', neural_network())])))


# In[35]:


# Create empty dataframe to store the results
cv_scores = pd.DataFrame({'Regressor':[], 'RMSE':[], 'Std':[]})

# Cross-validation score for each pipeline for training data
for ind, val in enumerate(pipelines):
    name, pipeline = val
    kfold = KFold(n_splits=10) 
    rmse = np.sqrt(-cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring="neg_mean_squared_error"))
    cv_scores.loc[ind] = [name, rmse.mean(), rmse.std()]
cv_scores


# ## Model Tuning
# Using LGBMTuner from verstack - automated lightgbm models tuner with optuna (automated hyperparameter optimization framework). <br>
# Using Root Mean Squared Error as the metric for 500 trials.
# 
# According to the tuned model:
# - The most important hyperparameter is `num_leaves`: max number of leaves in one tree.
# - The features with the most impact to the model are the geographic features `phi`, `latitude`, `rho`, `longitude`. Suggesting the location is a major factor in dictating price per night.

# In[36]:


tuner = LGBMTuner(metric = 'rmse', trials = 500)
tuner.fit(X_train, y_train)


# In[37]:


# Best Performing Model
LGBMRegressor(**tuner.best_params)


# In[38]:


y_pred = tuner.predict(X_test)
y_pred


# ## Model Evaluation
# Measure model performance using Mean Squared Error, Mean Absolute Error, Root Mean Squared Error, and r2 Score.
# 
# $$ MSE(y, \hat{y}) = \frac{1}{N_{samples}} \sum_{i=0}^{n_{samples} - 1} (y_i - \hat{y_i})^2 $$
# $$ MAE(y, \hat{y}) = \frac{1}{N_{samples}} \sum_{i=0}^{n_{samples} - 1} | y_i - \hat{y_i} | $$
# $$ R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y_i})^2}{\sum_{i=1}^{n} (y_i - \bar{y_i})^2} $$
# 
# Hyperparameter tuning improved the model's RMSE from the base model. <br>
# Nevertheless, the RMSE scores is greater than 25 indicating that the model fails to predict price within 25 dollars off of the actual price.

# In[39]:


performance_dict = {'MSE': mean_squared_error(y_test, y_pred), 
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'MAE': mean_absolute_error(y_test, y_pred), 
                    'R2 Score': r2_score(y_test, y_pred)
                   }

performance_df = pd.DataFrame(performance_dict.items(), columns=['Metric', 'Score'])
performance_df


# ## Outcome Summary
# Aimed to predict price, allowing renters to estimate how much they could earn renting out their living space. <br>
# Adopted Supervised Machine Learning approach to predict labeled continuous target variable `price` using details about each property rented, as well as the price charged per night. <br>
# After model selection and tuning, the best model: 
# ```
# LGBMRegressor(colsample_bytree=0.6965557560964968, learning_rate=0.01,
#               metric='rmse', min_sum_hessian_in_leaf=0.37294528370449115,
#               n_estimators=314, num_leaves=142, num_threads=14,
#               objective='regression', random_state=42,
#               reg_alpha=6.532407584502388, reg_lambda=0.31928270201641645,
#               subsample=0.5790593334642422, verbosity=-1)
# ```
# The most important features of the model:
# - geospaital features e.g.(`phi`, `latitude`, `rho`, `longitude`) <br>
# Interesting considering `price` has the highest correlation with no. of `bedrooms`.
# 
# Resulting with a $r^2$ Score of ~0.61 and ~53.75 RMSE. Though, failing to reach target of estimating prices that are less than 25 dollars off of the actual price.

# ## Next steps
# To improve the model and reach the target of estimating prices without being more than 25 dollars off:
# - More data collection to increase training size.
# - Record more features about the living space, e.g. amenities (parking, appliances, etc.), geographic features (nearby public transportation, etc.).
# - Expand upon feature engineering and hyperparameter tuning.
# - Use more complex models or ensemble methods, e.g. blending or stacking models.

# In[ ]:




