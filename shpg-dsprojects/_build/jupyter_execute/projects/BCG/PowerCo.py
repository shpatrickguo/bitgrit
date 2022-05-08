#!/usr/bin/env python
# coding: utf-8

# # Power Co - Test whether churn is driven by price sensitivity.

# # 1: Business Understanding & Problem Framing
# 
# ## Objective
# 
# Formulate the hypothesis as a data science problem and lay out the major steps needed to test this hypothesis. Communicate your thoughts and findings in an email to your LDS, focusing on the potential data that you would need from the client and analytical models you would use to test such a hypothesis.
# 
# ## Background Information
# 
# PowerCo is a major gas and electricity utility that supplies to corporate, SME (Small & Medium Enterprise), and residential customers. The power-liberalization of the energy market in Europe has led to significant customer churn, especially in the SME segment. They have partnered with BCG to help diagnose the source of churning SME customers. 
# 
# One of the hypotheses under consideration is that churn is driven by the customers’ price sensitivities and that it is possible to predict customers likely to churn using a predictive model. The client also wants to try a discounting strategy, with the head of the SME division suggesting that offering customers at high propensity to churn a 20% discount might be effective.
# 
# The Lead Data Scientist (LDS) held an initial team meeting to discuss various hypotheses, including churn due to price sensitivity. After discussion with your team, you have been asked to go deeper on the hypothesis that the churn is driven by the customers’ price sensitivities.
# 
# Your LDS wants an email with your thoughts on how the team should go about to test this hypothesis.

# ## Email
# 
# **Subject:** PowerCo: Test whether churn is driven by price sensitivity.
# 
# Dear Iman Karimi,
# 
# To test client’s hypothesis: whether churn is driven by price sensitivity.We will need to predict customer’s likelihood to churn and understand the effect prices have on churn rates.
# 
# ### Data needed:
# 
# - **Customer data:** characteristics of each client i.e. previous energy
# consumption, enterprise etc.
# - **Price data:** historical prices the client charge to customers over time
# - **Churn data:** whether customer has churned.
# 
# ### Workflow (iterative process):
# 
# 1. Data cleaning: modifying/removing any incorrect, incomplete, irrelevant data.
# 2. Data preprocessing: normalization/standardization for better convergence, encoding categorical values, feature selection by whittling
# down predictors to a smaller set that is more informative.
# 3. Build Binary classification model: (e.g. Logistic regression, random forests, KNN, neural nets) pick the appropriate model after evaluating tradeoff between accuracy, complexity, and interpretability.
# 4. Model validation: Split data into training, validation, and test and score model performance.
# 
# From the model, we will understand the degree of impact prices have on churn rates. This will help us assess the effectiveness of the client’s discounting strategy.
# 
# ### Context:
# 
# Client is a major gas and electricity utility.
# - Supplies to corporate, SME, and residential customers.
# 
# Significant churn problem.
# - Drive by power-liberalization of the energy market in Europe
# - Problem is largest in the SME segment
# 
# #### Client’s Hypothesis
# 
# Churn is driven by price sensitivity.
# Client wants to try a discounting strategy by offering customers at high propensity to churn a 20% discount
# 
# Regards, <br>
# Patrick Guo

# # 2: Exploratory Data Analysis & Data Cleaning
# 
# ## Objective
# 
# Clean the data – address missing values, duplicates, data type conversions, transformations, and multi-co-linearity, as well as outliers.
# 
# Perform some exploratory data analysis. Look into the data types, data statistics, and identify any missing data or null values, and how often they appear in the data. Visualize specific parameters as well as variable distributions.
# 
# ## Background Information
# 
# The BCG project team thinks that building a churn model to understand whether price sensitivity is the largest driver of churn has potential. The client has sent over some data and the LDS wants you to perform some exploratory data analysis and data cleaning.
# 
# The data that was sent over includes:
# 
# Historical customer data: Customer data such as usage, sign up date, forecasted usage etc
# Historical pricing data: variable and fixed pricing data etc
# Churn indicator: whether each customer has churned or not
# These datasets are otherwise identical and have historical price data and customer data (including churn status for the customers in the training data).

# ## The Datasets

# ```ml_case_training_output.csv``` named as ```pco_output``` contains:
# - id: contact id 
# - churned: has the client churned over the next 3 months
# 
# ```ml_case_training_hist_data.csv``` named as ```pco_hist``` contains the history of energy and power consumption per client:
# - id: contact id 
# - price_date: reference date
# - price_p1_var: price of energy for the 1st period 
# - price_p2_var: price of energy for the 2nd 
# - periodprice_p3_var: price of energy for the 3rd period 
# - price_p1_fix: price of power for the 1st period
# - price_p2_fix: price of power for the 2nd period 
# - price_p3_fix: price of power for the 3rd period
# 
# ```ml_case_training_data.csv``` contains:
# 
# - id: contact id
# - activity_new: category of the company's activity. 
# - campaign_disc_elec: code of the electricity campaign the customer last subscribed to.
# - channel_sales: code of the sales channel
# - cons_12m: electricity consumption of the past 12 months
# - cons_gas_12m: gas consumption of the past 12 months
# - cons_last_month: electricity consupmtion of the last month
# - date_activ: date of activation of the contract
# - date_end: registered date of the end of the contract
# - date_first_activ: date of first contract of the client
# - date_modif_prod: date of last modification of the product
# - date_renewal: date of the next contract renewal
# - forecast_base_bill_ele: forecasted electricity bill baseline for next month
# - forecast_base_bill_year: forecasted electricity bill baseline for calendar year
# - forecast_bill_12m: forecasted electricity bill baseline for 12 months
# - forecast_cons: forecasted electricity consumption for next month
# - forecast_cons_12m: forecasted electricity consumption for next 12 months
# - forecast_cons_year: forecasted electricity consumption for next calendar year
# - forecast_discount_energy: forecasted value of current discount
# - forecast_meter_rent_12m: forecasted bill of meter rental for the next 12 months
# - forecast_price_energy_p1: forecasted energy price for 1st period
# - forecast_price_energy_p2: forecasted energy price for 2nd period
# - forecast_price_pow_p1: forecasted power price for 1st period
# - has_gas: indicated if client is also a gas client
# - imp_cons: current paid consumption
# - margin_gross_pow_ele: gross margin on power subscription
# - margin_net_pow_ele: net margin on power subscription
# - nb_prod_act: number of active products and services
# - net_margin: total net margin
# - num_years_antig: antiquity of the client (in number of years)
# - origin_up: code of the electricity campaign the customer first subscribed to
# - pow_max: subscribed power

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
# Show plots in jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

import missingno as msno
from scipy.stats import zscore as zscore

import warnings
warnings.filterwarnings("ignore")

# Set maximum number of columns to be displayed
pd.set_option('display.max_columns', 100)


# ## Load Data

# In[2]:


# list of dates
dt_lst = ['date_activ','date_end','date_first_activ','date_modif_prod','date_renewal']
     
pco_main = pd.read_csv('data/ml_case_training_data.csv', parse_dates=dt_lst)
pco_hist = pd.read_csv('data/ml_case_training_hist_data.csv', parse_dates=['price_date'])
pco_output = pd.read_csv('data/ml_case_training_output.csv')
pd.set_option('display.max_columns',None)


# ## Main Dataset

# In[3]:


pco_main.head()


# In[4]:


pco_main.info()


# In[5]:


# Percentage of nullity by column
missing_perc = pco_main.isnull().mean() * 100
print('Percentage of Missing Values:\n', missing_perc)


# In[6]:


# Descriptive statistics
pco_main.describe()


# ### Observations
# - 14 columns have negative minimum values.
# - ```campaign_disc_ele``` column is missing completely.
# - ```activity_new``` column is missing 59.3%.
# - The ```date_first_active```, ```forecast_base_bill_ele```, ```forecast_base_bill_year```, ```forecast_bill_12m```, and ```forecast_cons``` columns are each missing 78.2%.

# ## The History Dataset

# In[7]:


pco_hist.head()


# In[8]:


pco_hist.info()


# In[9]:


# Percentage of nullity by column
missing_perc = pco_hist.isnull().mean() * 100
print('Percentage of Missing Values:\n', missing_perc)


# In[10]:


# Descriptive statistics
pco_hist.describe()


# ### Observations

# - ```price_p1_var```, ```price_p2_var```, ```price_p3_var```, ```price_p1_fix```, ```price_p2_fix```, ```price_p3_fix``` are missing 70.4% values.
# - ```price_p1_fix```, ```price_p2_fix```, ```price_p3_fix``` contain negative values, which doesn't make sense for price of power.

# ## The Output Dataset

# In[11]:


pco_output.head()


# In[12]:


pco_output.info()


# In[13]:


# Percentage of nullity by column
missing_perc = pco_output.isnull().mean() * 100
print('Percentage of Missing Values:\n', missing_perc)


# In[14]:


# Descriptive statistics
pco_output.describe()


# ### Observations

# - Complete dataset.

# ## Data Cleaning and Imputation
# ### Missing Data
# 
# ### Types of missingness
# 
# **Missing Completely at Random (MCAR)** <br>
# Missingness has no relationship between any values, observed or missing
# 
# **Missing at Random (MAR)** <br>
# There is a systematic relationship between missingness and other observed data, but not the missing data
# 
# **Missing Not at Random (MNAR)** <br>
# There is a relationship between missingness and its values, missing or non-missing

# ### The History Dataset

# In[15]:


# Identify negative columns
negative_cols = ['price_p1_fix','price_p2_fix','price_p3_fix']
# Convert to positive the negative columns in pco_hist
pco_hist[negative_cols] = pco_hist[negative_cols].apply(abs)

pco_hist.describe()


# In[16]:


# Visualize the completeness of the dataframe
msno.bar(pco_hist)
plt.show()


# In[17]:


# Visualize the locations of the missing values of the dataset
sorted = pco_hist.sort_values(by = ['id','price_date'])
msno.matrix(sorted)
plt.show()


# In[18]:


# Visualize the correlation between the numeric variables of the dataframe
msno.heatmap(pco_hist)
plt.show()


# In[19]:


# Identify the index of the IDs containing missing values.
hist_NAN_index = pco_hist[pco_hist.isnull().any(axis=1)].index.values.tolist()

# Obtain a dataframe with the missing values
pco_hist_missing = pco_hist.iloc[hist_NAN_index,:]

# Glimpse at the NaN cases of the pco_hist dataset
pco_hist_missing.head(10)


# In[20]:


# extract the unique dates of missing data
date_lst = pco_hist_missing['price_date'].unique()
id_lst = pco_hist_missing['id'].unique()

# Create a time dataframe with the unique dates
time_df = pd.DataFrame(data=date_lst, columns=['price_date'] )

# Glimpse the time dataframe
time_df.sort_values(by=['price_date'])


# #### Observations
# The columns containing prices display strong positive correlation in the missingness, suggesting a case of **MNAR**.
# 
# We can use trend and cyclicality when imputing time series data.

# In[21]:


# Make a copy of pco_hist dataset
pco_hist_ff = pco_hist.copy(deep=True)

# Print prior to imputing missing values
print(pco_hist_ff.iloc[hist_NAN_index,3:9].head())

# Fill NaNs using forward fill
pco_hist_ff.fillna(method = 'ffill', inplace=True)

print(pco_hist_ff.iloc[hist_NAN_index,3:9].head())


# In[22]:


# Merge output dataset with historical forward fill dataset
pco_hist_ff_merged = pco_hist_ff.merge(right=pco_output,on=['id'])
pco_hist_ff_merged.head()


# ### The Main Dataset

# In[23]:


# Visualize the completeness of the dataframe
msno.bar(pco_main)
plt.show()


# In[24]:


# Visualize the locations of the missing values of the dataset
msno.matrix(pco_main)
plt.show()


# In[25]:


msno.heatmap(pco_main)
plt.show()


# In[26]:


# Demonstrate why the date_activ column cannot replace completely date_first_activ
activity = ['date_activ','date_first_activ']

# Filter the columns of interest
pco_activity = pco_main[activity]

# Obtain only the complete cases
pco_activity_cc = pco_activity.dropna(subset=['date_first_activ'],how='any',inplace=False)

# Test whether two objects contain the same elements.
pco_activity_cc.date_activ.equals(pco_activity_cc.date_first_activ)

# Describe it
pco_activity_cc.describe(datetime_is_numeric=True)


# In[27]:


# Drop the column activity_new and campaign_disc_elec
pco_main_drop = pco_main.drop(labels= ['activity_new','campaign_disc_ele'] , axis=1)

# Remove date_end date_modif_prod date_renewal origin_up pow_max margin_gross_pow_ele margin_net_pow_ele net_margin
brush = ['date_end','date_modif_prod','date_renewal','origin_up','pow_max','margin_gross_pow_ele',
         'margin_net_pow_ele', 'net_margin','forecast_discount_energy','forecast_price_energy_p1',
         'forecast_price_energy_p2','forecast_price_pow_p1']
pco_main_drop.dropna(subset=brush, how='any',inplace=True)

msno.matrix(pco_main_drop)
plt.show()


# #### Observations

# - ```activity_new``` is **MCAR** with low correlation with other variables. Can drop this column
# - ```campaign_disc_elec``` is **MCAR**. Can drop this column. Suggests that subscribers are not subscribing through campaign offers.
# - ```date_first_activ``` cannot replace ```date_active```. **MAR**
# - ```net_margin``` has strong correlation between ```margin_gross_pow_elec``` and ```margin)_net_pow_ele```. Suggests multi-colinearity. 
# - ```origin_up``` and ```pow_max``` is **MCAR**. Can drop.
# - ```Forecast_base_bill_ele```, ```forecast_base_bill_year```, ```forecast_bill_12m``` and ```forecast_cons variables``` are highly correlated with ```date_first_activ```. **MNAR**

# In[28]:


# Choose the columns without missing values
incomplete_cols = ['channel_sales','date_first_activ','forecast_base_bill_ele','forecast_base_bill_year','forecast_bill_12m','forecast_cons']

complete_cols = [column_name for column_name in pco_main_drop.columns if column_name not in incomplete_cols]

pco_main_cc = pco_main_drop[complete_cols]

# Fix negative numeric variables
numeric = [column_name for column_name in pco_main_cc.columns
           if pco_main_cc[column_name].dtype == 'float64' 
           or pco_main_cc[column_name].dtype == 'int64']

# Overwrite positive values on negative values
pco_main_cc[numeric] = pco_main_cc[numeric].apply(abs)

# Describe
pco_main_cc.describe()


# In[29]:


# Convert the has_gas column to  Yes/No
pco_main_cc['has_gas'] = pco_main_cc['has_gas'].replace({'t':'Yes','f':'No'})

# Merge the main dataset with the output dataset
pco_main_cc_merged = pco_main_cc.merge(right=pco_output,on=['id'])

# Convert the churn column to Churned/Stayed
pco_main_cc_merged['churn'] = pco_main_cc_merged['churn'].replace({1:'Churned',0:'Stayed'})


# In[30]:


pco_main_cc_merged.head()


# In[31]:


# Obtain all the variables except for id
variables = [column_name for column_name in pco_main_cc_merged.columns if column_name != 'id']

# Obtain all the categorical variables except for id
categorical = [column_name for column_name in variables if pco_main_cc_merged[column_name].dtype == 'object']

# Obtain all the Date Variables
dates = [column_name for column_name in variables if pco_main_cc_merged[column_name].dtype == 'datetime64[ns]']

# Obtain all the numeric columns
numeric = [column_name for column_name in variables
           if column_name not in categorical 
           and column_name != 'id'
           and column_name != 'churn'
           and column_name not in dates]


# ## Data Visualization

# ### The Output Dataset

# In[32]:


# Calculate the zcores of tenure
tenure_zcores = zscore(a=pco_main_cc_merged['num_years_antig'])
# Convert to absolute values
abs_tenure_zscores = np.abs(tenure_zcores)
# Extract Columns of interest
churn_tenure = pco_main_cc_merged[['churn','num_years_antig']]
# Add z-score column
churn_tenure['z_score'] = list(abs_tenure_zscores)
# Remove outliers 
churned_tenure_filtered = churn_tenure[churn_tenure['z_score'] < 3]
# Visualize tenure by retained customer and churner
vio = sns.violinplot( y=churned_tenure_filtered["churn"], x=churned_tenure_filtered["num_years_antig"] )
# Settings
vio.set(xlabel='Years', ylabel='')
vio.set_title("Customer Attrition by Tenure")
plt.show()


# #### Facts
# - The median age of churners is 4 years
# - Customers are more likely to churn during the 4th year than the 7th year
# - The median age of retained customers is 5 years

# ### The Main Dataset

# In[33]:


# Most popular electricty campaign
ele_nm = pco_main_cc_merged.loc[(pco_main_cc_merged['churn']>='Stayed') & (pco_main_cc_merged['net_margin']>0),['id', 'origin_up','net_margin']]

ele_nm.value_counts(subset=['origin_up'])


# In[34]:


# Highest netting electricity subscription campaign
print(ele_nm.groupby('origin_up')['net_margin'].agg('sum').sort_values(ascending=False))


# #### Facts
# - The most popular electricity campaign is ```lxidpiddsbxsbosboudacockeimpuepw``` which has brought 6,584 current customers. With a net margin of $1,541,159.95 in 2015.

# In[35]:


# Select current customers with positive net margins
top_customers = pco_main_cc_merged.loc[(pco_main_cc_merged['churn']>='Stayed') & (pco_main_cc_merged['net_margin']>0),['id','num_years_antig','net_margin']]

# Top 10 customers by net margin
top_customers.sort_values(by=['net_margin'],ascending=False).head(10)


# These are the most profitable customers for PowerCo in terms of net margin. Notet that most of them are within the likely tenure of attrition.

# In[36]:


#!jupyter-nbconvert --to PDFviaHTML BCG.ipynb


# In[ ]:




