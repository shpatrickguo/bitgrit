#!/usr/bin/env python
# coding: utf-8

# # Travel Assured

# ## Introduction
# Travel Assured is a travel insurance company. Due to the COVID pandemic they have had to cut their marketing budget by over 50%. It is more important than ever that they advertise in the right places and to the right people.
# 
# Travel Assured has data on their current customers as well as people who got quotes but never bought insurance. They want to know if there are differences in the travel habits between customers and non-customers - they believe they are more likely to travel often (buying tickets from frequent flyer miles) and travel abroad.

# ## The Data
# - `age`: Numeric, the customer’s age
# - `Employment Type`: Character, the sector of employment
# - `GraduateOrNot`: Character, whether the customer is a college graduate
# - `AnnualIncome`: Numeric, the customer’s yearly income
# - `FamilyMembers`: Numeric, the number of family members living with the customer
# - `ChronicDiseases`: Numeric, whether the customer has any chronic conditions
# - `FrequentFlyer`: Character, whether a customer books frequent tickets
# - `EverTravelledAbroad`: Character, has the customer ever travelled abroad
# - `TravelInsurance`: Numeric, whether the customer bought travel insurance

# ## Import Libraries
# Let us set up all the objects (packages, and modules) that we will need to explore the dataset.

# In[1]:


# Import necessary modules
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


# ## Load the Data

# In[2]:


# Read file
df = pd.read_csv('data/travel_insurance.csv')

# Split the data in two subgroups, to compare data with/without Travel Insurance
insured_df = df[df['TravelInsurance'] == 1]
uninsured_df = df[df['TravelInsurance'] == 0]
df.head()


# ## Data Size and Structure
# 
# - Dataset comprises of 1987 observations and 9 columns.
# - 8 independent variables: `Age`, `Employement Type`, `GraduateOrNot`, `AnnualIncome`, `FamilyMembers`, `ChronicDiseases`, `FrequentFlyer`, `EverTravelledAbroad`
#     - 3 numerical features, 5 categorical features
# - 1 dependent variable: `TravelInsurance`
# - Complete dataset with no missing/null values.

# In[3]:


# Check the size and datatypes of the DataFrame
print(f"Shape: {df.shape}")
print('\n')
print(df.info())


# In[4]:


# For all categorical data columns
# Print categories and the number of times they appear
# ChornicDiseases and TravelInsurance at categorical because 1 and 0 represent yes and no
CAT_COLS = ['Employment Type', 'GraduateOrNot', "ChronicDiseases", "FrequentFlyer", "EverTravelledAbroad", "TravelInsurance"]

for col in df[CAT_COLS]:
    print(df[col].value_counts())
    print('\n')


# In[5]:


df.describe()


# In[6]:


# Percentage of dataset is null
df.isnull().sum() / len(df) * 100


# In[7]:


insured_df.value_counts('EverTravelledAbroad')


# ## Exploratory Data Analysis
# Let's define the global variables and functions that we will use regularly throughout the analysis.

# In[8]:


LABELS = ['No', 'Yes'] 
EMPLOYMENT_LABELS = ['Private Sector/Self Employed', 'Government Sector'] 

def label_fontsize(fontsize):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)


# ### What are the travel habits of the insured and uninsured?
# 
# Now let's first look at the how many people have travel insurance and their travel habits. The countplot shows that a large proportion of people that have never travelled abroad are uninsured. However for those that have, a significant proportion have gotten travel insurance. This pattern is also seen for people that aren't/are frequent flyers. Indicating that people that have travelled abroad or are frequent flyers are more likely to get travel insurance.

# In[9]:


insured_travelled = insured_df.value_counts('EverTravelledAbroad').values
uninsured_travelled = uninsured_df.value_counts('EverTravelledAbroad').values

x = np.arange(len(LABELS))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(15, 8))
rects1 = ax.bar(x + width/2, insured_travelled, width, label='Insured', color='forestgreen')
rects2 = ax.bar(x - width/2, uninsured_travelled, width, label='Uninsured', color='firebrick')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Ever travelled abroad')
ax.set_ylabel('Count')
ax.set_title('Count of people that have travelled abroad by whether they are insured')
ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.legend(prop={'size': 20})

ax.bar_label(rects1, padding=3, fontsize=20)
ax.bar_label(rects2, padding=3, fontsize=20)

label_fontsize(20)

fig.tight_layout()
plt.show()


# In[10]:


insured_freqfly = insured_df.value_counts('FrequentFlyer').values
uninsured_freqfly = uninsured_df.value_counts('FrequentFlyer').values

x = np.arange(len(LABELS))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(15, 8))
rects1 = ax.bar(x + width/2, insured_freqfly, width, label='Insured', color='forestgreen')
rects2 = ax.bar(x - width/2, uninsured_freqfly, width, label='Uninsured', color='firebrick')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Frequent Flyer')
ax.set_ylabel('Count')
ax.set_title('Count of people that are frequent fliers by whether they are insured')
ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.legend(prop={'size': 20})

ax.bar_label(rects1, padding=3, fontsize=20)
ax.bar_label(rects2, padding=3, fontsize=20)

label_fontsize(20)

fig.tight_layout()
plt.show()


# ### Other demographic information
# Now that we have an understanding of their travel habits, let's explore other demographic information.

# In[11]:


insured_chronic = insured_df.value_counts('ChronicDiseases').values
uninsured_chronic = uninsured_df.value_counts('ChronicDiseases').values

x = np.arange(len(LABELS))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(15, 8))
rects1 = ax.bar(x + width/2, insured_chronic, width, label='Insured', color='forestgreen')
rects2 = ax.bar(x - width/2, uninsured_chronic, width, label='Uninsured', color='firebrick')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Has a chronic disease')
ax.set_ylabel('Count')
ax.set_title('Count of people that have a chronic disease by whether they are insured')
ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.legend(prop={'size': 20})
ax.bar_label(rects1, padding=3, fontsize=20)
ax.bar_label(rects2, padding=3, fontsize=20)
label_fontsize(20)

fig.tight_layout()
plt.show()


# ### Family Members

# In[12]:


fig, ax = plt.subplots(figsize=(15, 8))
ax = sns.countplot(x="FamilyMembers", data=df, hue="TravelInsurance")
ax.set_xlabel('Number of family members')
ax.set_title('Frequency of family size by whether they are insured')
ax.legend(title="Insured", labels=LABELS, fontsize=20, title_fontsize=20)
label_fontsize(20)
fig.tight_layout()
plt.show()


# In[13]:


insured_fam_prop = insured_df.value_counts('FamilyMembers') / df.value_counts('FamilyMembers')

fig, ax = plt.subplots(figsize=(15, 8))
ax.bar(insured_fam_prop.index, insured_fam_prop, color='royalblue')

ax.set_xlabel('Number of family members')
ax.set_ylabel('Proportion insured')
ax.set_title('Proportion of the insured by family size')
label_fontsize(20)
fig.tight_layout()
plt.show()


# In[14]:


fig, ax = plt.subplots(figsize=[16,7])

# plot histogram comparing the density distribution of the Annual Income in each groups
_ = plt.hist(insured_df['AnnualIncome'], 
             histtype='step', 
             hatch='/', 
             density=True, 
             color='green', 
             bins=10, 
             linestyle='--', 
             linewidth=2,
            )
_ = plt.hist(uninsured_df['AnnualIncome'],
             density=True,
             color='red', 
             bins=10,
            )

# put a title, label the axes, define the legend and display
_ = plt.suptitle("PDF of Annual Income by whether they are insured", 
                 fontsize=20, 
                 color='black',
                 x=0.52,
                 y=0.92,
                )
_ = plt.xlabel('Annual Income')
_ = plt.ylabel('Probability Density Function (PDF)')
_ = plt.legend(['Insured', 'Uninsured'])
_ = plt.ticklabel_format(axis="x", style="plain")
_ = plt.ticklabel_format(axis="y", style="plain")
plt.show()


# In[15]:


insured_grad = insured_df.value_counts('GraduateOrNot').values
uninsured_grad = uninsured_df.value_counts('GraduateOrNot').values

x = np.arange(len(LABELS))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(15, 8))
rects1 = ax.bar(x + width/2, insured_grad, width, label='Insured', color='forestgreen')
rects2 = ax.bar(x - width/2, uninsured_grad, width, label='Uninsured', color='firebrick')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Graduate')
ax.set_ylabel('Count')
ax.set_title('Count of people that are graduates by whether they are insured')
ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.legend(prop={'size': 20})

ax.bar_label(rects1, padding=3, fontsize=20)
ax.bar_label(rects2, padding=3, fontsize=20)

label_fontsize(20)

fig.tight_layout()
plt.show()


# In[16]:



insured_employ_type = insured_df.value_counts('Employment Type').values
uninsured_employ_type = uninsured_df.value_counts('GraduateOrNot').values

x = np.arange(len(EMPLOYMENT_LABELS))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(15, 8))
rects1 = ax.bar(x + width/2, insured_employ_type, width, label='Insured', color='forestgreen')
rects2 = ax.bar(x - width/2, uninsured_employ_type, width, label='Uninsured', color='firebrick')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Employment Type')
ax.set_ylabel('Count')
ax.set_title('Count of employment type by whether they are insured')
ax.set_xticks(x)
ax.set_xticklabels(EMPLOYMENT_LABELS)
ax.legend(prop={'size': 20})

ax.bar_label(rects1, padding=3, fontsize=20)
ax.bar_label(rects2, padding=3, fontsize=20)

label_fontsize(20)

fig.tight_layout()
plt.show()


# In[17]:


fig, ax = plt.subplots(figsize=[16,7])

# plot histogram comparing the density distribution of the Annual Income in each groups
_ = plt.hist(insured_df['Age'], 
             histtype='step', 
             hatch='/', 
             density=True, 
             color='green', 
             bins=10, 
             linestyle='--', 
             linewidth=2,
            )
_ = plt.hist(uninsured_df['Age'],
             density=True,
             color='red', 
             bins=10,
            )

# put a title, label the axes, define the legend and display
_ = plt.suptitle("PDF of Age by whether they are insured", 
                 fontsize=20, 
                 color='black',
                 x=0.52,
                 y=0.92,
                )
_ = plt.xlabel('Age')
_ = plt.ylabel('Probability Density Function (PDF)')
_ = plt.legend(['Insured', 'Uninsured'])
_ = plt.ticklabel_format(axis="x", style="plain")
_ = plt.ticklabel_format(axis="y", style="plain")
plt.show()


# In[ ]:




