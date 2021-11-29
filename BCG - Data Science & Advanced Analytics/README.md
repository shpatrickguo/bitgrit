# Tasks

## 1: Business Understanding & Problem Framing

**Objective**
Formulate the hypothesis as a data science problem and lay out the major steps needed to test this hypothesis. Communicate your thoughts and findings in an email to your LDS, focusing on the potential data that you would need from the client and analytical models you would use to test such a hypothesis.

**Background Information**
PowerCo is a major gas and electricity utility that supplies to corporate, SME (Small & Medium Enterprise), and residential customers. The power-liberalization of the energy market in Europe has led to significant customer churn, especially in the SME segment. They have partnered with BCG to help diagnose the source of churning SME customers. 

One of the hypotheses under consideration is that churn is driven by the customers’ price sensitivities and that it is possible to predict customers likely to churn using a predictive model. The client also wants to try a discounting strategy, with the head of the SME division suggesting that offering customers at high propensity to churn a 20% discount might be effective.

The Lead Data Scientist (LDS) held an initial team meeting to discuss various hypotheses, including churn due to price sensitivity. After discussion with your team, you have been asked to go deeper on the hypothesis that the churn is driven by the customers’ price sensitivities.

Your LDS wants an email with your thoughts on how the team should go about to test this hypothesis.

## 2: Exploratory Data Analysis & Data Cleaning

**Objective**
Clean the data – address missing values, duplicates, data type conversions, transformations, and multi-co-linearity, as well as outliers.

Perform some exploratory data analysis. Look into the data types, data statistics, and identify any missing data or null values, and how often they appear in the data. Visualize specific parameters as well as variable distributions.

**Background Information**
The BCG project team thinks that building a churn model to understand whether price sensitivity is the largest driver of churn has potential. The client has sent over some data and the LDS wants you to perform some exploratory data analysis and data cleaning.

The data that was sent over includes:

Historical customer data: Customer data such as usage, sign up date, forecasted usage etc
Historical pricing data: variable and fixed pricing data etc
Churn indicator: whether each customer has churned or not
These datasets are otherwise identical and have historical price data and customer data (including churn status for the customers in the training data).

## 3: Feature Engineering
**Objectives:**
- What key drivers of churn could be for our client
- Build the features in order to get ready to model

Feature engineering is one of the keys to unlocking predictive insight through mathematical modeling. Based on the data that is available and was cleaned, identify what  could be drivers of churn for our client and build those features to later use in your model.

