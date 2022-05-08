# 1: Business Understanding & Problem Framing

## Objective

Formulate the hypothesis as a data science problem and lay out the major steps needed to test this hypothesis. Communicate your thoughts and findings in an email to your LDS, focusing on the potential data that you would need from the client and analytical models you would use to test such a hypothesis.

### Background Information

PowerCo is a major gas and electricity utility that supplies to corporate, SME (Small & Medium Enterprise), and residential customers. The power-liberalization of the energy market in Europe has led to significant customer churn, especially in the SME segment. They have partnered with BCG to help diagnose the source of churning SME customers. 

One of the hypotheses under consideration is that churn is driven by the customers’ price sensitivities and that it is possible to predict customers likely to churn using a predictive model. The client also wants to try a discounting strategy, with the head of the SME division suggesting that offering customers at high propensity to churn a 20% discount might be effective.

The Lead Data Scientist (LDS) held an initial team meeting to discuss various hypotheses, including churn due to price sensitivity. After discussion with your team, you have been asked to go deeper on the hypothesis that the churn is driven by the customers’ price sensitivities.

Your LDS wants an email with your thoughts on how the team should go about to test this hypothesis.

## Email

**Subject:** PowerCo: Test whether churn is driven by price sensitivity.

Dear Iman Karimi,

To test client’s hypothesis: whether churn is driven by price sensitivity.We will need to predict customer’s likelihood to churn and understand the effect prices have on churn rates.

### Data needed

- **Customer data:** characteristics of each client i.e. previous energy
consumption, enterprise etc.
- **Price data:** historical prices the client charge to customers over time
- **Churn data:** whether customer has churned.

### Workflow (iterative process)

1. Data cleaning: modifying/removing any incorrect, incomplete, irrelevant data.
2. Data preprocessing: normalization/standardization for better convergence, encoding categorical values, feature selection by whittling
down predictors to a smaller set that is more informative.
3. Build Binary classification model: (e.g. Logistic regression, random forests, KNN, neural nets) pick the appropriate model after evaluating tradeoff between accuracy, complexity, and interpretability.
4. Model validation: Split data into training, validation, and test and score model performance.

From the model, we will understand the degree of impact prices have on churn rates. This will help us assess the effectiveness of the client’s discounting strategy.

### Context

Client is a major gas and electricity utility.

- Supplies to corporate, SME, and residential customers.

Significant churn problem.

- Drive by power-liberalization of the energy market in Europe
- Problem is largest in the SME segment

### Client’s Hypothesis

Churn is driven by price sensitivity.
Client wants to try a discounting strategy by offering customers at high propensity to churn a 20% discount

Regards, <br>
Patrick Guo