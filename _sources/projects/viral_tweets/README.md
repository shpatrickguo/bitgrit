# [Viral Tweet Prediction](https://www.kaggle.com/code/shpatrickguo/tweet-virality-prediction-with-light-gbm)

Clear and unambiguous instructions on how to reproduce the predictions from start to finish including data pre-processing, feature extraction, model training and predictions generation in notebook.

## In this notebook

- Data processing: one-hot encoding + cyclical encoding for categorical features. Normalization.
- LASSO regression for feature selection
- Memory footprint reduction of data
- Hyper-parameter tuning with RandomizedSearchCV
- Building LightGBM classifier model for prediction
- Feature importance visualization

## Environment details

**OS:** macOS Big Sur 11.4  
**Memory:** 16 GB 2133 MHz LPDDR3  
**Disk Space:** 1 TB Flask Storage  
**CPU/GPU:** Intel HD Graphics 530 1536 MB  

### Which data files are being used?

- train_tweets.csv
- train_tweets_vectorized_media.csv
- train_tweets_vectorized_text.csv
- users.csv
- user_vectorized_descriptions.csv
- user_vectorized_profile_images.csv
- test_tweets.csv
- test_tweets_vectorized_media.csv
- test_tweets_vectorized_text.csv

### How are these files processed?

- Filling missing topic_ids with ['0']
- One hot encoding for categorical variables
- Cyclical encoding for hour

### What is the algorithm used and what are its main hyper-parameters?

Used LightGBM Classifier:  

``` {code}
LGBMClassifier(colsample_bytree=0.7076074093370144, min_child_samples=105, min_child_weight=1e-05, num_leaves=26, reg_alpha=5, reg_lambda=5, subsample=0.7468773130235173)
```
