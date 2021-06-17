# Viral Tweet Prediction Challenge Submission
Author: Patrick Guo

Clear and unambiguous instructions on how to reproduce the predictions from start to finish including data pre-processing, feature extraction, model training and predictions generation in notebook.

## Environment details 
**OS:** macOS Big Sur 11.4  
**Memory:** 16 GB 2133 MHz LPDDR3  
**Disk Space:** 1 TB Flask Stroage  
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

### What is the algorithm used and what are its main hyperparameters?
Used Lightgbm model
``` lgb.LGBMClassifier() ```
### Any other comments considered relevant to understanding and using the model
