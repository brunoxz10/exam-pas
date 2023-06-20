import pandas as pd
from pandas.api.types import CategoricalDtype

import config

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import xgboost as xgb

import pickle

# Read processed data
df = pd.read_parquet('../../data/processed/scores_approvals_convocation_2020_2022.parquet')
df2 = pd.read_parquet('../../data/processed/scores_approvals_convocation_2019_2021.parquet')

# Converting course feature according to pre defined categories
cat_type = CategoricalDtype(categories=config.COURSE_NAMES)

df['course'] = df['course'].astype(cat_type)
df2['course'] = df2['course'].astype(cat_type)

# Shuffling both dataframes
shuffled_df = df.sample(frac=1, random_state=42)
df2_shuffled = df2.sample(frac=1, random_state=42)

# Splitting traning and test data, only 2020-2022 data for test
X = shuffled_df[config.FEATURES] # features
y = shuffled_df['label'] # labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=47) 

# Setting final training data, adding 2019-2022 data
X_train = pd.concat([X_train, df2_shuffled[config.FEATURES]])
y_train = pd.concat([y_train, df2_shuffled['label']])

# Final processing for flags features
X_train['min_flag'] = X_train['min_flag'].astype("category")
X_train['median_flag'] = X_train['median_flag'].astype("category")
X_test['min_flag'] = X_test['min_flag'].astype("category")
X_test['median_flag'] = X_test['median_flag'].astype("category")

# Create an XGBoost classifier
model = xgb.XGBClassifier(**config.HYPERPARAMETERS,
                          monotone_constraints='(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)',
                          enable_categorical=True)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print(classification_report(y_test, y_pred))

#with open('../../models/xgboost_categorical_not_calibrated.pickle','wb') as f:
#    pickle.dump(model, f)