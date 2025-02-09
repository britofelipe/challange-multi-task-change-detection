import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

change_type_map = {'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3, 'Industrial': 4,
       'Mega Projects': 5}

## Read csvs
train_df = gpd.read_file('train.geojson', index_col=0)
test_df = gpd.read_file('test.geojson', index_col=0)

# Convert date columns to datetime format
date_columns = [col for col in train_df.columns if "date" in col]
for col in date_columns:
    train_df[col] = pd.to_datetime(train_df[col], errors='coerce')  # Convert invalids to NaT
    test_df[col] = pd.to_datetime(test_df[col], errors='coerce')

# Time differences between dates
for i in range(len(date_columns) - 1):
    train_df[f"time_diff_{i}"] = (train_df[date_columns[i+1]] - train_df[date_columns[i]]).dt.days
    test_df[f"time_diff_{i}"] = (test_df[date_columns[i+1]] - test_df[date_columns[i]]).dt.days

# handle missing values 
# drop highly missing columns
threshold = 0.5 * len(train_df)
missing_columns = train_df.isnull().sum()
drop_cols = missing_columns[missing_columns > threshold].index.tolist()
train_df.drop(columns=drop_cols, inplace=True)
test_df.drop(columns=drop_cols, inplace=True)
# compute remining missing values with the median
imputer = SimpleImputer(strategy="median")
train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)

# extracting geomety features
train_df["area"] = train_df["geometry"].area
train_df["perimeter"] = train_df["geometry"].length
train_df["compactness"] = (4 * np.pi * train_df["area"]) / (train_df["perimeter"] ** 2)
train_df["min_x"], train_df["min_y"], train_df["max_x"], train_df["max_y"] = zip(*train_df["geometry"].bounds)

test_df["area"] = test_df["geometry"].area
test_df["perimeter"] = test_df["geometry"].length
test_df["compactness"] = (4 * np.pi * test_df["area"]) / (test_df["perimeter"] ** 2)
test_df["min_x"], test_df["min_y"], test_df["max_x"], test_df["max_y"] = zip(*test_df["geometry"].bounds)

train_df.drop(columns=["geometry"], inplace=True)
test_df.drop(columns=["geometry"], inplace=True)

## Filtering column "mail_type"
train_x = np.asarray(train_df[['geometry']].area)
train_x = train_x.reshape(-1, 1)
train_y = train_df['change_type'].apply(lambda x: change_type_map[x])

test_x = np.asarray(test_df[['geometry']].area)
test_x = test_x.reshape(-1, 1)

print (train_x.shape, train_y.shape, test_x.shape)


## Train a simple OnveVsRestClassifier using featurized data
# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(train_x, train_y)
# pred_y = neigh.predict(test_x)
# print (pred_y.shape)

## Save results to submission file
# pred_df = pd.DataFrame(pred_y, columns=['change_type'])
# pred_df.to_csv("knn_sample_submission.csv", index=True, index_label='Id')