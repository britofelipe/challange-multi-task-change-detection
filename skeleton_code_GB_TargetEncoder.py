# Import necessary libraries
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from category_encoders import TargetEncoder

# ----------------- Step 1: Define Change Type Mapping -----------------
change_type_map = {
    'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3,
    'Industrial': 4, 'Mega Projects': 5
}

# ----------------- Step 2: Read GeoJSON Files -----------------
train_df = gpd.read_file('train.geojson')
test_df = gpd.read_file('test.geojson')

# ----------------- Step 3: Ensure CRS Consistency -----------------
train_df = train_df.to_crs(epsg=3857)
test_df = test_df.to_crs(epsg=3857)

# Remove any invalid (null) geometries
train_df = train_df[train_df["geometry"].notnull()]
test_df = test_df[test_df["geometry"].notnull()]
train_df = train_df[train_df["geometry"].is_valid]
test_df = test_df[test_df["geometry"].is_valid]

# ----------------- Step 4: Feature Engineering -----------------
train_df["area"] = train_df["geometry"].area
train_df["area"].fillna(1e-6, inplace=True)
train_df["centroid_x"] = train_df["geometry"].centroid.x
train_df["centroid_y"] = train_df["geometry"].centroid.y
train_df["perimeter"] = train_df["geometry"].length
train_df["bounding_width"] = train_df["geometry"].bounds["maxx"] - train_df["geometry"].bounds["minx"]
train_df["bounding_height"] = train_df["geometry"].bounds["maxy"] - train_df["geometry"].bounds["miny"]
train_df["aspect_ratio"] = train_df["bounding_width"] / np.maximum(train_df["bounding_height"], 1e-9)
train_df["circularity"] = (4 * np.pi * train_df["area"]) / (train_df["perimeter"] ** 2)

# Apply transformations to the test dataset
test_df["area"] = test_df["geometry"].area
test_df["area"].fillna(1e-6, inplace=True)
test_df["centroid_x"] = test_df["geometry"].centroid.x
test_df["centroid_y"] = test_df["geometry"].centroid.y
test_df["perimeter"] = test_df["geometry"].length
test_df["bounding_width"] = test_df["geometry"].bounds["maxx"] - test_df["geometry"].bounds["minx"]
test_df["bounding_height"] = test_df["geometry"].bounds["maxy"] - test_df["geometry"].bounds["miny"]
test_df["aspect_ratio"] = test_df["bounding_width"] / np.maximum(test_df["bounding_height"], 1e-9)
test_df["circularity"] = (4 * np.pi * test_df["area"]) / (test_df["perimeter"] ** 2)

# ----------------- Step 5: Handle Categorical Features -----------------
cat_features = ["urban_type", "geography_type"]
train_df[cat_features] = train_df[cat_features].astype(str)
test_df[cat_features] = test_df[cat_features].astype(str)

# Convert labels to numerical values
train_y = train_df['change_type'].map(change_type_map).astype(int)

# Define target encoder
encoder = TargetEncoder()
train_df[cat_features] = encoder.fit_transform(train_df[cat_features], train_y)
test_df[cat_features] = encoder.transform(test_df[cat_features])  # Ensure no leakage

# ----------------- Step 6: Handle Date Features -----------------
date_features = ["date0", "date1", "date2", "date3", "date4"]
for col in date_features:
    for col in date_features:
        train_df[col] = pd.to_datetime(train_df[col], format='%d-%m-%Y', errors='coerce').astype('int64') // 10**9
        test_df[col] = pd.to_datetime(test_df[col], format='%d-%m-%Y', errors='coerce').astype('int64') // 10**9
# **Log-transform skewed numerical features BEFORE converting to NumPy**
log_features = ["area", "perimeter", "bounding_width", "bounding_height"]
for col in log_features:
    train_df[col] = np.log1p(np.maximum(train_df[col], 1e-9))  
    test_df[col] = np.log1p(np.maximum(test_df[col], 1e-9))

print(train_df.isna().sum())  # Identify any missing values
print((train_df[["area", "perimeter", "bounding_width", "bounding_height"]] <= 0).sum())
# ----------------- Step 7: Select Features -----------------
feature_cols = (
    cat_features + [
        "area", "centroid_x", "centroid_y", "perimeter", 
        "bounding_width", "bounding_height", "aspect_ratio", "circularity",
        "img_red_mean_date1", "img_green_mean_date1", "img_blue_mean_date1",
        "img_red_std_date1", "img_green_std_date1", "img_blue_std_date1",
        "img_red_mean_date2", "img_green_mean_date2", "img_blue_mean_date2",
        "img_red_std_date2", "img_green_std_date2", "img_blue_std_date2",
        "date0", "date1", "date2", "date3", "date4"
    ]
)

train_x = train_df[feature_cols]
test_x = test_df[feature_cols]

# Handle missing values
train_x = train_x.fillna(train_x.median())
test_x = test_x.fillna(test_x.median())

# Replace infinite values
train_x = np.nan_to_num(train_x, nan=0.0, posinf=1e10, neginf=-1e10)
test_x = np.nan_to_num(test_x, nan=0.0, posinf=1e10, neginf=-1e10)

# Standardize features
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# Feature Selection (Select Top 14 Features)
selector = SelectKBest(score_func=f_classif, k=14)  
train_x = selector.fit_transform(train_x, train_y)  
test_x = selector.transform(test_x)

selected_feature_indices = selector.get_support(indices=True)
selected_features = [feature_cols[i] for i in selected_feature_indices]
print(f"Selected Features: {selected_features}")

# ----------------- Step 8: Train Gradient Boosting Model -----------------
num_estimators = 60
learning_rate = 0.03
max_depth = 4
subsample = 0.75  

gbm_model = GradientBoostingClassifier(
    n_estimators=num_estimators, 
    learning_rate=learning_rate, 
    max_depth=max_depth,
    subsample=subsample
)

gbm_model.fit(train_x, train_y)

# ----------------- Step 9: Model Prediction -----------------
pred_y = gbm_model.predict(test_x)

# ----------------- Step 10: Save Results -----------------
pred_df = pd.DataFrame(pred_y, columns=['change_type'])
pred_df.to_csv("gradient_boosting_submission.csv", index=True, index_label='Id')

# ----------------- Step 11: Model Evaluation -----------------
train_accuracy = accuracy_score(train_y, gbm_model.predict(train_x))
train_f1 = f1_score(train_y, gbm_model.predict(train_x), average='macro')
train_f1_weighted = f1_score(train_y, gbm_model.predict(train_x), average='weighted')

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Mean F1-Score (Train - Macro): {train_f1:.4f}")
print(f"Weighted Mean F1-Score (Train): {train_f1_weighted:.4f}")

# ----------------- Step 12: Plot Training Loss -----------------
plt.plot(np.arange(1, num_estimators + 1), gbm_model.train_score_, label="Training Loss")
plt.xlabel("Number of Estimators")
plt.ylabel("Loss")
plt.title("Gradient Boosting Training Loss Over Time")
plt.legend()
plt.show()