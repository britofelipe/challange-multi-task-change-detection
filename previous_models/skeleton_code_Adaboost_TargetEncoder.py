# Import necessary libraries
import geopandas as gpd  # For handling geospatial data (GeoJSON)
import pandas as pd  # For data manipulation and handling tabular data
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting results
from sklearn.ensemble import AdaBoostClassifier  # AdaBoost
from sklearn.tree import DecisionTreeClassifier  # Weak learner for AdaBoost
from sklearn.metrics import accuracy_score, f1_score  # Evaluation metrics
from category_encoders import TargetEncoder  # Target encoding for categorical variables
from sklearn.utils.class_weight import compute_sample_weight  # Weighted samples

# ----------------- Step 1: Define Change Type Mapping -----------------
change_type_map = {
    'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3,
    'Industrial': 4, 'Mega Projects': 5
}

# ----------------- Step 2: Read GeoJSON Files -----------------
train_df = gpd.read_file('train.geojson')
test_df = gpd.read_file('test.geojson')

# ----------------- Step 3: Ensure CRS Consistency -----------------
train_df = train_df.to_crs(epsg=3857)  # Web Mercator projection
test_df = test_df.to_crs(epsg=3857)

train_df = train_df[train_df["geometry"].notnull()]
test_df = test_df[test_df["geometry"].notnull()]

# ----------------- Step 4: Feature Engineering -----------------
train_df["area"] = train_df["geometry"].area
train_df["centroid_x"] = train_df["geometry"].centroid.x
train_df["centroid_y"] = train_df["geometry"].centroid.y
train_df["perimeter"] = train_df["geometry"].length
train_df["bounding_width"] = train_df["geometry"].bounds["maxx"] - train_df["geometry"].bounds["minx"]
train_df["bounding_height"] = train_df["geometry"].bounds["maxy"] - train_df["geometry"].bounds["miny"]
train_df["aspect_ratio"] = train_df["bounding_width"] / train_df["bounding_height"]
train_df["circularity"] = (4 * np.pi * train_df["area"]) / (train_df["perimeter"] ** 2)

test_df["area"] = test_df["geometry"].area
test_df["centroid_x"] = test_df["geometry"].centroid.x
test_df["centroid_y"] = test_df["geometry"].centroid.y
test_df["perimeter"] = test_df["geometry"].length
test_df["bounding_width"] = test_df["geometry"].bounds["maxx"] - test_df["geometry"].bounds["minx"]
test_df["bounding_height"] = test_df["geometry"].bounds["maxy"] - test_df["geometry"].bounds["miny"]
test_df["aspect_ratio"] = test_df["bounding_width"] / test_df["bounding_height"]
test_df["circularity"] = (4 * np.pi * test_df["area"]) / (test_df["perimeter"] ** 2)

# ----------------- Step 5: Handle Categorical Features -----------------
cat_features = ["urban_type", "geography_type"]

train_df[cat_features] = train_df[cat_features].astype(str)
test_df[cat_features] = test_df[cat_features].astype(str)

train_y = train_df['change_type'].apply(lambda x: change_type_map[x])

# Target Encoding
encoder = TargetEncoder()
train_df[cat_features] = encoder.fit_transform(train_df[cat_features], train_y)
test_df[cat_features] = encoder.transform(test_df[cat_features])

# ----------------- Step 6: Handle Date Features -----------------
date_features = ["date0", "date1", "date2", "date3", "date4"]
for col in date_features:
    train_df[col] = pd.to_datetime(train_df[col], errors='coerce').astype('int64') // 10**9
    test_df[col] = pd.to_datetime(test_df[col], errors='coerce').astype('int64') // 10**9

# ----------------- Step 7: Select Features -----------------
feature_cols = cat_features + [
    "area", "centroid_x", "centroid_y", "perimeter", 
    "bounding_width", "bounding_height", "aspect_ratio", "circularity",
    "img_red_mean_date1", "img_green_mean_date1", "img_blue_mean_date1",
    "img_red_std_date1", "img_green_std_date1", "img_blue_std_date1",
    "img_red_mean_date2", "img_green_mean_date2", "img_blue_mean_date2",
    "img_red_std_date2", "img_green_std_date2", "img_blue_std_date2",
    "date0", "date1", "date2", "date3", "date4"
]

# Extract features
train_x = train_df[feature_cols]
test_x = test_df[feature_cols]

# Handle missing values
train_x.fillna(train_x.median(), inplace=True)
test_x.fillna(test_x.median(), inplace=True)

train_x.replace([np.inf, -np.inf], np.nan, inplace=True)
test_x.replace([np.inf, -np.inf], np.nan, inplace=True)

train_x.fillna(train_x.median(), inplace=True)
test_x.fillna(test_x.median(), inplace=True)

# ----------------- Step 8: Train AdaBoost Model -----------------
# Define hyperparameters
num_estimators = 50
learning_rate = 0.1

# Compute sample weights to balance class distribution
sample_weights = compute_sample_weight(class_weight="balanced", y=train_y)

# Create a weak learner (Decision Tree)
weak_learner = DecisionTreeClassifier(max_depth=3)

# Create and train the AdaBoost model
ada_model = AdaBoostClassifier(
    estimator=weak_learner,
    n_estimators=num_estimators,
    learning_rate=learning_rate
)

# Train with weighted samples
ada_model.fit(train_x, train_y, sample_weight=sample_weights)

# ----------------- Step 9: Model Prediction -----------------
pred_y = ada_model.predict(test_x)

# ----------------- Step 10: Save Results -----------------
pred_df = pd.DataFrame(pred_y, columns=['change_type'])
pred_df.to_csv("adaboost_submission.csv", index=True, index_label='Id')

# ----------------- Step 11: Model Evaluation -----------------
train_accuracy = accuracy_score(train_y, ada_model.predict(train_x))
print(f"Training Accuracy: {train_accuracy:.4f}")

train_f1 = f1_score(train_y, ada_model.predict(train_x), average='macro')
print(f"Mean F1-Score (Train - Macro): {train_f1:.4f}")

train_f1_weighted = f1_score(train_y, ada_model.predict(train_x), average='weighted')
print(f"Weighted Mean F1-Score (Train): {train_f1_weighted:.4f}")

# ----------------- Step 12: Plot AdaBoost Learning Curve -----------------
plt.plot(np.arange(1, num_estimators + 1), ada_model.estimator_errors_, label="Estimator Errors")
plt.xlabel("Number of Estimators")
plt.ylabel("Error")
plt.title("AdaBoost Training Error Over Time")
plt.legend()
plt.show()