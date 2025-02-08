# Import necessary libraries
import geopandas as gpd  # For handling geospatial data (GeoJSON)
import pandas as pd  # For data manipulation and handling tabular data
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting results
from sklearn.ensemble import GradientBoostingClassifier  # Gradient Boosting
from sklearn.metrics import accuracy_score, f1_score  # Evaluation metrics
from sklearn.preprocessing import LabelEncoder  # Encoding categorical data
from sklearn.utils.class_weight import compute_sample_weight
from category_encoders import TargetEncoder

# ----------------- Step 1: Define Change Type Mapping -----------------
# Mapping different land-use change types to numerical values for machine learning
change_type_map = {
    'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3,
    'Industrial': 4, 'Mega Projects': 5
}

# ----------------- Step 2: Read GeoJSON Files -----------------
# Load training and testing datasets
train_df = gpd.read_file('train.geojson', index_col=0)
test_df = gpd.read_file('test.geojson', index_col=0)

# ----------------- Step 3: Ensure CRS Consistency (Reprojection) -----------------
# Convert coordinate reference system (CRS) to a projected system for accurate area and length calculations
train_df = train_df.to_crs(epsg=3857)  # Web Mercator projection
test_df = test_df.to_crs(epsg=3857)

# Remove any invalid (null) geometries that might cause errors
train_df = train_df[train_df["geometry"].notnull()]
test_df = test_df[test_df["geometry"].notnull()]

# Print the count of different geometry types in the dataset
print(train_df["geometry"].geom_type.value_counts())
print(train_df["change_type"].value_counts())

# ----------------- Step 4: Feature Engineering -----------------
# Compute geometric features
train_df["area"] = train_df["geometry"].area
train_df["centroid_x"] = train_df["geometry"].centroid.x  # X-coordinate of the centroid
train_df["centroid_y"] = train_df["geometry"].centroid.y  # Y-coordinate of the centroid
train_df["perimeter"] = train_df["geometry"].length  # Perimeter length of the shape
train_df["bounding_width"] = train_df["geometry"].bounds["maxx"] - train_df["geometry"].bounds["minx"]
train_df["bounding_height"] = train_df["geometry"].bounds["maxy"] - train_df["geometry"].bounds["miny"]
train_df["aspect_ratio"] = train_df["bounding_width"] / train_df["bounding_height"]
train_df["circularity"] = (4 * np.pi * train_df["area"]) / (train_df["perimeter"] ** 2)

# Apply the same transformations to the test dataset
test_df["area"] = test_df["geometry"].area
test_df["centroid_x"] = test_df["geometry"].centroid.x
test_df["centroid_y"] = test_df["geometry"].centroid.y
test_df["perimeter"] = test_df["geometry"].length
test_df["bounding_width"] = test_df["geometry"].bounds["maxx"] - test_df["geometry"].bounds["minx"]
test_df["bounding_height"] = test_df["geometry"].bounds["maxy"] - test_df["geometry"].bounds["miny"]
test_df["aspect_ratio"] = test_df["bounding_width"] / test_df["bounding_height"]
test_df["circularity"] = (4 * np.pi * test_df["area"]) / (test_df["perimeter"] ** 2)

# Print the column names to verify feature extraction
print(train_df.columns)
print(test_df.columns)

# ----------------- Step 5: Handle Categorical Features -----------------
# Convert categorical variables ('urban_type', 'geography_type') using Target Encoding
cat_features = ["urban_type", "geography_type"]

# Ensure categorical variables are strings
train_df[cat_features] = train_df[cat_features].astype(str)
test_df[cat_features] = test_df[cat_features].astype(str)

# Convert labels to numerical values
train_y = train_df['change_type'].apply(lambda x: change_type_map[x])

# Define target encoder
encoder = TargetEncoder()

# Fit and transform categorical variables (pass `train_y` as a Pandas Series)
train_df[cat_features] = encoder.fit_transform(train_df[cat_features], train_y)

# Transform test set (do not fit again to avoid data leakage)
test_df[cat_features] = encoder.transform(test_df[cat_features])

# Get updated column names after one-hot encoding
encoded_cat_features = cat_features  # Extracts new feature names

# ----------------- Step 6: Handle Date Features -----------------
# Convert date columns to numeric timestamps
date_features = ["date0", "date1", "date2", "date3", "date4"]

for col in date_features:
    train_df[col] = pd.to_datetime(train_df[col], errors='coerce').astype('int64') // 10**9
    test_df[col] = pd.to_datetime(test_df[col], errors='coerce').astype('int64') // 10**9

# ----------------- Step 7: Select Features -----------------
# List of numerical features (includes original + generated ones)
feature_cols = (
    encoded_cat_features + [  # Add encoded categorical features
        "area", "centroid_x", "centroid_y", "perimeter", 
        "bounding_width", "bounding_height", "aspect_ratio", "circularity",  # Geometric features
        "img_red_mean_date1", "img_green_mean_date1", "img_blue_mean_date1",
        "img_red_std_date1", "img_green_std_date1", "img_blue_std_date1",
        "img_red_mean_date2", "img_green_mean_date2", "img_blue_mean_date2",
        "img_red_std_date2", "img_green_std_date2", "img_blue_std_date2",
        "date0", "date1", "date2", "date3", "date4"  # Date-based features
    ]
)

print(feature_cols)

# Extract features
train_x = train_df[feature_cols]
test_x = test_df[feature_cols]

# Handle missing values by replacing NaNs with median values
train_x.fillna(train_x.median(), inplace=True)
test_x.fillna(test_x.median(), inplace=True)

# Replace infinity values with NaNs
train_x.replace([np.inf, -np.inf], np.nan, inplace=True)
test_x.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaNs again (for safety) with median values
train_x.fillna(train_x.median(), inplace=True)
test_x.fillna(test_x.median(), inplace=True)


# ----------------- Step 8: Train Gradient Boosting Model -----------------
# Define hyperparameters
num_estimators = 25
learning_rate = 0.1
max_depth = 4

# Create and train the model
gbm_model = GradientBoostingClassifier(
    n_estimators=num_estimators, 
    learning_rate=learning_rate, 
    max_depth=max_depth
)

gbm_model.fit(train_x, train_y)

# ----------------- Step 9: Model Prediction -----------------
pred_y = gbm_model.predict(test_x)

# ----------------- Step 10: Save Results -----------------
pred_df = pd.DataFrame(pred_y, columns=['change_type'])
pred_df.to_csv("gradient_boosting_submission.csv", index=True, index_label='Id')

# ----------------- Step 11: Model Evaluation -----------------
train_accuracy = accuracy_score(train_y, gbm_model.predict(train_x))
print(f"Training Accuracy: {train_accuracy:.4f}")

train_f1 = f1_score(train_y, gbm_model.predict(train_x), average='macro')
print(f"Mean F1-Score (Train - Macro): {train_f1:.4f}")

train_f1_weighted = f1_score(train_y, gbm_model.predict(train_x), average='weighted')
print(f"Weighted Mean F1-Score (Train): {train_f1_weighted:.4f}")

# ----------------- Step 12: Plot Training Loss -----------------
plt.plot(np.arange(1, num_estimators + 1), gbm_model.train_score_, label="Training Loss")
plt.xlabel("Number of Estimators")
plt.ylabel("Loss")
plt.title("Gradient Boosting Training Loss Over Time")
plt.legend()
plt.show()