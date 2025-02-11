# Import necessary libraries
import geopandas as gpd  # For handling geospatial data (GeoJSON)
import pandas as pd  # For data manipulation and handling tabular data
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting results
from sklearn.ensemble import GradientBoostingClassifier  # Gradient Boosting
from sklearn.metrics import accuracy_score, f1_score  # Evaluation metrics
from sklearn.utils.class_weight import compute_sample_weight

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

# ----------------- Step 4: Extract Geometric Features -----------------
# Compute various spatial features for both train and test datasets

# Area of the shape (only meaningful for polygons)
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

# ----------------- Step 5: Feature Selection and Data Cleaning -----------------
# Define the list of features to be used for training
feature_cols = ["area", "centroid_x", "centroid_y", "perimeter", "bounding_width", "bounding_height", "aspect_ratio", "circularity"]

# Extract features from train and test datasets
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

# Convert the change_type column to numerical values using the predefined mapping
train_y = train_df['change_type'].apply(lambda x: change_type_map[x])

# Print the number of NaN and Inf values in the dataset to ensure all issues are handled
print(np.isnan(train_x).sum())  # Count NaN values per feature
print(np.isinf(train_x).sum())  # Count Inf values per feature

print(train_x.columns)
print(test_x.columns)

# Convert DataFrames to NumPy arrays for machine learning models
train_x = train_x.values
test_x = test_x.values

# ----------------- Step 6: Train the Gradient Boosting Model -----------------
# Set hyperparameters for the Gradient Boosting model
num_estimators = 25  # Increase number of estimators for better learning
learning_rate = 0.05  # Lower learning rate improves generalization
max_depth = 4  # Keep tree depth balanced for performance

# Create and train the Gradient Boosting model with class balancing
gbm_model = GradientBoostingClassifier(
    n_estimators=num_estimators, 
    learning_rate=learning_rate, 
    max_depth=max_depth
)

# Train the model
gbm_model.fit(train_x, train_y)

# ----------------- Step 7: Model Prediction -----------------
# Use the trained Gradient Boosting model to make predictions on the test dataset
pred_y = gbm_model.predict(test_x)

# ----------------- Step 8: Save Results -----------------
# Save predictions to a CSV file for submission
pred_df = pd.DataFrame(pred_y, columns=['change_type'])
pred_df.to_csv("gradient_boosting_submission.csv", index=True, index_label='Id')

# ----------------- Step 9: Model Evaluation -----------------
# Print training accuracy
train_accuracy = accuracy_score(train_y, gbm_model.predict(train_x))
print(f"Training Accuracy: {train_accuracy:.4f}")

# Compute and print the Mean F1-Score (Macro)
train_f1 = f1_score(train_y, gbm_model.predict(train_x), average='macro')
print(f"Mean F1-Score (Train - Macro): {train_f1:.4f}")

# Compute and print the Weighted F1-Score
train_f1_weighted = f1_score(train_y, gbm_model.predict(train_x), average='weighted')
print(f"Weighted Mean F1-Score (Train): {train_f1_weighted:.4f}")

# ----------------- Step 10: Plot Training Loss -----------------
# Plot negative log-likelihood loss over iterations
plt.plot(np.arange(1, num_estimators + 1), gbm_model.train_score_, label="Training Loss")
plt.xlabel("Number of Estimators")
plt.ylabel("Loss")
plt.title("Gradient Boosting Training Loss Over Time")
plt.legend()
plt.show()