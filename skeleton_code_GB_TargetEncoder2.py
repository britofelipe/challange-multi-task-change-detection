import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# ----------------- Step 1: Define Change Type Mapping -----------------
change_type_map = {
    'Demolition': 0, 
    'Road': 1, 
    'Residential': 2, 
    'Commercial': 3,
    'Industrial': 4, 
    'Mega Projects': 5
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
train_df["centroid_x"] = train_df["geometry"].centroid.x
train_df["centroid_y"] = train_df["geometry"].centroid.y
train_df["perimeter"] = train_df["geometry"].length
train_df["bounding_width"] = (
    train_df["geometry"].bounds["maxx"] - train_df["geometry"].bounds["minx"]
)
train_df["bounding_height"] = (
    train_df["geometry"].bounds["maxy"] - train_df["geometry"].bounds["miny"]
)
train_df["aspect_ratio"] = train_df["bounding_width"] / np.maximum(train_df["bounding_height"], 1e-9)
train_df["circularity"] = (4 * np.pi * train_df["area"]) / (train_df["perimeter"] ** 2)

test_df["area"] = test_df["geometry"].area
test_df["centroid_x"] = test_df["geometry"].centroid.x
test_df["centroid_y"] = test_df["geometry"].centroid.y
test_df["perimeter"] = test_df["geometry"].length
test_df["bounding_width"] = (
    test_df["geometry"].bounds["maxx"] - test_df["geometry"].bounds["minx"]
)
test_df["bounding_height"] = (
    test_df["geometry"].bounds["maxy"] - test_df["geometry"].bounds["miny"]
)
test_df["aspect_ratio"] = test_df["bounding_width"] / np.maximum(test_df["bounding_height"], 1e-9)
test_df["circularity"] = (4 * np.pi * test_df["area"]) / (test_df["perimeter"] ** 2)

# ----------------- Step 5: Check Missing/Invalid Data -----------------
print("=== Missing Value Counts (Train) ===")
print(train_df.isna().sum())
print("\n=== Negative or Zero Counts for Key Geometries (Train) ===")
print((train_df[["area", "perimeter", "bounding_width", "bounding_height"]] <= 0).sum())

# ----------------- Step 6: Categorical Features & Label Conversion -----------------
cat_features = ["urban_type", "geography_type"]
train_y = train_df["change_type"].map(change_type_map).astype(int)

# ----------------- Step 7: Date Decomposition (Day/Month/Year) -----------------
date_features = ["date0", "date1", "date2", "date3", "date4"]

for col in date_features:
    # Convert to datetime
    train_df[col] = pd.to_datetime(train_df[col], format='%d-%m-%Y', errors='coerce')
    test_df[col] = pd.to_datetime(test_df[col], format='%d-%m-%Y', errors='coerce')

    # Extract day, month, and year
    train_df[f"{col}_day"] = train_df[col].dt.day
    train_df[f"{col}_month"] = train_df[col].dt.month
    train_df[f"{col}_year"] = train_df[col].dt.year

    test_df[f"{col}_day"] = test_df[col].dt.day
    test_df[f"{col}_month"] = test_df[col].dt.month
    test_df[f"{col}_year"] = test_df[col].dt.year

    # Drop the original date column
    train_df.drop(columns=[col], inplace=True)
    test_df.drop(columns=[col], inplace=True)

# ----------------- Step 8: One-Hot Encode Both Categorical Features -----------------
train_ohe = pd.get_dummies(train_df[cat_features], prefix=cat_features)
test_ohe = pd.get_dummies(test_df[cat_features], prefix=cat_features)

# Align columns (if train/test differ in categories)
train_ohe, test_ohe = train_ohe.align(test_ohe, join='outer', axis=1, fill_value=0)

# Concatenate OHE columns with original data
train_df = pd.concat([train_df, train_ohe], axis=1)
test_df = pd.concat([test_df, test_ohe], axis=1)

# Drop the original categorical columns
train_df.drop(columns=cat_features, inplace=True)
test_df.drop(columns=cat_features, inplace=True)

# ----------------- Step 9: Build Final Feature List -----------------
# Get the OHE column names
ohe_cols = list(train_ohe.columns)

feature_cols = (
    ohe_cols
    + [
        "area", "centroid_x", "centroid_y", 
        "perimeter", "bounding_width", "bounding_height", 
        "aspect_ratio", "circularity",
        "img_red_mean_date1", "img_green_mean_date1", "img_blue_mean_date1",
        "img_red_std_date1", "img_green_std_date1", "img_blue_std_date1",
        "img_red_mean_date2", "img_green_mean_date2", "img_blue_mean_date2",
        "img_red_std_date2", "img_green_std_date2", "img_blue_std_date2",
        # The newly created day/month/year columns:
        "date0_day", "date0_month", "date0_year",
        "date1_day", "date1_month", "date1_year",
        "date2_day", "date2_month", "date2_year",
        "date3_day", "date3_month", "date3_year",
        "date4_day", "date4_month", "date4_year",
    ]
)

train_x = train_df[feature_cols]
test_x = test_df[feature_cols]

# ----------------- Step 10: Handle Missing & Infinite Values -----------------
train_x = train_x.fillna(train_x.median())
test_x = test_x.fillna(test_x.median())

train_x = np.nan_to_num(train_x, nan=0.0, posinf=1e10, neginf=-1e10)
test_x = np.nan_to_num(test_x, nan=0.0, posinf=1e10, neginf=-1e10)

# ----------------- Step 11: (Optional) Log-transform numeric columns -----------------
# e.g. If you want to log-transform these
log_features = ["area", "perimeter", "bounding_width", "bounding_height"]
for col in log_features:
    train_x[col] = np.log1p(np.maximum(train_x[col], 1e-9))  
    test_x[col] = np.log1p(np.maximum(test_x[col], 1e-9))

# ----------------- Step 12: Standardize Features -----------------
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# ----------------- Step 13: Feature Selection (SelectKBest) -----------------
selector = SelectKBest(score_func=f_classif, k=14)  
train_x_selected = selector.fit_transform(train_x_scaled, train_y)  
test_x_selected = selector.transform(test_x_scaled)

selected_feature_indices = selector.get_support(indices=True)
selected_features = [feature_cols[i] for i in selected_feature_indices]
print(f"Selected Features: {selected_features}")

# ===========================================================================
# =           NOW, REINSERT THE MODEL TRAINING / EVALUATION CODE            =
# ===========================================================================
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

# Train
gbm_model.fit(train_x_selected, train_y)

# Predict
pred_y = gbm_model.predict(test_x_selected)

# Save results
pred_df = pd.DataFrame(pred_y, columns=['change_type'])
pred_df.to_csv("gradient_boosting_submission.csv", index=True, index_label='Id')

# Evaluate on train set
train_accuracy = accuracy_score(train_y, gbm_model.predict(train_x_selected))
train_f1 = f1_score(train_y, gbm_model.predict(train_x_selected), average='macro')
train_f1_weighted = f1_score(train_y, gbm_model.predict(train_x_selected), average='weighted')

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Mean F1-Score (Train - Macro): {train_f1:.4f}")
print(f"Weighted Mean F1-Score (Train): {train_f1_weighted:.4f}")

# Plot training loss
plt.plot(np.arange(1, num_estimators + 1), gbm_model.train_score_, label="Training Loss")
plt.xlabel("Number of Estimators")
plt.ylabel("Loss")
plt.title("Gradient Boosting Training Loss Over Time")
plt.legend()
plt.show()