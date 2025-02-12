import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
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

# Remove invalid (null) geometries
train_df = train_df[train_df["geometry"].notnull()]
test_df = test_df[test_df["geometry"].notnull()]
train_df = train_df[train_df["geometry"].is_valid]
test_df = test_df[test_df["geometry"].is_valid]

# ----------------- Step 4: Basic Feature Engineering Example -----------------
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
train_df.drop(columns=["geometry"], inplace=True)
test_df.drop(columns=["geometry"], inplace=True)

# Pick sample rows for debugging
sample_rows = train_df.sample(3, random_state=42).index
print("\n===== RAW DATA SAMPLE =====")
print(train_df.loc[sample_rows, ["change_type", "urban_type", "geography_type", "date0"]])
# Print after feature engineering
print("\n===== AFTER FEATURE ENGINEERING =====")
print(train_df.loc[sample_rows, ["area", "centroid_x", "centroid_y", "perimeter"]])
print(train_df.columns)
# ----------------- Step 5: Convert Labels -----------------
train_y = train_df['change_type'].map(change_type_map).astype(int)

# ----------------- Step 6: Multi-Label Binarize 'geography_type' & 'urban_type' -----------------
def split_on_commas(series, fill_value="Unknown"):
    """Split a column by commas into lists (for multi-label data)."""
    return (
        series.fillna(fill_value)
        .apply(lambda x: x.split(","))  # e.g. "Dense Forest,Grass Land" => ["Dense Forest", "Grass Land"]
    )

# Split both columns on commas
train_df["urban_list"] = split_on_commas(train_df["urban_type"])
test_df["urban_list"] = split_on_commas(test_df["urban_type"])

train_df["geo_list"] = split_on_commas(train_df["geography_type"])
test_df["geo_list"] = split_on_commas(test_df["geography_type"])

# MultiLabelBinarizer for 'urban_type'
mlb_urban = MultiLabelBinarizer()
train_urban = mlb_urban.fit_transform(train_df["urban_list"])
test_urban = mlb_urban.transform(test_df["urban_list"])

urban_cols = [f"urban_{cat.strip()}" for cat in mlb_urban.classes_]
train_urban_df = pd.DataFrame(train_urban, columns=urban_cols, index=train_df.index)
test_urban_df = pd.DataFrame(test_urban, columns=urban_cols, index=test_df.index)

# MultiLabelBinarizer for 'geography_type'
mlb_geo = MultiLabelBinarizer()
train_geo = mlb_geo.fit_transform(train_df["geo_list"])
test_geo = mlb_geo.transform(test_df["geo_list"])

geo_cols = [f"geo_{cat.strip()}" for cat in mlb_geo.classes_]
train_geo_df = pd.DataFrame(train_geo, columns=geo_cols, index=train_df.index)
test_geo_df = pd.DataFrame(test_geo, columns=geo_cols, index=test_df.index)

# Concatenate new columns, drop the originals
train_df = pd.concat([train_df, train_urban_df, train_geo_df], axis=1).drop(
    ["urban_type", "urban_list", "geography_type", "geo_list"], axis=1
)
test_df = pd.concat([test_df, test_urban_df, test_geo_df], axis=1).drop(
    ["urban_type", "urban_list", "geography_type", "geo_list"], axis=1
)
print("\n===== AFTER MULTI-LABEL ENCODING =====")
print(train_df.loc[sample_rows, list(train_urban_df.columns) + list(train_geo_df.columns)])
# Save after multi-label encoding
train_df.loc[sample_rows, urban_cols + geo_cols].to_csv("multi_label_encoding_sample.csv")
print(train_df.columns)
# ----------------- Step 7: Date Handling (Example) -----------------
s = ["date0", "date1", "date2", "date3", "date4"]

for col in s:
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
print("\n===== AFTER DATE HANDLING =====")
print(train_df.loc[sample_rows, ["date0_year", "date0_month", "date0_day"]])
train_df.loc[sample_rows, ["date0_year", "date0_month", "date0_day"]].to_csv("date_handling_sample.csv")
# ----------------- Step 8: (Optional) Log-transform Certain Numerical Features -----------------
log_features = ["area", "perimeter", "bounding_width", "bounding_height"]
for col in log_features:
    train_df[col] = np.log1p(np.maximum(train_df[col], 1e-9))
    test_df[col] = np.log1p(np.maximum(test_df[col], 1e-9))

# ----------------- Step : Processing change_status_date -----------------
# Get all unique values from the three columns across train and test sets
unique_status_0 = set(train_df["change_status_date0"].dropna().unique()).union(set(test_df["change_status_date0"].dropna().unique()))
unique_status_1 = set(train_df["change_status_date1"].dropna().unique()).union(set(test_df["change_status_date1"].dropna().unique()))
unique_status_2 = set(train_df["change_status_date2"].dropna().unique()).union(set(test_df["change_status_date2"].dropna().unique()))
unique_status_3 = set(train_df["change_status_date3"].dropna().unique()).union(set(test_df["change_status_date3"].dropna().unique()))
unique_status_4 = set(train_df["change_status_date4"].dropna().unique()).union(set(test_df["change_status_date4"].dropna().unique()))

# Combine all unique values into a single set
all_unique_statuses = unique_status_1.union(unique_status_2).union(unique_status_3).union(unique_status_4).union(unique_status_0)

# Print all unique values
print("Unique values for change_status_date1, change_status_date2, change_status_date3:")
print(all_unique_statuses)
# Create a mapping dictionary
status_mapping = {status: idx for idx, status in enumerate(sorted(all_unique_statuses))}

# Print the mapping
print("\nMapping of change_status_date values to integers:")
print(status_mapping)
# Apply mapping to the datasets
for col in ["change_status_date1", "change_status_date2", "change_status_date3", "change_status_date4", "change_status_date0"]:
    train_df[col] = train_df[col].map(status_mapping)
    test_df[col] = test_df[col].map(status_mapping)

# ----------------- Step 9: Build Final Feature List -----------------
drop_cols = ["geometry", "change_type"]
feature_cols =  [col for col in train_df.columns if "geometry" not in col]
print(feature_cols)
print(train_df)
# ----------------- Step 5: Check Missing/Invalid Data -----------------
print("=== Missing Value Counts (Train) ===")
print(train_df.isna().sum())
print("\n=== Negative or Zero Counts for Key Geometries (Train) ===")
print((train_df[["area", "perimeter", "bounding_width", "bounding_height"]] <= 0).sum())
print(feature_cols)
train_x = train_df
test_x = test_df

print("\n===== AFTER LOG TRANSFORMATION =====")
print(train_df.loc[sample_rows, feature_cols])

print(feature_cols)
train_df["change_type"] = train_df["change_type"].map(change_type_map).astype(int)
train_y = train_df["change_type"]
# ----------------- Step 10: Handle Missing & Infinite Values -----------------
train_x = train_x.fillna(train_x.median())
test_x = test_x.fillna(test_x.median())

train_x = np.nan_to_num(train_x, nan=0.0, posinf=1e10, neginf=-1e10)
test_x = np.nan_to_num(test_x, nan=0.0, posinf=1e10, neginf=-1e10)

# ----------------- Step 11: Standardize Features -----------------
# Identify date-related columns (exclude from standardization)
print(feature_cols)
print(train_x)
train_df = pd.DataFrame(train_x, columns=feature_cols)
train_df = train_df[[col for col in feature_cols if "change_type" not in col]]
test_df = pd.DataFrame(test_x, columns=[col for col in feature_cols if "change_type" not in col])
feature_cols =  [col for col in train_df.columns if "_day" in col or "_month" in col or "_year" in col or "geo_" in col or "urban_" in col or "img_" in col or "geometry" in col or "change_status" in col or "index" in col]

# Select columns for standardization (exclude date-related columns)
standardized_cols = [col for col in train_df.columns if col not in feature_cols]

# Standardize only the selected columns
scaler = StandardScaler()
train_x_scaled_standardized = scaler.fit_transform(train_df[standardized_cols])
test_x_scaled_standardized = scaler.transform(test_df[[col for col in standardized_cols if "change_type" not in col]])

# Convert back to DataFrame
train_x_scaled_df = pd.DataFrame(train_x_scaled_standardized, columns=standardized_cols, index=train_df.index)
test_x_scaled_df = pd.DataFrame(test_x_scaled_standardized, columns=standardized_cols, index=test_df.index)
print(train_x_scaled_df.columns)
# Concatenate with date-related columns (which remain unchanged)
train_x_scaled = pd.concat([train_x_scaled_df, train_df[feature_cols]], axis=1)
test_x_scaled = pd.concat([test_x_scaled_df, test_df[[col for col in feature_cols if "change_type" not in col]]], axis=1)
print(train_x_scaled.columns)
print(feature_cols)
print("\n===== AFTER STANDARDIZATION =====")
#print(train_x_scaled[sample_rows])
pd.DataFrame(train_x_scaled).loc[sample_rows].to_csv("standardization_sample.csv")
# ----------------- Step 12: (Optional) Feature Selection -----------------
selector = SelectKBest(score_func=f_classif, k=35)
train_x_selected = selector.fit_transform(train_x_scaled, train_y)
test_x_selected = selector.transform(test_x_scaled)

feature_cols =  [col for col in train_x_scaled.columns if "geometry" not in col]
selected_feature_indices = selector.get_support(indices=True)
selected_features = [feature_cols[i] for i in selected_feature_indices]
print("\nSelected Features (SelectKBest):")
print(selected_features)
print("\n===== FINAL =====")
print(train_x_selected[sample_rows])
pd.DataFrame(train_x_selected).loc[sample_rows].to_csv("feature_selection_sample.csv")
# ----------------- Step 13: Model Training (Gradient Boosting) -----------------
num_estimators = 100
learning_rate = 0.03
max_depth = 4
subsample = 0.75

gbm_model = GradientBoostingClassifier(
    n_estimators=num_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    subsample=subsample
)

gbm_model.fit(train_x_selected, train_y)

# ----------------- Step 14: Predictions -----------------
pred_y = gbm_model.predict(test_x_selected)

# ----------------- Step 15: Save Results -----------------
pred_df = pd.DataFrame(pred_y, columns=['change_type'])
pred_df.to_csv("gradient_boosting_submission.csv", index=True, index_label='Id')

# ----------------- Step 16: Evaluate Model -----------------
train_preds = gbm_model.predict(train_x_selected)
train_accuracy = accuracy_score(train_y, train_preds)
train_f1_macro = f1_score(train_y, train_preds, average='macro')
train_f1_weighted = f1_score(train_y, train_preds, average='weighted')

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"F1 (Macro): {train_f1_macro:.4f}")
print(f"F1 (Weighted): {train_f1_weighted:.4f}")

# ----------------- Step 17: Plot Training Loss -----------------
plt.plot(range(1, num_estimators + 1), gbm_model.train_score_, label="Training Loss")
plt.xlabel("Number of Estimators")
plt.ylabel("Loss")
plt.title("Gradient Boosting Training Loss Over Time")
plt.legend()
plt.show()


def apply_pca(train_x_scaled, test_x_scaled):
    # Define all feature columns excluding 'change_type' (target variable)
    feature_cols = [col for col in train_x_scaled.columns if col != "change_type"]

    # Apply PCA for dimensionality reduction (reduce to 35 principal components)
    n_components = 35
    pca = PCA(n_components=n_components)

    # Fit PCA on training data and transform both train and test datasets
    train_x_selected = pca.fit_transform(train_x_scaled[feature_cols])
    test_x_selected = pca.transform(test_x_scaled[feature_cols])

    # Print explained variance ratio to check how much variance is retained
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"\nTotal Explained Variance by {n_components} Components: {explained_variance:.4f}")

    # Create feature names for PCA components
    selected_features = [f"PC{i+1}" for i in range(n_components)]
    print("\nSelected PCA Features:", selected_features)

    # Convert back to DataFrame with PCA components
    train_x_selected_df = pd.DataFrame(train_x_selected, columns=selected_features, index=train_x_scaled.index)
    test_x_selected_df = pd.DataFrame(test_x_selected, columns=selected_features, index=test_x_scaled.index)

    # Convert to NumPy arrays for model training
    train_x_selected = train_x_selected_df.values
    test_x_selected = test_x_selected_df.values
    return train_x_scaled, test_x_scaled

#train_x_selected,test_x_selected = apply_pca(train_x_scaled, test_x_scaled)