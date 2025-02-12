import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV

# ----------------- Step 1: Define Change Type Mapping -----------------
change_type_map = {
    'Demolition': 0,
    'Road': 1,
    'Residential': 2,
    'Commercial': 3,
    'Industrial': 4,
    'Mega Projects': 5
}

# ----------------- Step 2: Read & Preprocess GeoJSON Files -----------------
def load_and_preprocess_geojson(filename):
    df = gpd.read_file(filename, index_col=0)

    # Check if the current CRS is geographic (EPSG:4326) and reproject
    if df.crs.is_geographic:
        df = df.to_crs(epsg=3857)  # Convert to Web Mercator (or another appropriate projected CRS)

    print(f"Number of samples in {filename}: {df.shape[0]}")
    return df

train_df = load_and_preprocess_geojson('train.geojson')
test_df = load_and_preprocess_geojson('test.geojson')
print(f"Number of samples in train dataset: {train_df.shape[0]}")
print(f"Number of samples in test dataset: {test_df.shape[0]}")

# ----------------- Step 3: Feature Engineering -----------------
def extract_geometric_features(df):
    """Compute geometric features only for valid geometries, filling missing values with zero."""

    valid_geometry = df["geometry"].notnull() & df["geometry"].is_valid

    # Compute centroid only for valid geometries
    df.loc[valid_geometry, "centroid_x"] = df.loc[valid_geometry, "geometry"].centroid.x
    df.loc[valid_geometry, "centroid_y"] = df.loc[valid_geometry, "geometry"].centroid.y

    df.loc[valid_geometry, "area"] = df.loc[valid_geometry, "geometry"].area
    df.loc[valid_geometry, "perimeter"] = df.loc[valid_geometry, "geometry"].length

    df.loc[valid_geometry, "bounding_width"] = (
        df.loc[valid_geometry, "geometry"].bounds["maxx"] - df.loc[valid_geometry, "geometry"].bounds["minx"]
    )
    df.loc[valid_geometry, "bounding_height"] = (
        df.loc[valid_geometry, "geometry"].bounds["maxy"] - df.loc[valid_geometry, "geometry"].bounds["miny"]
    )

    df.loc[valid_geometry, "aspect_ratio"] = (
        df.loc[valid_geometry, "bounding_width"] / np.maximum(df.loc[valid_geometry, "bounding_height"], 1e-9)
    )
    df.loc[valid_geometry, "circularity"] = (
        (4 * np.pi * df.loc[valid_geometry, "area"]) / np.maximum(df.loc[valid_geometry, "perimeter"] ** 2, 1e-9)
    )

    # Fill NaN values with 0 for rows where geometry is missing or invalid
    df.fillna(0, inplace=True)

    # Drop the geometry column after processing
    df.drop(columns=["geometry"], inplace=True)

extract_geometric_features(train_df)
extract_geometric_features(test_df)

# ----------------- Step 4: Convert Labels -----------------
train_df["change_type"] = train_df["change_type"].map(change_type_map).astype(int)
train_y = train_df["change_type"]

# ----------------- Step 5: Multi-Label Binarization -----------------
def split_on_commas(series):
    return series.fillna("Unknown").apply(lambda x: x.split(","))

train_df["urban_list"] = split_on_commas(train_df["urban_type"])
test_df["urban_list"] = split_on_commas(test_df["urban_type"])
train_df["geo_list"] = split_on_commas(train_df["geography_type"])
test_df["geo_list"] = split_on_commas(test_df["geography_type"])

def apply_mlb(df, mlb, column_name):
    return pd.DataFrame(mlb.fit_transform(df[column_name]), columns=[f"{column_name}_{cat.strip()}" for cat in mlb.classes_], index=df.index)

mlb_urban = MultiLabelBinarizer()
mlb_geo = MultiLabelBinarizer()

train_urban_df = apply_mlb(train_df, mlb_urban, "urban_list")
test_urban_df = apply_mlb(test_df, mlb_urban, "urban_list")
train_geo_df = apply_mlb(train_df, mlb_geo, "geo_list")
test_geo_df = apply_mlb(test_df, mlb_geo, "geo_list")

train_df = pd.concat([train_df, train_urban_df, train_geo_df], axis=1).drop(["urban_type", "urban_list", "geography_type", "geo_list"], axis=1)
test_df = pd.concat([test_df, test_urban_df, test_geo_df], axis=1).drop(["urban_type", "urban_list", "geography_type", "geo_list"], axis=1)

# ----------------- Step 6: Date Handling -----------------
date_cols = ["date0", "date1", "date2", "date3", "date4"]

for col in date_cols:
    train_df[col] = pd.to_datetime(train_df[col], format='%d-%m-%Y', errors='coerce')
    test_df[col] = pd.to_datetime(test_df[col], format='%d-%m-%Y', errors='coerce')

    train_df[f"{col}_day"] = train_df[col].dt.day
    train_df[f"{col}_month"] = train_df[col].dt.month
    train_df[f"{col}_year"] = train_df[col].dt.year

    test_df[f"{col}_day"] = test_df[col].dt.day
    test_df[f"{col}_month"] = test_df[col].dt.month
    test_df[f"{col}_year"] = test_df[col].dt.year

train_df.drop(columns=date_cols, inplace=True)
test_df.drop(columns=date_cols, inplace=True)

# ----------------- Step 7: Process `change_status_date` -----------------
status_cols = ["change_status_date0", "change_status_date1", "change_status_date2", "change_status_date3", "change_status_date4"]

# Collect all unique values
all_unique_statuses = set()
for col in status_cols:
    all_unique_statuses.update(train_df[col].dropna().astype(str).unique())
    all_unique_statuses.update(test_df[col].dropna().astype(str).unique())

# Create a mapping
status_mapping = {status: idx for idx, status in enumerate(sorted(all_unique_statuses))}

# Apply the mapping
for col in status_cols:
    train_df[col] = train_df[col].map(status_mapping)
    test_df[col] = test_df[col].map(status_mapping)

# ----------------- Step 8: Log-Transform Certain Features -----------------
log_features = ["area", "perimeter", "bounding_width", "bounding_height"]
for col in log_features:
    train_df[col] = np.log1p(np.maximum(train_df[col], 1e-9))
    test_df[col] = np.log1p(np.maximum(test_df[col], 1e-9))

# ----------------- Step 9: Handle Missing & Infinite Values -----------------
train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)

train_df.replace([np.inf, -np.inf], 0, inplace=True)
test_df.replace([np.inf, -np.inf], 0, inplace=True)

# ----------------- Step 10: Standardize Features (Excluding Dates & Categorical Data) -----------------
exclude_cols = [col for col in train_df.columns if "_day" in col or "_month" in col or "_year" in col or "urban_" in col or "geo_" in col or "change_status_date" in col]
standardized_cols = [col for col in train_df.columns if col not in exclude_cols + ["change_type"]]

scaler = StandardScaler()
train_x_scaled_df = pd.DataFrame(scaler.fit_transform(train_df[standardized_cols]), columns=standardized_cols, index=train_df.index)
test_x_scaled_df = pd.DataFrame(scaler.transform(test_df[standardized_cols]), columns=standardized_cols, index=test_df.index)

train_x_scaled = pd.concat([train_x_scaled_df, train_df[exclude_cols]], axis=1)
test_x_scaled = pd.concat([test_x_scaled_df, test_df[exclude_cols]], axis=1)

# ----------------- Step 11: Feature Selection -----------------
def select_features(train_x_scaled, test_x_scaled, k):
    # Define all feature columns excluding 'change_type' (target variable)
    feature_cols = [col for col in train_x_scaled.columns if col != "change_type"]

    # Apply SelectKBest across ALL features
    selector = SelectKBest(score_func=f_classif, k=k)
    train_x_selected = selector.fit_transform(train_x_scaled[feature_cols], train_y)
    test_x_selected = selector.transform(test_x_scaled[feature_cols])

    # Get selected feature names (from all features, not just standardized ones)
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    print("\nSelected Features:", selected_features)
    return train_x_selected,test_x_selected

train_x_selected,test_x_selected = select_features(train_x_scaled, test_x_scaled, 35)

#rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=35) #Its supposed to be better but takes a lot of time to compute
#train_x_selected = rfe.fit_transform(train_x_scaled, train_y)
#test_x_selected = rfe.transform(test_x_scaled)
#print(train_x_selected.columns)

# ----------------- Step 12: Model Training (Gradient Boosting) -----------------
""" 
param_grid = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.01, 0.03, 0.1],
    'max_depth': [3, 4, 6],
    'subsample': [0.7, 0.75, 0.85]
}

gbm_grid = GridSearchCV(GradientBoostingClassifier(), param_grid, scoring='f1_macro', cv=5, n_jobs=-1)
gbm_grid.fit(train_x_selected, train_y)

print("Best Parameters:", gbm_grid.best_params_)
gbm_model = gbm_grid.best_estimator_ 
num_estimators = gbm_grid.best_params_['n_estimators']
"""

num_estimators = 100
gbm_model = GradientBoostingClassifier(n_estimators=num_estimators, learning_rate=0.03, max_depth=4, subsample=0.75)
gbm_model.fit(train_x_selected, train_y)

# ----------------- Step 13: Predictions & Evaluation -----------------
pred_y = gbm_model.predict(test_x_selected)

train_preds = gbm_model.predict(train_x_selected)
train_accuracy = accuracy_score(train_y, train_preds)
train_f1_macro = f1_score(train_y, train_preds, average='macro')
train_f1_weighted = f1_score(train_y, train_preds, average='weighted')

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"F1 (Macro): {train_f1_macro:.4f}")
print(f"F1 (Weighted): {train_f1_weighted:.4f}")

plt.plot(range(1, num_estimators + 1), gbm_model.train_score_, label="Training Loss")
plt.xlabel("Number of Estimators")
plt.ylabel("Loss")
plt.title("Gradient Boosting Training Loss Over Time")
plt.legend()
plt.show()

# ----------------- Step 14: Save Results -----------------
pd.DataFrame(pred_y, columns=['change_type']).to_csv("gradient_boosting_submission.csv", index=True, index_label='Id')