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
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from joblib import Parallel, delayed
import re
from collections import defaultdict

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

    df.loc[valid_geometry, "compactness"] = (
        (4 * np.pi * df.loc[valid_geometry, "area"]) / (df.loc[valid_geometry, "perimeter"] ** 2 + 1e-9)
    )

    df.loc[valid_geometry, "bounding_width"] = (
        df.loc[valid_geometry, "geometry"].bounds["maxx"] - df.loc[valid_geometry, "geometry"].bounds["minx"]
    )
    df.loc[valid_geometry, "bounding_height"] = (
        df.loc[valid_geometry, "geometry"].bounds["maxy"] - df.loc[valid_geometry, "geometry"].bounds["miny"]
    )

    df.loc[valid_geometry, "bounding_diagonal"] = np.sqrt(
        df.loc[valid_geometry, "bounding_width"] ** 2 + df.loc[valid_geometry, "bounding_height"] ** 2
    )

    df.loc[valid_geometry, "elongation"] = df.loc[valid_geometry, "bounding_width"] / np.maximum(
        df.loc[valid_geometry, "bounding_height"], 1e-9
    )

    df.loc[valid_geometry, "convexity"] = df.loc[valid_geometry, "geometry"].apply(
        lambda g: g.area / g.convex_hull.area if g else 0
    )

    # Momentos de Hu (mantém as features existentes e adiciona novos momentos)
    def hu_moments(polygon):
        try:
            if polygon.is_empty:
                return [0] * 7
            coords = np.array(polygon.exterior.coords).astype(np.float32)
            moments = cv2.moments(coords)
            hu_moments = cv2.HuMoments(moments).flatten()
            return hu_moments.tolist()
        except:
            return [0] * 7

    df["hu_moments"] = df["geometry"].apply(hu_moments)
    for i in range(7):
        df[f"hu_moment_{i}"] = df["hu_moments"].apply(lambda x: x[i])
        df[f"hu_moment_{i}_log"] = np.sign(df[f"hu_moment_{i}"]) * np.log1p(np.abs(df[f"hu_moment_{i}"]))

    df.fillna(0, inplace=True)
    df.drop(columns=["geometry", "hu_moments"], inplace=True)

    df.loc[valid_geometry, "aspect_ratio"] = (
        df.loc[valid_geometry, "bounding_width"] / np.maximum(df.loc[valid_geometry, "bounding_height"], 1e-9)
    )
    df.loc[valid_geometry, "circularity"] = (
        (4 * np.pi * df.loc[valid_geometry, "area"]) / np.maximum(df.loc[valid_geometry, "perimeter"] ** 2, 1e-9)
    )

    # Fill NaN values with 0 for rows where geometry is missing or invalid
    df.fillna(0, inplace=True)

    # Drop the geometry column after processing
    if "geometry" in df.columns:
        df.drop(columns=["geometry"], inplace=True)


extract_geometric_features(train_df)
extract_geometric_features(test_df)

def extract_spectral_features(df):
    """Adiciona índices espectrais do segundo código ao primeiro código."""
    for d in range(5):
        if f'img_red_mean_date{d}' in df.columns and f'img_blue_mean_date{d}' in df.columns:
            df[f'NDVI_date{d}'] = (df[f'img_red_mean_date{d}'] - df[f'img_blue_mean_date{d}']) / \
                                  (df[f'img_red_mean_date{d}'] + df[f'img_blue_mean_date{d}'] + 1e-9)

        if f'img_red_mean_date{d}' in df.columns and f'img_green_mean_date{d}' in df.columns:
            df[f'NDBI_date{d}'] = (df[f'img_red_mean_date{d}'] - df[f'img_green_mean_date{d}']) / \
                                  (df[f'img_red_mean_date{d}'] + df[f'img_green_mean_date{d}'] + 1e-9)

        if f'img_green_mean_date{d}' in df.columns and f'img_blue_mean_date{d}' in df.columns:
            df[f'NDWI_date{d}'] = (df[f'img_green_mean_date{d}'] - df[f'img_blue_mean_date{d}']) / \
                                  (df[f'img_green_mean_date{d}'] + df[f'img_blue_mean_date{d}'] + 1e-9)

extract_spectral_features(train_df)  
extract_spectral_features(test_df) 
print("---------DEU CERTO AS ESPECTRAIS---------")

# ----------------- Step 4: Convert Labels -----------------
train_df["change_type"] = train_df["change_type"].map(change_type_map).astype(int)
train_y = train_df["change_type"]


# ----------------- Step 5: Multi-Label Binarization & One-Hot Encoding -----------------
def split_on_commas(series):
    """Splits a column by commas into lists for MultiLabelBinarization."""
    return series.fillna("Unknown").apply(lambda x: x.split(","))

# Create MultiLabelBinarization lists
train_df["urban_list"] = split_on_commas(train_df["urban_type"])
test_df["urban_list"] = split_on_commas(test_df["urban_type"])
train_df["geo_list"] = split_on_commas(train_df["geography_type"])
test_df["geo_list"] = split_on_commas(test_df["geography_type"])

# Function to apply MultiLabelBinarizer (MLB)
def apply_mlb(df, mlb, column_name):
    return pd.DataFrame(
        mlb.fit_transform(df[column_name]), 
        columns=[f"{column_name}_mlb_{cat.strip()}" for cat in mlb.classes_], 
        index=df.index
    )

# Apply MultiLabelBinarizer (MLB)
mlb_urban = MultiLabelBinarizer()
mlb_geo = MultiLabelBinarizer()

train_urban_mlb_df = apply_mlb(train_df, mlb_urban, "urban_list")
test_urban_mlb_df = apply_mlb(test_df, mlb_urban, "urban_list")
train_geo_mlb_df = apply_mlb(train_df, mlb_geo, "geo_list")
test_geo_mlb_df = apply_mlb(test_df, mlb_geo, "geo_list")

# Drop original columns after encoding
train_df.drop([ "urban_list", "geo_list"], axis=1, inplace=True)
test_df.drop(["urban_list", "geo_list"], axis=1, inplace=True)

print("✅ MultiLabelBinarization and One-Hot Encoding applied successfully!")
# ----------------- Step 5.2: One-Hot Encoding (OHE) on Original Columns -----------------
categorical_cols = ["urban_type", "geography_type"]

# One-Hot Encoding for categorical features
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit on train, transform train & test
train_ohe = ohe.fit_transform(train_df[categorical_cols])
test_ohe = ohe.transform(test_df[categorical_cols])

# Convert to DataFrame
train_ohe_df = pd.DataFrame(train_ohe, columns=ohe.get_feature_names_out(categorical_cols), index=train_df.index)
test_ohe_df = pd.DataFrame(test_ohe, columns=ohe.get_feature_names_out(categorical_cols), index=test_df.index)

# Concatenate OneHotEncoded Data with original DataFrame
train_df = pd.concat([train_df, train_ohe_df], axis=1)
test_df = pd.concat([test_df, test_ohe_df], axis=1)

# Drop original categorical columns
train_df.drop(columns=categorical_cols, inplace=True)
test_df.drop(columns=categorical_cols, inplace=True)

# Ensure both datasets have the same columns (fill missing with 0 in test set)
missing_cols = set(train_df.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0  # Add missing columns with value 0

# Ensure column order is the same
test_df = test_df[train_df.columns]

print("✅ OneHotEncoding Applied Successfully!")


def clean_feature_names(df):
    """Replace special characters in feature names to make them compatible with LightGBM."""
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]
    return df

# Apply this to both train and test datasets **before training**
train_df = clean_feature_names(train_df)
test_df = clean_feature_names(test_df)

print("✅ Feature names cleaned for LightGBM compatibility!")

# ----------------- Step 6: Sort Dates & Compute Differences -----------------
date_cols = ["date0", "date1", "date2", "date3", "date4"]
status_cols = ["change_status_date0", "change_status_date1", "change_status_date2", "change_status_date3", "change_status_date4"]

def sort_dates_and_compute_differences(df):
    """Sorts dates in ascending order, reorders status changes accordingly, and computes date differences."""
    date_cols = ["date0", "date1", "date2", "date3", "date4"]
    # Convert dates to datetime format
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')

    # Print date columns BEFORE sorting
    print("\n🔹 Before Sorting Dates & Status:")
    print(df[date_cols + status_cols].head())

    # Sort dates and status labels per row
    def sort_dates(row):
        """Sorts date columns and reorders corresponding status columns."""
        valid_dates = row[date_cols].dropna().values  # Remove NaN values
        valid_statuses = row[status_cols].dropna().values  # Remove NaN values

        if len(valid_dates) > 0:
            sorted_indices = np.argsort(valid_dates)  # Get sorted index positions
            sorted_dates = valid_dates[sorted_indices].tolist()  # Apply sorted order
            sorted_statuses = valid_statuses[sorted_indices].tolist()  # Apply to statuses
        else:
            sorted_dates = [pd.NaT] * len(date_cols)  # Fill missing dates
            sorted_statuses = [np.nan] * len(status_cols)  # Fill missing statuses

        # Ensure the output always matches the number of columns
        sorted_dates += [pd.NaT] * (len(date_cols) - len(sorted_dates))  # Pad missing values
        sorted_statuses += [np.nan] * (len(status_cols) - len(sorted_statuses))  # Pad missing values

        return pd.Series(sorted_dates + sorted_statuses, index=date_cols + status_cols) 

    df[date_cols + status_cols] = df.apply(sort_dates, axis=1)

    # Print date columns AFTER sorting
    print("\n🔹 After Sorting Dates & Status:")
    print(df[date_cols + status_cols].head())

    # Create new columns for the difference in days between consecutive dates
    for i in range(len(date_cols) - 1):
        col1, col2 = date_cols[i], date_cols[i + 1]
        diff_col = f"{col2}_diff_days"

        # Compute differences, handling NaN values
        df[diff_col] = (df[col2].fillna(df[col2].median()) - df[col1].fillna(df[col1].median())).dt.days

        # Fill any remaining NaNs using the median
        df[diff_col].fillna(df[diff_col].median(), inplace=True)

    # Print computed date differences
    print("\n🔹 Date Differences Computed:")
    print(df[[f"{date_cols[i+1]}_diff_days" for i in range(len(date_cols) - 1)]].head())
    
    # Número de mudanças registradas
    df["change_frequency"] = df[[f"{date_cols[i+1]}_diff_days" for i in range(len(date_cols) - 1)]].count(axis=1)

    # Velocidade média de mudança
    df["change_speed"] = df[[f"{date_cols[i+1]}_diff_days" for i in range(len(date_cols) - 1)]].mean(axis=1)

    # Total de dias entre a primeira e a última mudança
    col_first, col_last = date_cols[0], date_cols[-1]
    df["total_days"] = (df[col_last].fillna(df[col_last].median()) - df[col_first].fillna(df[col_first].median())).dt.days
    df["total_days"] = df["total_days"].fillna(df["total_days"].median())

    # Taxa de mudança ao longo do tempo
    df["change_rate"] = df["change_frequency"] / (df["total_days"] + 1e-9)

    # Aceleração da mudança (variação da velocidade das mudanças)
    df["change_acceleration"] = df["change_speed"].diff().fillna(0)

    # Taxa acumulada de mudanças ao longo do tempo
    df["cumulative_change_rate"] = df["change_rate"].cumsum()

    # Identificação de períodos críticos de mudanças (2015-2018)
    df["high_change_2015_2018"] = (
        ((df[col_first].dt.year >= 2015) & (df[col_first].dt.year <= 2018)) |
        ((df[col_last].dt.year >= 2015) & (df[col_last].dt.year <= 2018))
    ).astype(int)

    # Extrair o mês das datas para analisar sazonalidade
    for i, col in enumerate(date_cols):
        df[f'month_date{i}'] = df[col].dt.month
        df[f'month_date{i}'] = df[f'month_date{i}'].fillna(df[f'month_date{i}'].median())

sort_dates_and_compute_differences(train_df)
sort_dates_and_compute_differences(test_df)
print("-------------THE END OF NEW FEATURES--------------")

# ----------------- Step 6.1: Date Handling -----------------
for col in date_cols:
    train_df[col] = pd.to_datetime(train_df[col], format='%d-%m-%Y', errors='coerce')
    test_df[col] = pd.to_datetime(test_df[col], format='%d-%m-%Y', errors='coerce')

    # Extract day, month, and year components
    for unit in ["day", "month", "year"]:
        train_df[f"{col}_{unit}"] = getattr(train_df[col].dt, unit)
        test_df[f"{col}_{unit}"] = getattr(test_df[col].dt, unit)

        # Fill NaN values with the **median**
        train_df[f"{col}_{unit}"].fillna(train_df[f"{col}_{unit}"].median(), inplace=True)
        test_df[f"{col}_{unit}"].fillna(test_df[f"{col}_{unit}"].median(), inplace=True)

# Drop original date columns after extracting necessary features
train_df.drop(columns=date_cols, inplace=True)
test_df.drop(columns=date_cols, inplace=True)

# Verify changes
print("Missing values in date-related columns (Train):")
print(train_df[[f"{col}_{unit}" for col in date_cols for unit in ["day", "month", "year"]]].isna().sum())

print("Missing values in date-related columns (Test):")
print(test_df[[f"{col}_{unit}" for col in date_cols for unit in ["day", "month", "year"]]].isna().sum())

# ----------------- Step 6.2: Create Status Sequence Column -----------------
def create_status_sequence(df):
    """Generates a new column combining the five sorted change_status columns as a string."""
    df["status_sequence"] = df[status_cols].astype(str).agg(", ".join, axis=1)
    return df

# Apply to both train and test sets
train_df = create_status_sequence(train_df)
test_df = create_status_sequence(test_df)

# Print sample output
print("\n🔹 Sample of Status Sequence Column:")
print(train_df[["status_sequence"]].head())

# ----------------- Step 6.3: One-Hot Encoding (OHE) for Status Sequence -----------------
"""ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit on train, but transform in parallel
ohe.fit(train_df[["status_sequence"]])
print(ohe.get_feature_names_out(["status_sequence"]))
# Define a function to parallelize transformation
def parallel_transform(df, encoder, n_jobs=-1):
    #Splits DataFrame and applies OHE in parallel.
    num_partitions = min(len(df), 4)  # Set partitions (at most 4)
    df_split = np.array_split(df, num_partitions)  # Split DataFrame into chunks

    # Run transformation in parallel
    transformed_splits = Parallel(n_jobs=n_jobs)(
        delayed(encoder.transform)(split) for split in df_split
    )
    
    # Combine results
    return np.vstack(transformed_splits)

# Apply parallel transformation
train_ohe = parallel_transform(train_df[["status_sequence"]], ohe)
test_ohe = parallel_transform(test_df[["status_sequence"]], ohe)

# Convert to DataFrame
train_ohe_df = pd.DataFrame(train_ohe, columns=ohe.get_feature_names_out(["status_sequence"]), index=train_df.index)
test_ohe_df = pd.DataFrame(test_ohe, columns=ohe.get_feature_names_out(["status_sequence"]), index=test_df.index)

# Concatenate the OneHotEncoded Data with original DataFrame
train_df = pd.concat([train_df, train_ohe_df], axis=1)
test_df = pd.concat([test_df, test_ohe_df], axis=1)

# Drop the original status_sequence column after encoding
train_df.drop(columns=["status_sequence"], inplace=True)
test_df.drop(columns=["status_sequence"], inplace=True)

# Print sample output
print("\n✅ One-Hot Encoding Applied to Status Sequence!")

# ----------------- Step 6.4: Clean Feature Names -----------------
def clean_feature_names(df):
    #Replace special characters in feature names to make them compatible with LightGBM.
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]
    return df

# Apply cleaning
train_df = clean_feature_names(train_df)
test_df = clean_feature_names(test_df)

print("✅ Feature names cleaned for LightGBM compatibility!")
"""
# Combine the 5 status columns into a single string sequence
train_df["status_sequence"] = train_df[status_cols].astype(str).agg(", ".join, axis=1)
test_df["status_sequence"] = test_df[status_cols].astype(str).agg(", ".join, axis=1)

# Create a unique integer mapping for each unique sequence
unique_sequences = pd.concat([train_df["status_sequence"], test_df["status_sequence"]]).unique()
sequence_mapping = {seq: idx for idx, seq in enumerate(sorted(unique_sequences))}

# Apply mapping to train and test data
train_df["status_sequence_encoded"] = train_df["status_sequence"].map(sequence_mapping)
test_df["status_sequence_encoded"] = test_df["status_sequence"].map(sequence_mapping)

# Drop the original status_sequence column since we have an integer representation
train_df.drop(columns=["status_sequence"], inplace=True)
test_df.drop(columns=["status_sequence"], inplace=True)

# Print mapping summary
print(f"✅ Encoded {len(sequence_mapping)} unique status sequences into integers.")
print("Sample Encoding Mapping:")
print(dict(list(sequence_mapping.items())[:10]))  # Print first 10 mappings

# Verify encoded values
print(train_df["status_sequence_encoded"].head())
print(test_df["status_sequence_encoded"].head())

# ----------------- Step 7: Process change_status_date -----------------
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
# ----------------- Step 9: Handle Missing & Infinite Values -----------------
train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)

train_df.replace([np.inf, -np.inf], 0, inplace=True)
test_df.replace([np.inf, -np.inf], 0, inplace=True)

# ----------------- Step 10: Standardize Features (Excluding Dates & Categorical Data) -----------------
# "_day" in col or "_month" in col or "_year" in col or "change_status_date" in col
exclude_cols = [col for col in train_df.columns if "urban_" in col or "geo_" in col]
standardized_cols = [col for col in train_df.columns if col not in exclude_cols + ["change_type"]]

scaler = StandardScaler()
train_x_scaled_df = pd.DataFrame(scaler.fit_transform(train_df[standardized_cols]), columns=standardized_cols, index=train_df.index)
test_x_scaled_df = pd.DataFrame(scaler.transform(test_df[standardized_cols]), columns=standardized_cols, index=test_df.index)

train_x_scaled = pd.concat([train_x_scaled_df, train_df[exclude_cols + ["change_type"]]], axis=1)
test_x_scaled = pd.concat([test_x_scaled_df, test_df[exclude_cols]], axis=1)

# Convert train_x_scaled to a DataFrame (ensuring correct columns)
train_x_df = pd.DataFrame(train_x_scaled, columns=standardized_cols + exclude_cols + ["change_type"], index=train_df.index)

# Save treated dataset BEFORE feature selection
train_x_df.to_csv("train_x_treated.csv", index=True, index_label='index')

# Check if file is correctly saved
print(f"Saved processed dataset with shape: {train_x_df.shape}")

# ----------------- Step 11: Feature Selection -----------------
def select_features(train_x_scaled, test_x_scaled, k):
    # Define all feature columns excluding 'change_type' (target variable)
    feature_cols = [col for col in train_x_scaled.columns if col != "change_type"]

    # Apply SelectKBest across ALL features
    selector = SelectKBest(score_func=f_classif, k=k)
    train_x_selected_arr = selector.fit_transform(train_x_scaled[feature_cols], train_y)
    test_x_selected_arr = selector.transform(test_x_scaled[feature_cols])

    # Get selected feature names (from all features, not just standardized ones)
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    print("\nSelected Features:", selected_features)
    return train_x_selected_arr,test_x_selected_arr

#train_x_selected,test_x_selected = select_features(train_x_scaled, test_x_scaled, 50)
train_x_selected = train_x_scaled
test_x_selected = test_x_scaled

# Save selected feature datasets to CSV
train_x_selected.to_csv("train_x_selected.csv", index=True, index_label='Id')
test_x_selected.to_csv("test_x_selected.csv", index=True, index_label='Id')

print("\n💾 Saved selected training data to 'train_x_selected.csv'")
print("💾 Saved selected test data to 'test_x_selected.csv'")

train_x_selected = train_x_selected.drop(columns=["change_type"])

# # ----------------- Step 12: Model Training (Gradient Boosting) -----------------

# ## Train a simple OnveVsRestClassifier using featurized data
# neigh = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
# neigh.fit(train_x_selected, train_y)

# pred_y = neigh.predict(test_x_selected)
# print(pred_y.shape)

# # ----------------- Step 13: Predictions & Evaluation -----------------


# train_preds = neigh.predict(train_x_selected)
# train_accuracy = accuracy_score(train_y, train_preds)
# train_f1_macro = f1_score(train_y, train_preds, average='macro')
# train_f1_weighted = f1_score(train_y, train_preds, average='weighted')

# print(f"\nTraining Accuracy: {train_accuracy:.4f}")
# print(f"F1 (Macro): {train_f1_macro:.4f}")
# print(f"F1 (Weighted): {train_f1_weighted:.4f}")

# # ----------------- Step 14: Save Results -----------------
# pred_df = pd.DataFrame(pred_y, columns=['change_type'])
# pred_df.to_csv("knn_sample_submission.csv", index=True, index_label='Id')
