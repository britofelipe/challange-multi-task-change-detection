import time
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KDTree
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

print("🔍 Starting script execution...\n")

change_type_map = {'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3, 'Industrial': 4,
       'Mega Projects': 5}
# Load datasets
print("📂 Loading datasets...")
start = time.time()
train_df = gpd.read_file('train.geojson')
test_df = gpd.read_file('test.geojson')
print(f"✅ Datasets loaded in {time.time()-start:.2f}s")
print(f"📏 Training data shape: {train_df.shape}, Test data shape: {test_df.shape}\n")

# Feature Engineering Functions
def compute_geometry_features(df):
    print("📐 Computing geometry features...")
    df = df.to_crs(epsg=3857)  # Convert to a projected CRS for accurate area/length calculations
    
    # Check for invalid geometries
    invalid_geoms = df[~df.geometry.is_valid]
    if not invalid_geoms.empty:
        print(f"⚠️ Found {len(invalid_geoms)} invalid geometries. Attempting to fix...")
        df.geometry = df.geometry.buffer(0)  # Attempt to fix invalid geometries
    
    start = time.time()
    
    # Compute basic geometric features
    df['area'] = df.geometry.area
    df['perimeter'] = df.geometry.length
    df['compactness'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
    df['elongation'] = df['perimeter'] / np.sqrt(df['area'])
    
    # Compute convex hull with error handling
    df['convex_hull'] = df.geometry.apply(lambda geom: geom.convex_hull if geom.is_valid else None)
    
    # Handle cases where convex hull calculation failed
    if df['convex_hull'].isnull().any():
        print(f"⚠️ Convex hull calculation failed for {df['convex_hull'].isnull().sum()} geometries. Filling with original geometry.")
        df['convex_hull'] = df['convex_hull'].fillna(df.geometry)
    
    # Compute additional features based on convex hull
    df['convex_hull_area'] = df['convex_hull'].area
    df['convexity'] = df['perimeter'] / df['convex_hull'].length
    df['solidity'] = df['area'] / df['convex_hull_area']
    
    # Log transformations and derived features
    df['convexity_log'] = np.log1p(df['convexity'])
    df['solidity_ratio'] = df['solidity'] / df['convexity']
    df['centroid'] = df.geometry.centroid
    df['centroid_x'] = df['centroid'].x
    df['centroid_y'] = df['centroid'].y
    df['elongation_log'] = np.log1p(df['elongation'])
    df['compactness_inv'] = 1 - df['compactness']
    
    print(f"✅ Geometry features added in {time.time()-start:.2f}s")
    print(f"📊 New geometry features: {list(df.columns[-12:])}\n")
    return df

def compute_spatial_features(train_df, test_df):
    print("🌍 Computing spatial features...")
    start = time.time()
    
    # Check for NaN centroids in training data
    print("🔍 Checking for NaN centroids in training data...")
    nan_centroids_train = train_df[['centroid_x', 'centroid_y']].isnull().any(axis=1)
    if nan_centroids_train.any():
        print(f"⚠️ Found {nan_centroids_train.sum()} NaN centroids in training data. Removing these rows.")
        train_df = train_df.loc[~nan_centroids_train].copy()  # Use .loc and .copy() to avoid warnings
        print(f"📏 Training data shape after removing NaN centroids: {train_df.shape}")
    else:
        print("✅ No NaN centroids found in training data.")
    
    # Check for NaN centroids in test data
    print("🔍 Checking for NaN centroids in test data...")
    nan_centroids_test = test_df[['centroid_x', 'centroid_y']].isnull().any(axis=1)
    if nan_centroids_test.any():
        print(f"⚠️ Found {nan_centroids_test.sum()} NaN centroids in test data. Removing these rows.")
        test_df = test_df.loc[~nan_centroids_test].copy()  # Use .loc and .copy() to avoid warnings
        print(f"📏 Test data shape after removing NaN centroids: {test_df.shape}")
    else:
        print("✅ No NaN centroids found in test data.")
    
    # Training spatial features
    print("🌐 Building KDTree for training data...")
    coords_train = train_df[['centroid_x', 'centroid_y']].values
    
    # Ensure no NaN values remain
    if np.isnan(coords_train).any():
        raise ValueError("Training coordinates still contain NaN values after cleaning.")
    
    tree_train = KDTree(coords_train)
    print("✅ KDTree built successfully.")
    
    # Training data features
    print("📏 Calculating nearest neighbor distances for training data...")
    dist_train, _ = tree_train.query(coords_train, k=2)
    train_df = train_df.copy()  # Ensure we're working with a copy
    train_df.loc[:, 'nearest_neighbor_dist'] = dist_train[:, 1]  # Use .loc for assignment
    print("✅ Nearest neighbor distances calculated for training data.")
    
    print("📏 Calculating average distance to neighbors for training data...")
    dist_avg_train, _ = tree_train.query(coords_train, k=5)
    train_df.loc[:, 'avg_distance_to_neighbors'] = dist_avg_train.mean(axis=1)  # Use .loc for assignment
    print("✅ Average distance to neighbors calculated for training data.")
    
    # KDE for training data
    # print("📊 Calculating centroid density for training data...")
    # xy_train = np.vstack([train_df['centroid_x'], train_df['centroid_y']]).T
    # kde = gaussian_kde(xy_train.T)
    # train_df.loc[:, 'centroid_density'] = kde(xy_train.T)  # Use .loc for assignment
    # print("✅ Centroid density calculated for training data.")
    
    # Test data features
    print("🌐 Processing test data features...")
    coords_test = test_df[['centroid_x', 'centroid_y']].values
    
    # Ensure no NaN values in test coordinates
    if np.isnan(coords_test).any():
        raise ValueError("Test coordinates contain NaN values.")
    
    print("📏 Calculating nearest neighbor distances for test data...")
    dist_test, _ = tree_train.query(coords_test, k=1)
    test_df = test_df.copy()  # Ensure we're working with a copy
    test_df.loc[:, 'nearest_neighbor_dist'] = dist_test[:, 0]  # Use .loc for assignment
    print("✅ Nearest neighbor distances calculated for test data.")
    
    print("📏 Calculating average distance to neighbors for test data...")
    dist_avg_test, _ = tree_train.query(coords_test, k=5)
    test_df.loc[:, 'avg_distance_to_neighbors'] = dist_avg_test.mean(axis=1)  # Use .loc for assignment
    print("✅ Average distance to neighbors calculated for test data.")
    
    # print("📊 Calculating centroid density for test data...")
    # xy_test = np.vstack([test_df['centroid_x'], test_df['centroid_y']]).T
    # test_df.loc[:, 'centroid_density'] = kde(xy_test.T)  # Use .loc for assignment
    # print("✅ Centroid density calculated for test data.")
    
    print(f"✅ Spatial features computed in {time.time()-start:.2f}s")
    # print(f"📌 Training spatial features summary:\n{train_df[['nearest_neighbor_dist', 'avg_distance_to_neighbors', 'centroid_density']].describe()}\n")
    return train_df, test_df

def compute_date_features(df):
    print("📅 Processing date features...")
    start = time.time()
    
    # Define date columns
    date_cols = ['date1', 'date2', 'date3', 'date4']
    
    # Convert date columns to datetime with the correct format
    print("🔍 Parsing date columns...")
    for col in date_cols:
        print(f"   Parsing {col}...")
        df[col] = pd.to_datetime(df[col], format="%d-%m-%Y")  # Specify the correct format
    
    # Calculate delta days between consecutive dates
    print("📏 Calculating delta days between consecutive dates...")
    for i in range(1, len(date_cols)):  # Only loop up to the last available date
        df[f'delta_days_{i}'] = (df[f'date{i+1}'] - df[f'date{i}']).dt.days
        print(f"   Calculated delta_days_{i}.")
    
    # Calculate total days between the first and last date
    print("📏 Calculating total days between first and last date...")
    df['total_days'] = (df[date_cols[-1]] - df[date_cols[0]]).dt.days  # Use the last and first date columns
    print("✅ Total days calculated.")
    
    # Print statistics
    print(f"✅ Date features processed in {time.time()-start:.2f}s")
    print(f"⏳ Date delta statistics:\n{df[[f'delta_days_{i}' for i in range(1, len(date_cols))] + ['total_days']].describe()}\n")
    
    return df

# Apply feature engineering
print("🛠️ Starting feature engineering pipeline...")
train_df = compute_geometry_features(train_df)
test_df = compute_geometry_features(test_df)

train_df, test_df = compute_spatial_features(train_df, test_df)
train_df = compute_date_features(train_df)
test_df = compute_date_features(test_df)

# Process categorical features
print("🏙️ Processing categorical features...")
start = time.time()

# Process urban_type (single category)
mlb_urban = MultiLabelBinarizer()
urban_train = mlb_urban.fit_transform(train_df['urban_type'].apply(lambda x: [x]))  # Wrap in a list for MultiLabelBinarizer
urban_train = pd.DataFrame(urban_train, columns=mlb_urban.classes_, index=train_df.index)
urban_test = pd.DataFrame(mlb_urban.transform(test_df['urban_type'].apply(lambda x: [x])), 
                        columns=mlb_urban.classes_, index=test_df.index)

# Process geography_type (multi-label categories)
mlb_geo = MultiLabelBinarizer()
geo_train = mlb_geo.fit_transform(train_df['geography_type'].str.split(','))  # Split by comma for multi-label
geo_train = pd.DataFrame(geo_train, columns=mlb_geo.classes_, index=train_df.index)
geo_test = pd.DataFrame(mlb_geo.transform(test_df['geography_type'].str.split(',')), 
                      columns=mlb_geo.classes_, index=test_df.index)

print(f"✅ Categorical features processed in {time.time()-start:.2f}s")
print(f"🏘️ Urban types: {mlb_urban.classes_}")
print(f"🌳 Geography types: {mlb_geo.classes_}\n")

# After feature engineering, check the columns in train_df
print("🔍 Checking columns in train_df after feature engineering...")
print("Columns in train_df:", train_df.columns.tolist())

# Define numerical_features based on available columns
numerical_features = [
    'area', 'perimeter', 'compactness', 'elongation', 'convex_hull_area',
    'convexity', 'solidity', 'convexity_log', 'solidity_ratio',
    'nearest_neighbor_dist', 'avg_distance_to_neighbors',
    'elongation_log', 'compactness_inv', 'delta_days_1', 'delta_days_2',
    'delta_days_3', 'delta_days_4', 'total_days'
]

# Filter numerical_features to include only columns that exist in train_df
numerical_features = [col for col in numerical_features if col in train_df.columns]
print("✅ Final numerical features:", numerical_features)

# Combine features
print("🔗 Combining features...")
X_train = pd.concat([train_df[numerical_features], urban_train, geo_train], axis=1)
X_test = pd.concat([test_df[numerical_features], urban_test, geo_test], axis=1)
print(f"✅ Final feature set combined")
print(f"📈 Training features shape: {X_train.shape}, Test features shape: {X_test.shape}")
print(f"🔣 Feature breakdown: {len(numerical_features)} numerical, {urban_train.shape[1]} urban, {geo_train.shape[1]} geographical\n")
# Handle missing values
print("🔄 Handling missing values...")
print(f"⚠️ Missing values before handling - Train: {X_train.isna().sum().sum()}, Test: {X_test.isna().sum().sum()}")
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
print(f"✅ Missing values after handling - Train: {X_train.isna().sum().sum()}, Test: {X_test.isna().sum().sum()}\n")

# Scale numerical features
print("⚖️ Scaling numerical features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_features])
X_train_scaled = pd.DataFrame(X_train_scaled, columns=numerical_features, index=X_train.index)
X_train_final = pd.concat([X_train_scaled, X_train.drop(columns=numerical_features)], axis=1)

X_test_scaled = scaler.transform(X_test[numerical_features])
X_test_scaled = pd.DataFrame(X_test_scaled, columns=numerical_features, index=X_test.index)
X_test_final = pd.concat([X_test_scaled, X_test.drop(columns=numerical_features)], axis=1)
print("✅ Features scaled successfully")
print(f"📊 Sample scaled training data:\n{X_train_scaled.iloc[:3, :5]}\n")

# Encode target variable
print("🎯 Encoding target variable...")
label_encoder = LabelEncoder()
train_y = label_encoder.fit_transform(train_df['change_type'])
class_names = label_encoder.classes_
print(f"✅ Target classes encoded: {dict(zip(class_names, label_encoder.transform(class_names)))}")
print(f"📊 Class distribution:\n{pd.Series(train_y).value_counts().rename(index=dict(zip(range(len(class_names)), class_names)))}\n")

# Random Forest Training
print("🌳 Starting Random Forest training...")
start_rf = time.time()
rf = RandomForestClassifier(n_estimators=200, 
                          class_weight='balanced',
                          random_state=42,
                          n_jobs=-1)

# Cross-validation
print("🔍 Performing cross-validation for Random Forest...")
cv_scores = cross_val_score(rf, X_train_final, train_y, 
                           cv=3, scoring='f1_macro', n_jobs=-1)
print(f"✅ RF Cross-val F1 scores: {cv_scores}")
print(f"📈 Mean F1: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")

# Full training
rf.fit(X_train_final, train_y)
print(f"✅ Random Forest trained in {time.time()-start_rf:.2f}s")
print(f"📝 RF Feature importances (top 10):")
importances = pd.Series(rf.feature_importances_, index=X_train_final.columns)
print(importances.sort_values(ascending=False).head(10).to_string() + "\n")

# XGBoost Training
print("🚀 Starting XGBoost training...")
start_xgb = time.time()
xgb = XGBClassifier(n_estimators=300,
                   learning_rate=0.1,
                   max_depth=5,
                   subsample=0.8,
                   colsample_bytree=0.8,
                   objective='multi:softmax',
                   num_class=len(class_names),
                   random_state=42,
                   n_jobs=-1)

# Cross-validation
print("🔍 Performing cross-validation for XGBoost...")
xgb_cv_scores = cross_val_score(xgb, X_train_final, train_y, 
                              cv=3, scoring='f1_macro', n_jobs=-1)
print(f"✅ XGB Cross-val F1 scores: {xgb_cv_scores}")
print(f"📈 Mean F1: {xgb_cv_scores.mean():.4f}, Std: {xgb_cv_scores.std():.4f}")

# Full training
xgb.fit(X_train_final, train_y)
print(f"✅ XGBoost trained in {time.time()-start_xgb:.2f}s")
print(f"📝 XGB Feature importances (top 10):")
xgb_importances = pd.Series(xgb.feature_importances_, index=X_train_final.columns)
print(xgb_importances.sort_values(ascending=False).head(10).to_string() + "\n")

# Model comparison
print("🏆 Model comparison:")
print(f"Random Forest | Mean CV F1: {cv_scores.mean():.4f}")
print(f"XGBoost       | Mean CV F1: {xgb_cv_scores.mean():.4f}")

# Select best model
if xgb_cv_scores.mean() > cv_scores.mean():
    print("🎖️ Selected XGBoost as final model")
    final_model = xgb
else:
    print("🎖️ Selected Random Forest as final model")
    final_model = rf

# Predict and prepare submission
print("\n🔮 Making predictions...")
start_pred = time.time()
pred_y = final_model.predict(X_test_final)
pred_labels = label_encoder.inverse_transform(pred_y)
print(f"✅ Predictions completed in {time.time()-start_pred:.2f}s")

print("\n📊 Prediction distribution:")
print(pd.Series(pred_labels).value_counts().to_string() + "\n")

# Save results
print("💾 Saving submission file...")
pred_df = pd.DataFrame(pred_labels, columns=['change_type'], index=test_df.index)
pred_df.to_csv("submission.csv", index=True, index_label='Id')
print("✅ Submission file saved as submission.csv")

print("\n🎉 Script execution completed successfully!")