# General
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Preprocessing
from sklearn.impute import SimpleImputer

# Dimensionality Reduction and Scaling
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# Learning algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

change_type_map = {'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3, 'Industrial': 4,
       'Mega Projects': 5}

## Read csvs
print("Reading data...")
# train_df = gpd.read_file('train.geojson')
# test_df = gpd.read_file('test.geojson')
train_df = gpd.read_file('train.geojson').head(100)
test_df = gpd.read_file('test.geojson').head(100)
print("Data read successfully")

# FEATURE ENGINEERING
print("Feature Engineering...")

# 1. Geometry-based features
#    For example, compute area, perimeter, and compactness of each polygon.
#    Compactness is defined as the area divided by the perimeter squared.
train_df['area'] = train_df.geometry.area
train_df['perimeter'] = train_df.geometry.length
train_df['compactness'] = train_df['area'] / (train_df['perimeter']**2 + 1e-6)

test_df['area'] = test_df.geometry.area
test_df['perimeter'] = test_df.geometry.length
test_df['compactness'] = test_df['area'] / (test_df['perimeter']**2 + 1e-6)

# 2. Date-based features
#    For example, compute the number of days between consecutive dates.
#    This can be useful since the dates represent a sequence of events.
date_cols = ['date1', 'date2', 'date3', 'date4', 'date5']
if all(col in train_df.columns for col in date_cols):
    # Convert to datetime
    for col in date_cols:
        train_df[col] = pd.to_datetime(train_df[col])
        test_df[col] = pd.to_datetime(test_df[col])
    # Compute the number of days between consecutive dates
    for i in range(len(date_cols) - 1):
        diff_col = f'days_diff_{i+1}_{i+2}'
        train_df[diff_col] = (train_df[date_cols[i+1]] - train_df[date_cols[i]]).dt.days
        test_df[diff_col] = (test_df[date_cols[i+1]] - test_df[date_cols[i]]).dt.days
else:
    print("Date columns not found, skipping date-based features.")

# 3. Categorical features
#    For example, one-hot encode 'urban_type' and 'geography_type' if they exist.
categorical_features = []
if 'urban_type' in train_df.columns:
    categorical_features.append('urban_type')
if 'geography_type' in train_df.columns:
    categorical_features.append('geography_type')

if categorical_features:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    train_cat = encoder.fit_transform(train_df[categorical_features])
    test_cat = encoder.transform(test_df[categorical_features])
else:
    # No categorical features found; create empty arrays with the correct number of rows.
    train_cat = np.empty((len(train_df), 0))
    test_cat = np.empty((len(test_df), 0))

# 4. Combine numerical features
#    Include geometry-based features and date differences (if any)
num_features = ['area', 'perimeter', 'compactness']
# Add date difference columns if they were created
date_diff_cols = [col for col in train_df.columns if col.startswith('days_diff_')]
num_features.extend(date_diff_cols)

# Impute missing values if any exist (for example, if some date differences are missing)
imputer = SimpleImputer(strategy='median')
train_num = imputer.fit_transform(train_df[num_features])
test_num = imputer.transform(test_df[num_features])

# 5. Combine all features into one feature matrix
train_x = np.hstack([train_num, train_cat])
test_x = np.hstack([test_num, test_cat])

# Target labels (using the change_type_map)
train_y = train_df['change_type'].apply(lambda x: change_type_map[x]).values

print("Feature engineering completed. Feature shapes:")
print("Train features:", train_x.shape)
print("Test features:", test_x.shape)


## Filtering column "mail_type"
train_x = np.asarray(train_df[['geometry']].area)
train_x = train_x.reshape(-1, 1)
train_y = train_df['change_type'].apply(lambda x: change_type_map[x])

test_x = np.asarray(test_df[['geometry']].area)
test_x = test_x.reshape(-1, 1)

print (train_x.shape, train_y.shape, test_x.shape)

# TODO

# DIMENSIONALITY REDUCTION

# Normalizing data
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# Applying PCA with 95% variance
pca = PCA(n_components=0.95)
train_x_pca = pca.fit_transform(train_x_scaled)
test_x_pca = pca.transform(test_x_scaled)

print(f"Reduction from {train_x_scaled.shape[1]} to {train_x_pca.shape[1]} dimensions due to PCA")

# Separate training data into training and validation
X_train, X_val, y_train, y_val = train_test_split(train_x_pca, train_y, test_size=0.2, random_state=42, stratify=train_y)


# TRAINING
# Different classifiers
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "SVM": SVC(kernel="rbf", probability=True),
    "LogisticRegression": LogisticRegression(max_iter=500, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", n_jobs=-1)
}

best_model = None
best_score = 0
model_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    mean_score = np.mean(scores)
    model_results[name] = mean_score
    print(f"{name} - Mean F1 Score: {mean_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_model = model

print(f"\Best model: {best_model.__class__.__name__} with F1-score: {best_score:.4f}")
best_model.fit(X_train, y_train)
pred_y = best_model.predict(test_x_pca)

## Save results to submission file
pred_df = pd.DataFrame(pred_y, columns=['change_type'])
pred_df.to_csv("submission.csv", index=True, index_label='Id')