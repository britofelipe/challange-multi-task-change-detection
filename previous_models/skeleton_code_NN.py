# Import necessary libraries
import geopandas as gpd  # For handling geospatial data (GeoJSON)
import pandas as pd  # For data manipulation and handling tabular data
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting results
from sklearn.metrics import accuracy_score, f1_score  # Evaluation metrics
from sklearn.preprocessing import StandardScaler  # Feature scaling
from sklearn.decomposition import PCA  # Dimensionality reduction
from category_encoders import TargetEncoder  # Encoding categorical data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight

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

# ----------------- Step 4: Feature Engineering -----------------
# Compute geometric features
train_df["area"] = train_df["geometry"].area
train_df["centroid_x"] = train_df["geometry"].centroid.x
train_df["centroid_y"] = train_df["geometry"].centroid.y
train_df["perimeter"] = train_df["geometry"].length
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

# ----------------- Step 5: Handle Categorical Features -----------------
cat_features = ["urban_type", "geography_type"]
train_df[cat_features] = train_df[cat_features].astype(str)
test_df[cat_features] = test_df[cat_features].astype(str)

# Convert labels to numerical values
train_y = train_df['change_type'].apply(lambda x: change_type_map[x])

# Define target encoder
encoder = TargetEncoder()
train_df[cat_features] = encoder.fit_transform(train_df[cat_features], train_y)
test_df[cat_features] = encoder.transform(test_df[cat_features])

# ----------------- Step 6: Handle Date Features -----------------
date_features = ["date0", "date1", "date2", "date3", "date4"]
for col in date_features:
    train_df[col] = pd.to_datetime(train_df[col], errors='coerce').astype('int64') // 10**9
    test_df[col] = pd.to_datetime(test_df[col], errors='coerce').astype('int64') // 10**9

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
train_x.fillna(train_x.median(), inplace=True)
test_x.fillna(test_x.median(), inplace=True)

# Replace infinity values
train_x.replace([np.inf, -np.inf], np.nan, inplace=True)
test_x.replace([np.inf, -np.inf], np.nan, inplace=True)

train_x.fillna(train_x.median(), inplace=True)
test_x.fillna(test_x.median(), inplace=True)

# Standardize features
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

selector = SelectKBest(score_func=f_classif, k=14)  # Keep top 50 best features
train_x = selector.fit_transform(train_x, train_y.argmax(axis=1))
test_x= selector.transform(test_x)

# Convert target to categorical
train_y = to_categorical(train_y, num_classes=6)

# ----------------- Step 8: Train Neural Network -----------------
model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.0005), input_shape=(train_x.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu', kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu', kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(6, activation='softmax')  # Multi-class classification (6 classes)
])

# Define F1 Score Metric for Keras
def f1_m(y_true, y_pred):
    y_pred = K.round(y_pred)  # Convert softmax output to 0s and 1s
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    precision = tp / (K.sum(K.cast(y_pred, 'float'), axis=0) + K.epsilon())
    recall = tp / (K.sum(K.cast(y_true, 'float'), axis=0) + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.mean(f1)

# Compile model
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.95
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(
    optimizer=optimizer, 
    loss='categorical_crossentropy', 
    metrics=['accuracy', f1_m])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

class_weights = compute_class_weight('balanced', classes=np.unique(train_y.argmax(axis=1)), y=train_y.argmax(axis=1))
class_weight_dict = dict(enumerate(class_weights))

# Train the model
model.fit(train_x, train_y,
        epochs=50, batch_size=32, 
        validation_split=0.2, 
        callbacks=[early_stopping], 
        class_weight=class_weight_dict)

# ----------------- Step 9: Model Prediction -----------------
pred_y = np.argmax(model.predict(train_x), axis=1)

# ----------------- Step 10: Save Results -----------------
pred_df = pd.DataFrame(pred_y, columns=['change_type'])
pred_df.to_csv("neural_network_submission.csv", index=True, index_label='Id')

# Convert one-hot encoded train_y back to class labels
train_y_labels = np.argmax(train_y, axis=1)  # Convert from one-hot to class labels
pred_y = np.argmax(model.predict(train_x), axis=1)  # Convert softmax output to class labels
# Compute F1-score
f1_macro = f1_score(train_y_labels, pred_y, average='macro')
f1_weighted = f1_score(train_y_labels, pred_y, average='weighted')

print(f"Mean F1-Score (Train - Macro): {f1_macro:.4f}")
print(f"Weighted Mean F1-Score (Train): {f1_weighted:.4f}")

# ----------------- Step 11: Plot Training Loss -----------------
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Neural Network Training Loss")
plt.legend()
plt.show()