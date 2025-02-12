import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import warnings
warnings.filterwarnings('ignore')

# ----------------- Step 1: Advanced Feature Engineering -----------------
def calculate_temporal_intervals(df, date_cols):
    for i in range(len(date_cols)-1):
        df[f'days_between_{i}_{i+1}'] = (df[date_cols[i+1]] - df[date_cols[i]]).dt.days
    return df

def extract_sequence_features(status_cols):
    # Implement status transition matrix and sequence encoding
    pass

# ----------------- Step 2: Spatial-Temporal Data Loader -----------------
class GeoDataLoader:
    def __init__(self, train_path, test_path):
        self.train = gpd.read_file(train_path)
        self.test = gpd.read_file(test_path)
        self._preprocess()
    
    def _preprocess(self):
        # CRS normalization and geometry validation
        self.train = self.train.to_crs(epsg=3857).dropna(subset=['geometry'])
        self.test = self.test.to_crs(epsg=3857).dropna(subset=['geometry'])
        
        # Advanced feature engineering
        for df in [self.train, self.test]:
            # Geometric features
            df['compactness'] = 4 * np.pi * df.geometry.area / (df.geometry.length**2)
            df['hull_ratio'] = df.geometry.area / df.geometry.convex_hull.area
            
            # Temporal features
            date_cols = [f'date{i}' for i in range(5)]
            df[date_cols] = df[date_cols].apply(pd.to_datetime)
            df = calculate_temporal_intervals(df, date_cols)
            
            # Spectral features for all dates
            for d in range(5):
                for band in ['red', 'green', 'blue']:
                    df[f'spectral_ratio_{d}_{band}'] = df[f'img_{band}_mean_date{d}'] / \
                                                      (df[f'img_{band}_std_date{d}'] + 1e-9)

# ----------------- Step 3: Hybrid Model Architecture -----------------
class ChampionModel:
    def __init__(self, n_classes, temporal_dims, spatial_dims):
        # Temporal branch (LSTM)
        temporal_input = Input(shape=(temporal_dims,))
        x = Dense(128, activation='swish')(temporal_input)
        x = BatchNormalization()(x)
        
        # Spatial branch
        spatial_input = Input(shape=(spatial_dims,))
        y = Dense(128, activation='swish')(spatial_input)
        y = BatchNormalization()(y)
        
        # Merge branches
        combined = concatenate([x, y])
        z = Dense(256, activation='swish')(combined)
        z = Dropout(0.3)(z)
        output = Dense(n_classes, activation='softmax')(z)
        
        self.model = Model(inputs=[temporal_input, spatial_input], outputs=output)
        self.model.compile(optimizer=Adam(0.001),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
    
    def fit(self, X_temp, X_spat, y, val_data, class_weights):
        early_stop = EarlyStopping(patience=10, restore_best_weights=True)
        self.model.fit([X_temp, X_spat], y,
                      validation_data=val_data,
                      epochs=100,
                      batch_size=512,
                      class_weight=class_weights,
                      callbacks=[early_stop],
                      verbose=1)

# ----------------- Step 4: Optimized Training Pipeline -----------------
def main():
    # Load and preprocess data
    loader = GeoDataLoader('train.geojson', 'test.geojson')
    
    # Generate class weights
    classes = loader.train['change_type'].unique()
    class_weights = compute_class_weight('balanced', classes=classes, y=loader.train['change_type'])
    class_weights = dict(enumerate(class_weights))
    
    # Feature selection and engineering
    temporal_features = [f'days_between_{i}_{i+1}' for i in range(4)] + \
                       [f'spectral_ratio_{d}_{b}' for d in range(5) for b in ['red','green','blue']]
    
    spatial_features = ['area', 'compactness', 'hull_ratio', 'centroid_x', 'centroid_y']
    
    # Ensemble models
    xgb = XGBClassifier(objective='multi:softmax', 
                       tree_method='gpu_hist',
                       use_label_encoder=False,
                       scale_pos_weight=class_weights)
    
    lgbm = LGBMClassifier(class_weight=class_weights,
                         device='gpu')
    
    # Neural model
    neural_model = ChampionModel(n_classes=6,
                                temporal_dims=len(temporal_features),
                                spatial_dims=len(spatial_features))
    
    # Stratified K-Fold ensemble
    skf = StratifiedKFold(n_splits=5)
    test_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(loader.train, loader.train['change_type'])):
        # Prepare data
        X_temp = loader.train[temporal_features].iloc[train_idx]
        X_spat = loader.train[spatial_features].iloc[train_idx]
        y_train = loader.train['change_type'].iloc[train_idx]
        
        # Train neural model
        neural_model.fit(X_temp.values, X_spat.values, y_train.values,
                        val_data=([X_temp.iloc[val_idx].values, X_spat.iloc[val_idx].values], 
                                  y_val.values),
                        class_weights=class_weights)
        
        # Train XGBoost/LGBM on full feature set
        xgb.fit(loader.train.iloc[train_idx], y_train)
        lgbm.fit(loader.train.iloc[train_idx], y_train)
        
        # Ensemble predictions
        neural_preds = neural_model.model.predict([loader.test[temporal_features], 
                                                 loader.test[spatial_features]])
        xgb_preds = xgb.predict_proba(loader.test)
        lgbm_preds = lgbm.predict_proba(loader.test)
        
        # Weighted ensemble
        final_preds = (0.4 * neural_preds + 0.3 * xgb_preds + 0.3 * lgbm_preds).argmax(axis=1)
        test_preds.append(final_preds)
    
    # Final submission (mode of fold predictions)
    submission = pd.DataFrame({'Id': loader.test.index,
                              'change_type': pd.Series(test_preds).mode()[0]})
    submission.to_csv('champion_submission.csv', index=False)

if __name__ == "__main__":
    main()