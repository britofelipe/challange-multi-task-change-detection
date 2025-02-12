import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Global configurations
N_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 120526
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class GeoFeatureEngineer:
    def __init__(self):
        self.crs = 'EPSG:3857'
        self.date_cols = [f'date{i}' for i in range(5)]
        
    def _calculate_temporal_features(self, df):
        for i in range(len(self.date_cols)-1):
            df[f'days_between_{i}_{i+1}'] = (df[self.date_cols[i+1]] - df[self.date_cols[i]]).dt.days
        return df

    def _calculate_geometric_features(self, df):
        print("Calculating geometric features...")
        df['area'] = df.geometry.apply(lambda g: g.area if g else 0)
        df['perimeter'] = df.geometry.apply(lambda g: g.length if g else 0)
        df['centroid_x'] = df.geometry.apply(lambda g: g.centroid.x if g else 0)
        df['centroid_y'] = df.geometry.apply(lambda g: g.centroid.y if g else 0)
        df['compactness'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2 + 1e-9)
        return df

    def _calculate_spectral_features(self, df):
        print("Calculating spectral features...")
        # Check available spectral bands dynamically
        available_dates = []
        for d in range(3):  # Adjusted for typical date ranges in geospatial data
            if f'img_red_mean_date{d}' in df.columns:
                available_dates.append(d)
        
        print(f"Found spectral data for dates: {available_dates}")
        for d in available_dates:
            for band in ['red', 'green', 'blue']:
                mean_col = f'img_{band}_mean_date{d}'
                std_col = f'img_{band}_std_date{d}'
                
                if mean_col in df.columns:
                    df[f'spectral_mean_{d}_{band}'] = df[mean_col]
                if std_col in df.columns:
                    df[f'spectral_var_{d}_{band}'] = df[std_col] ** 2
        return df

    def process(self, df):
        print("\nProcessing dataframe...")
        # Date conversion
        for col in self.date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Feature engineering
        df = self._calculate_temporal_features(df)
        df = self._calculate_geometric_features(df)
        df = self._calculate_spectral_features(df)
        
        # Categorical interactions
        if 'urban_type' in df.columns and 'geography_type' in df.columns:
            df['urban_geo_interaction'] = df['urban_type'] + "_" + df['geography_type']
        return df

class CompetitionPipeline:
    def __init__(self, train_path, test_path):
        print(f"\nLoading data from:\n- Train: {train_path}\n- Test: {test_path}")
        self.train = gpd.read_file(train_path)
        self.test = gpd.read_file(test_path)
        self.test_ids = self.test.index.tolist()
        self._validate_test_size()
        
        self.fe = GeoFeatureEngineer()
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
    def _validate_test_size(self):
        if len(self.test) != TEST_SIZE:
            raise ValueError(f"Test size must be {TEST_SIZE}, got {len(self.test)}")
        print(f"Test size validation passed: {TEST_SIZE} rows")
        
    def _handle_missing_values(self, df):
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        return df

    def prepare_data(self):
        print("\nPreparing data...")
        # Process dataframes
        self.train = self.fe.process(self.train)
        self.test = self.fe.process(self.test)
        
        # Handle categorical features
        if 'urban_geo_interaction' in self.train.columns:
            print("Encoding categorical features...")
            self.ohe.fit(pd.concat([
                self.train[['urban_geo_interaction']], 
                self.test[['urban_geo_interaction']]
            ]))
            
            train_encoded = self.ohe.transform(self.train[['urban_geo_interaction']])
            test_encoded = self.ohe.transform(self.test[['urban_geo_interaction']])
        else:
            train_encoded = test_encoded = pd.DataFrame()
        
        # Feature selection
        temporal_features = [c for c in self.train.columns if 'days_between' in c]
        spectral_features = [c for c in self.train.columns if 'spectral_' in c]
        geometric_features = ['area', 'perimeter', 'compactness', 'centroid_x', 'centroid_y']
        
        print(f"Selected features:\n- Temporal: {len(temporal_features)}\n- Spectral: {len(spectral_features)}\n- Geometric: {len(geometric_features)}")
        
        # Convert all feature names to strings
        def sanitize_columns(df):
            df.columns = df.columns.astype(str)
            return df

        # Create final datasets
        self.X_train = pd.concat([
            self.train[temporal_features + spectral_features + geometric_features].pipe(sanitize_columns),
            pd.DataFrame(train_encoded, 
                        columns=self.ohe.get_feature_names_out(['urban_geo_interaction']),
                        index=self.train.index)
        ], axis=1)

        self.X_test = pd.concat([
            self.test[temporal_features + spectral_features + geometric_features].pipe(sanitize_columns),
            pd.DataFrame(test_encoded,
                        columns=self.ohe.get_feature_names_out(['urban_geo_interaction']),
                        index=self.test.index)
        ], axis=1)

        # Explicitly name all columns as strings
        self.X_train.columns = self.X_train.columns.astype(str)
        self.X_test.columns = self.X_test.columns.astype(str)
        
        # Normalization
        print("Applying feature scaling...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Target processing
        if 'change_type' in self.train.columns:
            self.y_train = self.train['change_type'].map(change_type_map)
            print("Class distribution:\n", self.y_train.value_counts())

    def train_ensemble(self):
        print("\nTraining model ensemble...")
        # Class weights
        class_weights = compute_class_weight('balanced', 
                                           classes=np.unique(self.y_train), 
                                           y=self.y_train)
        
        # Model configurations
        self.models = [
            ('XGBoost', XGBClassifier(
                objective='multi:softmax',
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                scale_pos_weight=class_weights,
                random_state=RANDOM_STATE,
                eval_metric='mlogloss',
                early_stopping_rounds=50
            )),
            
            ('LightGBM', LGBMClassifier(
                objective='multiclass',
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                early_stopping_round=50
            )),
            
            ('CatBoost', CatBoostClassifier(
                loss_function='MultiClass',
                iterations=1000,
                learning_rate=0.05,
                depth=8,
                subsample=0.8,
                random_strength=0.1,
                verbose=False,
                class_weights=class_weights,
                random_state=RANDOM_STATE,
                early_stopping_rounds=50
            )),

            ('KNN', KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='minkowski'
            )),

            ('RadiusNeighbors', RadiusNeighborsClassifier(
                radius=1.0,
                weights='distance',
                metric='minkowski'
            ))
        ]
        
        # Cross-validation setup
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        self.test_preds = np.zeros((len(self.models), TEST_SIZE, len(change_type_map)))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_tr, X_val = self.X_train[train_idx], self.X_train[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            for model_idx, (model_name, model) in enumerate(self.models):
                print(f"\nTraining {model_name} - Fold {fold+1}/{N_FOLDS}")

                if model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                else:  # KNN e RadiusNeighbors
                    model.fit(X_tr, y_tr)
                
                # Generate predictions
                fold_probs = model.predict_proba(self.X_test)
                self.test_preds[model_idx] += fold_probs / N_FOLDS
                
                # Validation metrics
                val_pred = model.predict(X_val)
                f1 = f1_score(y_val, val_pred, average='macro')
                print(f"{model_name} Fold {fold+1} Macro F1: {f1:.4f}")
                
                # Save individual model predictions
                model_preds = fold_probs.argmax(axis=1)
                self._save_submission(model_preds, f"{model_name}_fold{fold+1}")

    def _save_submission(self, preds, model_name):
        filename = f"submission_{model_name}_{TIMESTAMP}.csv"
        submission = pd.DataFrame({'Id': self.test_ids, 'change_type': preds})
        
        if len(submission) != TEST_SIZE:
            raise ValueError(f"Invalid submission size: {len(submission)}")
        
        submission.to_csv(filename, index=False)
        print(f"Saved submission: {filename}")

    def generate_final_submission(self):
        print("\nGenerating final ensemble submission...")
        # Weighted ensemble probabilities
        weights = [0.4, 0.3, 0.3]  # XGBoost, LightGBM, CatBoost
        weighted_probs = sum(w * self.test_preds[i] for i, w in enumerate(weights))
        final_preds = weighted_probs.argmax(axis=1)
        
        # Save final submission
        filename = f"submission_ENSEMBLE_{TIMESTAMP}.csv"
        submission = pd.DataFrame({'Id': self.test_ids, 'change_type': final_preds})
        submission.to_csv(filename, index=False)
        print(f"Final ensemble submission saved: {filename}")

# Main execution
if __name__ == "__main__":
    change_type_map = {
        'Demolition': 0, 
        'Road': 1, 
        'Residential': 2, 
        'Commercial': 3,
        'Industrial': 4, 
        'Mega Projects': 5
    }
    
    try:
        pipeline = CompetitionPipeline('train.geojson', 'test.geojson')
        pipeline.prepare_data()
        
        if hasattr(pipeline, 'y_train'):
            pipeline.train_ensemble()
            pipeline.generate_final_submission()
        else:
            print("Skipping training - no target column found")
            
    except Exception as e:
        print(f"Error in pipeline execution: {str(e)}")
        raise