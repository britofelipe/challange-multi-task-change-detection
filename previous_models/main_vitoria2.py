import geopandas as gpd
from sklearn.impute import KNNImputer
from category_encoders import TargetEncoder
import pandas as pd
import numpy as np
import cv2
from shapely.geometry import Polygon
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


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
        # Converter colunas de data para datetime
        for col in self.date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        # Criar features de diferença entre datas
        for i in range(len(self.date_cols) - 1):
            df[f'days_between_{i}_{i+1}'] = (df[self.date_cols[i+1]] - df[self.date_cols[i]]).dt.days

        # Criar feature de frequência de mudança
        df["change_frequency"] = df[[f'days_between_{i}_{i+1}' for i in range(len(self.date_cols) - 1)]].count(axis=1)

        # Criar feature de velocidade média das mudanças
        df["change_speed"] = df[[f'days_between_{i}_{i+1}' for i in range(len(self.date_cols) - 1)]].mean(axis=1)

        # Nova feature: Taxa de variação temporal
        total_days = (df[self.date_cols[-1]] - df[self.date_cols[0]]).dt.days
        df["change_rate"] = df["change_frequency"] / (total_days + 1e-9)  # Evita divisão por zero

        # Aceleração da Mudança**
        df["change_acceleration"] = df["change_speed"].diff().fillna(0)

        # Taxa de Mudança Acumulada**
        df["cumulative_change_rate"] = df["change_rate"].cumsum()

        # Nova feature: Identificação de períodos críticos (2015-2018)
        df["high_change_2015_2018"] = (
            ((df[self.date_cols[0]].dt.year >= 2015) & (df[self.date_cols[0]].dt.year <= 2018)) |
            ((df[self.date_cols[-1]].dt.year >= 2015) & (df[self.date_cols[-1]].dt.year <= 2018))
        ).astype(int)

        # Nova feature: Extração de mês para sazonalidade
        for i, col in enumerate(self.date_cols):
            df[f'month_date{i}'] = df[col].dt.month
        
        return df


    def _calculate_geometric_features(self, df):
        print("Calculating geometric features...")
        df['area'] = df.geometry.apply(lambda g: g.area if g else 0)
        df['perimeter'] = df.geometry.apply(lambda g: g.length if g else 0)
        df['centroid_x'] = df.geometry.apply(lambda g: g.centroid.x if g else 0)
        df['centroid_y'] = df.geometry.apply(lambda g: g.centroid.y if g else 0)
        
        # Compactação (quão circular a geometria é)
        df['compactness'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2 + 1e-9)
        
        # Largura e altura da bounding box
        df['bounding_width'] = df.geometry.bounds['maxx'] - df.geometry.bounds['minx']
        df['bounding_height'] = df.geometry.bounds['maxy'] - df.geometry.bounds['miny']
        df['bounding_diagonal'] = np.sqrt(df['bounding_width'] ** 2 + df['bounding_height'] ** 2)

        df['diagonal_area_ratio'] = df['bounding_diagonal'] / (df['area'] + 1e-9)
        df['elongation_ratio'] = df['bounding_height'] / (df['bounding_width'] + 1e-9)
        df['elongation'] = df['bounding_width'] / (df['bounding_height'] + 1e-9)
        df['convexity'] = df.geometry.apply(lambda g: g.area / g.convex_hull.area if g else 0)
        df['circularity'] = df.geometry.apply(lambda g: (4 * np.pi * g.area) / (g.length ** 2 + 1e-9) if g else 0)

        # Razão área/perímetro normalizada
        df['area_perimeter_ratio'] = df['area'] / (df['perimeter'] + 1e-9)

        # **Cálculo dos Momentos de Hu**
        def hu_moments(polygon):
            try:
                if polygon.is_empty:
                    return [0] * 7
                coords = np.array(polygon.exterior.coords)
                coords = coords.astype(np.float32)
                moments = cv2.moments(coords)
                hu_moments = cv2.HuMoments(moments).flatten()
                return hu_moments.tolist()
            except:
                return [0] * 7

        # Aplicando o cálculo dos momentos de Hu e extraindo cada um dos 7 valores
        df['hu_moments'] = df.geometry.apply(hu_moments)
        for i in range(7):
            df[f'hu_moment_{i}'] = df['hu_moments'].apply(lambda x: x[i])
        # Transformação logarítmica dos Momentos de Hu
        for i in range(7):
            df[f'hu_moment_{i}_log'] = np.sign(df[f'hu_moment_{i}']) * np.log1p(np.abs(df[f'hu_moment_{i}']))

        # Remover coluna auxiliar
        df.drop(columns=['hu_moments'], inplace=True)

        return df
        


    def _calculate_spectral_features(self, df):
        print("Calculating spectral features...")
        # Check available spectral bands dynamically
        available_dates = []
        for d in range(5):  # Adjusted for typical date ranges in geospatial data
            if f'img_red_mean_date{d}' in df.columns:
                available_dates.append(d)
        
        print(f"Found spectral data for dates: {available_dates}")
        for d in available_dates:
            # Calcular NDVI para todas as datas disponíveis
            if f'img_red_mean_date{d}' in df.columns and f'img_blue_mean_date{d}' in df.columns:
                df[f'NDVI_date{d}'] = (df[f'img_red_mean_date{d}'] - df[f'img_blue_mean_date{d}']) / \
                                      (df[f'img_red_mean_date{d}'] + df[f'img_blue_mean_date{d}'] + 1e-9)

            # Calcular SAVI para todas as datas disponíveis (ajuste do NDVI para solo seco)
            L = 0.5  # Fator de ajuste para áreas com vegetação escassa
            if f'img_red_mean_date{d}' in df.columns and f'img_blue_mean_date{d}' in df.columns:
                df[f'SAVI_date{d}'] = ((df[f'img_red_mean_date{d}'] - df[f'img_blue_mean_date{d}']) / 
                                      (df[f'img_red_mean_date{d}'] + df[f'img_blue_mean_date{d}'] + L + 1e-9)) * (1 + L)

            # Calcular NDBI para todas as datas disponíveis (Índice de áreas construídas)
            if f'img_red_mean_date{d}' in df.columns and f'img_green_mean_date{d}' in df.columns:
                df[f'NDBI_date{d}'] = (df[f'img_red_mean_date{d}'] - df[f'img_green_mean_date{d}']) / \
                                      (df[f'img_red_mean_date{d}'] + df[f'img_green_mean_date{d}'] + 1e-9)

            # NDWI (Índice de Diferença Normalizada da Água)**
            if f'img_green_mean_date{d}' in df.columns and f'img_blue_mean_date{d}' in df.columns:
                df[f'NDWI_date{d}'] = (df[f'img_green_mean_date{d}'] - df[f'img_blue_mean_date{d}']) / \
                                    (df[f'img_green_mean_date{d}'] + df[f'img_blue_mean_date{d}'] + 1e-9)

        # **🔹 Criar diferenças espectrais entre datas consecutivas**
        for i in range(1, len(available_dates)):
            for index in ['NDVI', 'SAVI', 'NDBI', 'NDWI']:  # Adicionado NDWI
                if f'{index}_date{available_dates[i]}' in df.columns and f'{index}_date{available_dates[i-1]}' in df.columns:
                    df[f'{index}_diff_{available_dates[i-1]}_{available_dates[i]}'] = (
                        df[f'{index}_date{available_dates[i]}'] - df[f'{index}_date{available_dates[i-1]}']
                    )

        # Coeficiente de Variação (CV) dos Índices Espectrais**
        for index in ["NDVI", "NDBI", "SAVI", "NDWI"]:
            index_cols = [f"{index}_date{d}" for d in available_dates if f"{index}_date{d}" in df.columns]
            if len(index_cols) > 1:
                df[f"{index}_cv"] = df[index_cols].std(axis=1) / (df[index_cols].mean(axis=1) + 1e-9)

        #Média e Desvio-Padrão dos Índices Espectrais ao longo do tempo**
        for index in ["NDVI", "NDBI", "SAVI", "NDWI"]:
            index_cols = [f"{index}_date{d}" for d in available_dates if f"{index}_date{d}" in df.columns]
            if len(index_cols) > 1:
                df[f"{index}_mean"] = df[index_cols].mean(axis=1)
                df[f"{index}_std"] = df[index_cols].std(axis=1)

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

        return grid_search.best_estimator_    
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
        # Aplicar encoding de frequências em vez de One-Hot Encoding
        for col in ['urban_type', 'geography_type']:
            freq_map = self.train[col].value_counts(normalize=True)  # Frequência das categorias
            self.train[f'{col}_freq'] = self.train[col].map(freq_map)
            self.test[f'{col}_freq'] = self.test[col].map(freq_map)

        # Criar interação entre urban_type e geography_type
        self.train["urban_geo_interaction"] = self.train["urban_type"] + "_" + self.train["geography_type"]
        self.test["urban_geo_interaction"] = self.test["urban_type"] + "_" + self.test["geography_type"]

    # Tratar variáveis categóricas (usar One-Hot apenas para urban_geo_interaction)
        if 'urban_geo_interaction' in self.train.columns:
            print("Encoding categorical features...")
            self.ohe.fit(pd.concat([self.train[['urban_geo_interaction']], self.test[['urban_geo_interaction']]]))

            train_encoded = self.ohe.transform(self.train[['urban_geo_interaction']])
            test_encoded = self.ohe.transform(self.test[['urban_geo_interaction']])

            train_encoded_df = pd.DataFrame(train_encoded, 
                                            columns=self.ohe.get_feature_names_out(['urban_geo_interaction']),
                                            index=self.train.index)

            test_encoded_df = pd.DataFrame(test_encoded, 
                                        columns=self.ohe.get_feature_names_out(['urban_geo_interaction']),
                                        index=self.test.index)
        else:
            train_encoded_df = test_encoded_df = pd.DataFrame()

        # Feature selection
        temporal_features = [c for c in self.train.columns if 'days_between' in c] + \
                            ['change_frequency', 'change_speed', 'change_rate', 
                            'change_acceleration', 'cumulative_change_rate',  # **Novas features**
                            'high_change_2015_2018'] + \
                            [f'month_date{i}' for i in range(5)]        
        spectral_features = [c for c in self.train.columns if 'NDVI' in c] + \
                            [c for c in self.train.columns if 'SAVI' in c] + \
                            [c for c in self.train.columns if 'NDBI' in c] + \
                            [c for c in self.train.columns if 'NDWI' in c] + \
                            [c for c in self.train.columns if 'spectral_diff_' in c] + \
                            [c for c in self.train.columns if '_cv' in c] + \
                            [c for c in self.train.columns if '_mean' in c] + \
                            [c for c in self.train.columns if '_std' in c]  # **Adicionando estatísticas dos índices espectrais**
        geometric_features = ['area', 'perimeter', 'compactness', 'centroid_x', 'centroid_y', 
                            'bounding_width', 'bounding_height', 'bounding_diagonal', 
                            'diagonal_area_ratio', 'elongation_ratio',  
                            'convexity', 'circularity', 'area_perimeter_ratio'] + \
                            [f'hu_moment_{i}' for i in range(7)] + \
                            [f'hu_moment_{i}_log' for i in range(7)]
        categorical_features = ['urban_type_freq', 'geography_type_freq']  

        print(f"Selected features BEFORE FILTERING:\n- Temporal: {len(temporal_features)}\n- Spectral: {len(spectral_features)}\n- Geometric: {len(geometric_features)}\n- Categorical: {len(categorical_features)}")

        # Convert all feature names to strings
        def sanitize_columns(df):
            df.columns = df.columns.astype(str)
            return df

        # Create final datasets
        self.X_train = pd.concat([
            self.train[temporal_features + spectral_features + geometric_features].pipe(sanitize_columns),
            train_encoded_df
        ], axis=1)

        self.X_test = pd.concat([
            self.test[temporal_features + spectral_features + geometric_features].pipe(sanitize_columns),
            test_encoded_df
        ], axis=1)

       
        ##
        self.X_train = self.X_train.fillna(self.X_train.median())
        self.X_test = self.X_test.fillna(self.X_test.median())
        # **Definir y_train antes da seleção de features**
        if 'change_type' in self.train.columns:
            self.y_train = self.train['change_type'].map(change_type_map)
            print("Class distribution:\n", self.y_train.value_counts())
        # Normalization
        print("Applying feature scaling...")
        num_features_to_keep = min(200, self.X_train.shape[1])  # Selecionar as 100 melhores features ou menos se houver menos features disponíveis

        selector = SelectKBest(score_func=f_classif, k=num_features_to_keep)
        self.X_train = selector.fit_transform(self.X_train, self.y_train)
        self.X_test = selector.transform(self.X_test)

        print(f"Reduced feature set to: {self.X_train.shape[1]} features.")

        print("Encoding categorical variables without Target Encoding...")
        self.train = pd.get_dummies(self.train, columns=['urban_type', 'geography_type'])
        self.test = pd.get_dummies(self.test, columns=['urban_type', 'geography_type'])

        # Garantir que todas as colunas do treino existam no teste
        missing_cols = set(self.train.columns) - set(self.test.columns)
        for col in missing_cols:
            self.test[col] = 0  # Adicionar colunas faltantes no test

        # Garantir a mesma ordem das colunas
        self.test = self.test[self.train.columns]
        print("Encoding completed.")

        print("Applying KNN imputation for missing values...")
        imputer = KNNImputer(n_neighbors=5)
        self.X_train = imputer.fit_transform(self.X_train)
        self.X_test = imputer.transform(self.X_test)

        # Normalizar os dados
        print("Applying feature scaling...")
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        print("Final feature set:", self.X_train.shape)    
    
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
                max_depth=10,
                feature_fraction=0.8,  # Usa 80% das features em cada árvore para evitar dependência excessiva
                min_gain_to_split=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                early_stopping_round=50,
                n_jobs=-1,  # Usa todos os núcleos disponíveis sem autodetecção
                min_child_samples=50,  # Evita divisões muito pequenas e melhora generalização
                verbosity=-1,
                num_leaves=63,
                max_bin=511
            )),

            
            ('KNN', KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='minkowski'
        ))

        ]
        
        # Cross-validation setup
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        self.test_preds = np.zeros((len(self.models), TEST_SIZE, len(change_type_map)))
        results = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_tr, X_val = self.X_train[train_idx], self.X_train[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            for model_idx, (model_name, model) in enumerate(self.models):
                print(f"\nTraining {model_name} - Fold {fold+1}/{N_FOLDS}")

                if model_name == 'XGBoost':
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                elif model_name == 'LightGBM':
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        eval_names=['validation'],
                        #eval_metric='f1',  
                        #verbose=False
                    )
                else:  # KNN does not require validation sets
                    model.fit(X_tr, y_tr)
                        
                
                # Generate predictions
                fold_probs = model.predict_proba(self.X_test)
                self.test_preds[model_idx] += fold_probs / N_FOLDS
                
                # Validation metrics
                val_pred = model.predict(X_val)
                f1 = f1_score(y_val, val_pred, average='macro')
                acc = accuracy_score(y_val, val_pred)
                print(f"{model_name} Fold {fold+1} - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
                
                # Save individual model predictions
                results.append({"Model": model_name, "Fold": fold+1, "F1": f1, "Accuracy": acc})
                model_preds = fold_probs.argmax(axis=1)
                self._save_submission(model_preds, f"{model_name}_fold{fold+1}")
        results_df = pd.DataFrame(results)
        print("\nCross-validation results:")
        print(results_df.groupby("Model")[["F1", "Accuracy"]].mean())
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
        weights = [ 0.4,0.4,0.2] 
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
