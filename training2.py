import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
import re
from catboost import CatBoostClassifier

# Constants
N_FOLDS = 5
N_ESTIMATORS = 1000
RANDOM_STATE = 42
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

def train_models():
    print("\n📥 Loading preprocessed datasets...")
    
    # Load preprocessed train and test data
    train_df = pd.read_csv("train_x_selected.csv")
    test_df = pd.read_csv("test_x_selected.csv")
    
    # Separate features and target variable
    X_train = train_df.drop(columns=["change_type"])
    y_train = train_df["change_type"]
    X_test = test_df

    print(f"✅ Data Loaded: Train Shape {X_train.shape}, Test Shape {X_test.shape}")

    def clean_feature_names(df):
        """Replace special characters in feature names to make them compatible with LightGBM."""
        df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]
        return df

    # Apply this to both train and test datasets **before training**
    X_train = clean_feature_names(X_train)
    X_test = clean_feature_names(X_test)

    print("✅ Feature names cleaned for LightGBM compatibility!")
    
    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

    # Define models
    models = [
        
        ('XGBoost', XGBClassifier(
            objective='multi:softmax',
            n_estimators=N_ESTIMATORS,
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
            
        ('LightGBM', LGBMClassifier(##mudei aqui para o modelo que tava no meu. ass:vitoria
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
    test_preds = np.zeros((len(models), len(X_test), len(np.unique(y_train))))  # Store predictions for test set
    train_preds = np.zeros((len(models), len(X_train), len(np.unique(y_train))))  # Store predictions for train set
    results = []
    print("📊 Training Models with Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        for model_idx, (model_name, model) in enumerate(models):
            print(f"\n🚀 Training {model_name} - Fold {fold+1}/{N_FOLDS}")

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
                    )
            else:  # KNN does not require validation sets
                model.fit(X_tr, y_tr)
            
            # Generate test set predictions
            fold_probs_test = model.predict_proba(X_test)
            test_preds[model_idx] += fold_probs_test / N_FOLDS  # Average predictions across folds
            
            # Generate train set predictions
            fold_probs_train = model.predict_proba(X_train)
            train_preds[model_idx] += fold_probs_train / N_FOLDS  # Average predictions across folds

            # Validation metrics
            val_pred = model.predict(X_val)
            f1 = f1_score(y_val, val_pred, average='macro')
            acc = accuracy_score(y_val, val_pred)
            print(f"{model_name} Fold {fold+1} - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
                
            # Save individual model predictions
            results.append({"Model": model_name, "Fold": fold+1, "F1": f1, "Accuracy": acc})
            model_preds = fold_probs_train.argmax(axis=1)
            #save_submission(val_pred, f"{model_name}_fold{fold+1}")
    results_df = pd.DataFrame(results)
    print("\nCross-validation results:")
    print(results_df.groupby("Model")[["F1", "Accuracy"]].mean())
    # Generate final ensemble prediction
    generate_final_submission(test_preds, train_preds, X_test, y_train)

def save_submission(preds, model_name):
    filename = f"submission_{model_name}_{TIMESTAMP}.csv"
    submission = pd.DataFrame({'Id': np.arange(len(preds)), 'change_type': preds})
    submission.to_csv(filename, index=False)
    print(f"💾 Saved submission: {filename}")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from collections import Counter

def generate_final_submission(test_preds, train_preds, X_test, y_train):
    print("\n📊 Generating Final Ensemble Submission...")

    # Weighted ensemble probabilities
    weights = [0.4, 0.4, 0.2]  # Weights for XGBoost, LightGBM, CatBoost
    weighted_probs_test = sum(w * test_preds[i] for i, w in enumerate(weights))
    weighted_probs_train = sum(w * train_preds[i] for i, w in enumerate(weights))

    # Assign final predictions with a fallback strategy
    def assign_labels(weighted_probs, model_preds):
        final_preds = np.argmax(weighted_probs, axis=1)  # Default highest probability

        for i in range(len(final_preds)):
            # If the highest probability is too low (i.e., uncertain prediction), fallback to model majority vote
            if np.max(weighted_probs[i]) < 1e-5:  # Threshold for "no confident prediction"
                model_votes = [np.argmax(model_preds[j][i]) for j in range(len(model_preds))]
                
                # If no confident vote is available, assign the most common class in y_train
                if len(model_votes) > 0:
                    final_preds[i] = Counter(model_votes).most_common(1)[0][0]
                else:
                    final_preds[i] = Counter(y_train).most_common(1)[0][0]  # Assign the most frequent class

        return final_preds

    # Apply label assignment to both train and test sets
    final_preds_test = assign_labels(weighted_probs_test, test_preds)
    final_preds_train = assign_labels(weighted_probs_train, train_preds)

    # Compute F1 Score on training set
    f1_macro = f1_score(y_train, final_preds_train, average='macro')
    f1_weighted = f1_score(y_train, final_preds_train, average='weighted')
    final_accuracy = accuracy_score(y_train, final_preds_train)

    print(f"🎯 Final Ensemble Accuracy: {final_accuracy:.4f}")
    print(f"🎯 Final Ensemble Train Macro F1 Score: {f1_macro:.4f}")
    print(f"🎯 Final Ensemble Train Weighted F1 Score: {f1_weighted:.4f}")


    # Save final test submission
    filename_test = f"submission_ENSEMBLE_training_{TIMESTAMP}.csv"
    submission_test = pd.DataFrame({'Id': np.arange(len(X_test)), 'change_type': final_preds_test})
    submission_test.to_csv(filename_test, index=False)
    print(f"✅ Final ensemble test submission saved: {filename_test}")

    # Save final train predictions
    filename_train = f"train_ensemble_preds_{TIMESTAMP}.csv"
    submission_train = pd.DataFrame({'Id': np.arange(len(y_train)), 'change_type': final_preds_train})
    submission_train.to_csv(filename_train, index=False)
    print(f"✅ Final ensemble train predictions saved: {filename_train}")

# Run training function
if __name__ == "__main__":
    train_models()
