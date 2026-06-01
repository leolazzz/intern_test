import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


def load_data(file_path='https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv'):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        try:
            df = pd.read_csv('predictive_maintenance.csv')
            print(f"Data loaded from predictive_maintenance.csv: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(
                "Dataset not found. Please download ai4i2020.csv from "
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
            )


def display_basic_info(df):
    """Display basic dataset information."""
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Dataset size: {df.shape[0]} samples, {df.shape[1]} features")

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    # Target distribution
    target_col = 'Machine failure'
    if target_col in df.columns:
        print(f"\n--- Target Distribution ({target_col}) ---")
        dist = df[target_col].value_counts()
        print(f"0 (No Failure): {dist[0]} ({dist[0] / len(df) * 100:.2f}%)")
        print(f"1 (Failure):    {dist[1]} ({dist[1] / len(df) * 100:.2f}%)")
    else:
        print("\nWARNING: 'Machine failure' column not found!")


def preprocess_data(df):
    if 'UDI' in df.columns:
        df = df.drop('UDI', axis=1)
    elif 'UID' in df.columns:
        df = df.drop('UID', axis=1)

    if 'Product ID' in df.columns:
        df['ProductType'] = df['Product ID'].str[0]
        df = df.drop('Product ID', axis=1)
    if 'Type' in df.columns:
        df['ProductType'] = df['Type']
        df = df.drop('Type', axis=1)

    le = LabelEncoder()
    df['ProductType'] = le.fit_transform(df['ProductType'])

    target_col = 'Machine failure'
    if target_col not in df.columns:
        raise ValueError("Target column 'Machine failure' not found")

    failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    for col in failure_modes:
        if col in df.columns:
            df = df.drop(col, axis=1)

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, feature_names


def train_random_forest(X_train, y_train):
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("Random Forest training completed.")
    return rf


def train_xgboost(X_train, y_train):
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    print("XGBoost training completed.")
    return xgb_model


def evaluate_model(model, X_val, y_val, model_name):
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred, zero_division=0),
        'Recall': recall_score(y_val, y_pred),
        'F1-Score': f1_score(y_val, y_pred),
        'ROC-AUC': roc_auc_score(y_val, y_proba)
    }

    print(f"\n--- {model_name} Results ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['No Failure', 'Failure']))

    return metrics, y_proba, y_pred


def generate_alert_table(X_val_original, y_val, y_proba, feature_names):
    results = pd.DataFrame({
        'sample_id': range(len(y_val)),
        'actual': y_val.values if hasattr(y_val, 'values') else y_val,
        'probability': y_proba,
        'alert': y_proba >= 0.05
    })
    results = results.sort_values('probability', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("ALERT TABLE (Top 20 samples by failure probability)")
    print("=" * 60)
    print(f"{'sample_id':<12} {'actual':<10} {'probability':<12} {'alert':<8}")
    print("-" * 60)

    for i in range(min(20, len(results))):
        row = results.iloc[i]
        print(f"{row['sample_id']:<12} {row['actual']:<10} {row['probability']:<12.4f} {row['alert']:<8}")

    total_alerts = results['alert'].sum()
    correct_alerts = ((results['alert']) & (results['actual'] == 1)).sum()
    false_alerts = ((results['alert']) & (results['actual'] == 0)).sum()

    print("\n" + "-" * 60)
    print(f"Total alerts (probability >= 0.05): {total_alerts} out of {len(results)} samples")
    print(f"Correct alerts (actual failure): {correct_alerts}")
    print(f"False alerts (false positives): {false_alerts}")

    return results


def main():
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE SOLUTION")
    print("AI4I 2020 Dataset - Machine Failure Prediction")
    print("=" * 60)
    df = load_data()
    display_basic_info(df)

    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    X, y, scaler, feature_names = preprocess_data(df)
    print(f"Features used: {feature_names}")
    print(f"Preprocessed shape: {X.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]} samples")
    print(f"Validation size: {X_val.shape[0]} samples")

    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    rf_metrics, rf_proba, _ = evaluate_model(rf_model, X_val, y_val, "Random Forest")
    xgb_metrics, xgb_proba, _ = evaluate_model(xgb_model, X_val, y_val, "XGBoost")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<15} {'Random Forest':<20} {'XGBoost':<20}")
    print("-" * 60)
    for metric in rf_metrics.keys():
        print(f"{metric:<15} {rf_metrics[metric]:<20.4f} {xgb_metrics[metric]:<20.4f}")

    print("\n" + "=" * 60)
    print("ALERT GENERATION (XGBoost Model)")
    print("=" * 60)

    alert_results = generate_alert_table(X_val, y_val, xgb_proba, feature_names)

    return rf_model, xgb_model, scaler


if __name__ == "__main__":
    rf_model, xgb_model, scaler = main()