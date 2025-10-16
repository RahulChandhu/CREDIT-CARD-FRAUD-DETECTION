# prediction_helper.py
"""
Helpers for Credit Card Fraud Detection project:
- data loading
- preprocessing & feature engineering
- feature selection
- model training, evaluation
- save/load model + predict

Assumes the uploaded dataset is the standard credit card dataset
with columns including: 'Time', 'V1'...'V28', 'Amount', 'Class'.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE

# ---------------------
# I/O helpers
# ---------------------
def load_data(path: str) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    df = pd.read_csv(path)
    return df

# ---------------------
# Preprocessing / EDA helpers
# ---------------------
def basic_eda(df: pd.DataFrame) -> dict:
    """Return quick EDA summary (counts, missing, class balance)."""
    ed = {}
    ed['shape'] = df.shape
    ed['dtypes'] = df.dtypes.to_dict()
    ed['missing_count'] = df.isnull().sum().to_dict()
    ed['class_counts'] = df['Class'].value_counts().to_dict()
    ed['class_ratio'] = (df['Class'].value_counts(normalize=True).to_dict())
    return ed

def preprocess(df: pd.DataFrame, drop_duplicates: bool = True, scale_amount: bool = True) -> pd.DataFrame:
    """
    Preprocess dataset:
    - drop duplicates (optional)
    - create Hour from Time (if present)
    - scale Amount (optional) using StandardScaler (returns new column 'Amount_scaled')
    - fill missing with median for numeric
    """
    df = df.copy()

    if drop_duplicates:
        df = df.drop_duplicates()

    # Time -> Hour (if Time exists and appears to be seconds from beginning)
    if 'Time' in df.columns:
        try:
            # Time in original dataset is seconds since first transaction
            df['Hour'] = ((df['Time'] // 3600) % 24).astype(int)
        except Exception:
            # fallback: normalized Time
            df['Hour'] = df['Time']

    # Numeric missing fill
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())

    # Scale Amount
    if 'Amount' in df.columns and scale_amount:
        scaler = StandardScaler()
        df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
        # keep scaler for inference? We'll create a pipeline for production later
    else:
        if 'Amount' in df.columns:
            df['Amount_scaled'] = df['Amount']

    return df

# ---------------------
# Feature engineering & selection
# ---------------------
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add any feature-engineering steps:
    - use Amount_scaled
    - optionally interactions or aggregates
    For creditcard dataset (V1..V28) we mostly keep these plus scaled amount & Hour.
    """
    df = df.copy()
    # Example: amount * some V-s variable interactions (kept simple)
    if 'Amount_scaled' in df.columns and 'V1' in df.columns:
        df['Amt_V1_mul'] = df['Amount_scaled'] * df['V1']

    # If Hour exists, treat as numeric or generate simple bins
    if 'Hour' in df.columns:
        df['is_night'] = df['Hour'].apply(lambda h: 1 if (h >= 0 and h <= 6) else 0)

    return df

def select_features_by_model(X: pd.DataFrame, y: pd.Series, threshold: float = 0.01) -> list:
    """
    Use a RandomForest to get feature importances and select features above threshold.
    Returns a list of column names selected.
    """
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    sel = SelectFromModel(rf, threshold=threshold, prefit=True)
    selected_mask = sel.get_support()
    selected_features = X.columns[selected_mask].tolist()
    # If selection is empty, fallback to top 10 features
    if not selected_features:
        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[-10:]
        selected_features = X.columns[top_idx].tolist()
    return selected_features

# ---------------------
# Model training / evaluation
# ---------------------
def train_and_select_model(
    df: pd.DataFrame,
    feature_cols: list = None,
    test_size: float = 0.2,
    random_state: int = 42,
    oversample: bool = True,
    save_model_path: str = "models/best_model.joblib",
):
    """
    End-to-end training:
    - split
    - optional SMOTE oversampling on train
    - feature selection via RandomForest importances
    - train a classifier with GridSearchCV (RandomForest / Logistic)
    - evaluate and save model (joblib)
    Returns: dict with metrics and the fitted pipeline
    """
    os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)

    df = df.copy()
    assert 'Class' in df.columns, "DataFrame must contain 'Class' column as target."

    # Choose features
    if feature_cols is None:
        # remove target and categorical non-numeric
        feature_cols = [c for c in df.columns if c != 'Class' and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8]]
    X = df[feature_cols]
    y = df['Class']

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Optionally oversample train set to handle imbalance
    if oversample:
        sm = SMOTE(random_state=random_state, n_jobs=-1)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    # Feature selection
    selected_features = select_features_by_model(X_train_res, y_train_res, threshold="median")
    # ensure at least some features
    if not selected_features:
        selected_features = X_train_res.columns.tolist()

    X_train_fs = X_train_res[selected_features]
    X_test_fs = X_test[selected_features]

    # Build pipeline: scaler + classifier
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=random_state, n_jobs=-1))
    ])

    param_grid = {
        'clf': [RandomForestClassifier(random_state=random_state, n_jobs=-1),
                LogisticRegression(max_iter=1000, random_state=random_state)],
        'clf__n_estimators': [100, 200],           # used only for RF; LR ignores
        'clf__C': [0.1, 1.0],                      # used only for LR; RF ignores
    }

    # Use StratifiedKFold
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    gs = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_train_fs, y_train_res)

    best = gs.best_estimator_

    # Evaluate
    y_pred = best.predict(X_test_fs)
    y_proba = best.predict_proba(X_test_fs)[:, 1] if hasattr(best, "predict_proba") else best.decision_function(X_test_fs)

    report = classification_report(y_test, y_pred, output_dict=True)
    conf = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    metrics = {
        'classification_report': report,
        'confusion_matrix': conf.tolist(),
        'roc_auc': float(roc_auc),
        'average_precision': float(avg_precision),
        'selected_features': selected_features,
        'best_params': gs.best_params_,
    }

    # Save model + metadata (we'll persist pipeline and selected features)
    to_save = {
        'model': best,
        'selected_features': selected_features,
        'feature_cols': feature_cols,
    }
    joblib.dump(to_save, save_model_path)

    return metrics, save_model_path

# ---------------------
# Load model & predict
# ---------------------
def load_model(path: str = "models/best_model.joblib"):
    """Load the saved object containing 'model' and 'selected_features'."""
    data = joblib.load(path)
    model = data.get('model')
    selected_features = data.get('selected_features')
    return model, selected_features

def prepare_single_input(record: dict, selected_features: list):
    """
    Convert a single-record dict (feature name -> value) to a DataFrame with required columns.
    Missing columns will be filled with 0.
    """
    df = pd.DataFrame([record])
    # Ensure all selected features exist
    for c in selected_features:
        if c not in df.columns:
            df[c] = 0
    # Keep only selected features in proper order
    X = df[selected_features]
    return X

def predict_batch(df: pd.DataFrame, model_obj: dict = None, model=None, selected_features=None):
    """
    Predict on a batch DataFrame. Either pass model_obj (loaded joblib) or model + selected_features.
    Returns df with predictions and probabilities.
    """
    if model_obj is not None:
        model = model_obj['model']
        selected_features = model_obj['selected_features']

    X = df.copy()
    # Ensure selected features present
    for c in selected_features:
        if c not in X.columns:
            X[c] = 0
    X_sel = X[selected_features]

    preds = model.predict(X_sel)
    proba = model.predict_proba(X_sel)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_sel)

    result = df.copy()
    result['prediction'] = preds
    result['score'] = proba
    return result

# ---------------------
# Quick CLI-like training entrypoint
# ---------------------
if __name__ == "__main__":
    # Basic script to run training locally:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='creditcard.csv', help='Path to creditcard CSV')
    parser.add_argument('--save', type=str, default='models/best_model.joblib', help='Path to save model joblib')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    df = load_data(args.data)
    print("Initial shape:", df.shape)
    df = preprocess(df)
    df = feature_engineer(df)
    metrics, model_path = train_and_select_model(df, test_size=args.test_size, save_model_path=args.save)
    print("Training finished. Model saved to:", model_path)
    print("Metrics (rounded):")
    import pprint
    pprint.pprint(metrics)
