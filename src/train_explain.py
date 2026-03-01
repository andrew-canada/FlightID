import argparse
from pathlib import Path
from pyexpat import features
from pyexpat import features
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
from utils import feature_engineer, compute_shap, detect_inflation, suggest_alternatives, rank_corridors

from utils import feature_engineer

def select_features():
    # Target is fare_per_mile, so we exclude it from features
    return [
        'nkm', 'passengers_clean',
        'orig_total', 'dest_total', 'hub_ratio', 'origin_is_hub',
        'hub_intensity_orig', 'hub_intensity_dest', 'market_concentration', 
        'competitive_pressure', 'carrier_lg_freq', 'carrier_low_freq'
    ]

def train_model(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators=1000, 
        learning_rate=0.05, 
        random_state=3169, 
        tree_method='hist',
        early_stopping_rounds=50,
        eval_metric='mae'
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', default='outputs')
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and Process
    df = feature_engineer(pd.read_csv(args.csv, dtype=str))
    df = pd.get_dummies(df, columns=[c for c in ['Year','quarter'] if c in df.columns], drop_first=True)
    
    all_features = select_features()
    features = [f for f in all_features if f in df.columns]
    
    if 'fare_per_mile' not in df.columns:
        df['fare_per_mile'] = df['fare_clean'] / df['nsmiles_clean'].replace(0, np.nan)

    # 2. Train with Leakage-Proofing
    cols_to_check = ['fare_per_mile'] + features
    df = df.dropna(subset=[c for c in cols_to_check if c in df.columns])
    
    X = df[features]
    y = df['fare_per_mile']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3169)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=3169)

    imputer = SimpleImputer(strategy='median')
    X_tr = pd.DataFrame(imputer.fit_transform(X_tr), columns=features)
    X_val = pd.DataFrame(imputer.transform(X_val), columns=features)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=features)

    model = train_model(X_tr, y_tr, X_val, y_val)
    
    # Save artifacts
    joblib.dump(model, out_dir / 'model.joblib')
    joblib.dump(imputer, out_dir / 'imputer.joblib')

    # 3. ANALYSIS PHASE (The missing pieces)
    print('Computing SHAP summary...')
    compute_shap(model, X_test, out_dir / 'shap_summary.png')

    print('Detecting inflated fares...')
    # Re-impute full dataset for inference
    X_full = pd.DataFrame(imputer.transform(df[features]), columns=features)
    df_preds = detect_inflation(df, features, model, X_full) # Note: Ensure detect_inflation uses X_full
    df_preds.to_csv(out_dir / 'all_with_predictions.csv', index=False)

    print('Suggesting alternatives...')
    suggestions = suggest_alternatives(df_preds, features)
    suggestions.to_csv(out_dir / 'inflated_flights.csv', index=False)

    print('Ranking corridors...')
    corridors = rank_corridors(df_preds)
    corridors.to_csv(out_dir / 'corridor_rankings.csv', index=False)

    print(f'Done. All artifacts saved to {out_dir}')

if __name__ == '__main__':
    main()