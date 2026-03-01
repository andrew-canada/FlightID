import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

def clean_numeric_currency(x):
    try: return float(str(x).replace('$','').replace(',',''))
    except: return np.nan

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Basic Cleansing
    df['fare_clean'] = df['fare'].apply(clean_numeric_currency)
    df['passengers_clean'] = pd.to_numeric(df['passengers'].str.replace(',',''), errors='coerce')
    df['nsmiles_clean'] = pd.to_numeric(df['nsmiles'], errors='coerce')
    
    # Derived Engineering
    df['nkm'] = df['nsmiles_clean'] * 1.60934
    df['fare_per_mile'] = df['fare_clean'] / df['nsmiles_clean'].replace(0, np.nan)
    
    # Hub & Market Indicators
    df['orig_total'] = pd.to_numeric(df.get('TotalFaredPax_city1', 0), errors='coerce')
    df['dest_total'] = pd.to_numeric(df.get('TotalFaredPax_city2', 0), errors='coerce')
    df['hub_ratio'] = df['orig_total'] / df['dest_total'].replace(0, np.nan)
    df['origin_is_hub'] = (df['orig_total'] > df['dest_total']).astype(int)
    
    # Competitive Index
    a = pd.to_numeric(df.get('TotalPerLFMkts_city1', 0), errors='coerce')
    b = pd.to_numeric(df.get('TotalPerLFMkts_city2', 0), errors='coerce')
    df['competitive_pressure'] = (a + b) / 2.0
    
    # Carrier Freq
    for col in ['carrier_lg','carrier_low']:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[col+'_freq'] = df[col].map(freq)
            
    return df

def compute_shap(model, X_test, out_path):
    """Generates and saves the SHAP summary plot for feature importance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('SHAP Feature Importance: Drivers of Fare Inefficiency')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def detect_inflation(df, features, model, X_full, threshold=0.15):
    """Uses the model to predict fares and flags those exceeding the threshold."""
    preds_per_mile = model.predict(X_full)
    
    df = df.copy()
    df['pred_per_mile'] = preds_per_mile
    df['predicted_fare'] = df['pred_per_mile'] * df['nsmiles_clean'].astype(float)
    df['actual_fare'] = df['fare_clean'].astype(float)
    
    # Calculate difference
    df['diff'] = df['actual_fare'] - df['predicted_fare']
    df['pct_diff'] = df['diff'] / (df['predicted_fare'].replace({0: np.nan}))
    df['inflated'] = (df['pct_diff'] > threshold)
    return df

def suggest_alternatives(df, features, top_k=3):
    """Finds lower-cost city-pair alternatives for inflated routes."""
    suggestions = []
    # Only look at inflated flights
    inflated = df[df['inflated'] == True]
    
    grouped = df.groupby('city1')
    for origin, g in grouped:
        candidates = g[['city2', 'predicted_fare', 'nsmiles_clean']].dropna()
        if candidates.empty: continue
        
        candidates['cost_per_km'] = candidates['predicted_fare'] / (candidates['nsmiles_clean'].astype(float) * 1.60934).replace({0: np.nan})
        ranked = candidates.sort_values('cost_per_km')
        
        # Filter for origin matches
        for _, r in inflated[inflated['city1'] == origin].iterrows():
            dest = r['city2']
            top = ranked[~(ranked['city2'] == dest)].head(top_k)
            alt_list = top['city2'].tolist()
            
            suggestions.append({
                'city1': origin, 'city2': dest,
                'actual_fare': r['actual_fare'], 'predicted_fare': r['predicted_fare'],
                'pct_diff': r['pct_diff'], 'suggested_alternatives': ';'.join(alt_list)
            })
    return pd.DataFrame(suggestions)

def rank_corridors(df):
    """Ranks corridors by cost-effectiveness based on model predictions."""
    grp = df.groupby(['city1', 'city2']).agg(
        median_pred=('predicted_fare', 'median'),
        median_dist=('nsmiles_clean', 'median'),
        n=('predicted_fare', 'count')
    ).reset_index()
    grp['pred_per_mile'] = grp['median_pred'] / grp['median_dist'].replace({0: np.nan})
    return grp.sort_values('pred_per_mile')