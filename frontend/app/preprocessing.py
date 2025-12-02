# preprocessing.py
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

def preprocess_for_streamlit(df_raw, scaler_obj, start_date, end_date):
    """
    Prepares raw data for prediction using a pre-fitted scaler and Seasonal Imputer logic.
    """
    # 1. Initial Resample and Indexing
    if 'Time' in df_raw.columns:
        df_raw.set_index('Time', inplace=True)
    df_raw.index = pd.to_datetime(df_raw.index)
    df_hourly = df_raw.resample('H').mean()

    # 2. Basic Feature Engineering
    df_hourly['hour'] = df_hourly.index.hour
    df_hourly['dayofweek'] = df_hourly.index.dayofweek
    df_hourly['month'] = df_hourly.index.month
    df_hourly['is_weekend'] = (df_hourly.index.dayofweek >= 5).astype(int)

    # --- 3. Seasonal Imputer Logic ---
    seasonal_map = df_hourly.groupby(['dayofweek', 'hour'])['Aggregate'].transform('mean')
    df_hourly['Aggregate'].fillna(seasonal_map, inplace=True)
    
    # Final cleanup
    df_hourly.fillna(method='ffill', inplace=True)
    df_hourly.fillna(method='bfill', inplace=True)

    # 4. Determine Required History
    LAG_PERIOD = 168 
    history_start = pd.to_datetime(start_date) - pd.Timedelta(hours=LAG_PERIOD)
    
    # Ensure we select the full range needed
    df_full_context = df_hourly.loc[history_start:end_date].copy()
    
    if df_full_context.empty:
        # Fallback if specific dates aren't found (e.g. testing with small data)
        df_full_context = df_hourly.copy()

    # 5. Add Lag Features
    df_full_context['lag_1hr'] = df_full_context['Aggregate'].shift(1)
    df_full_context['lag_24hr'] = df_full_context['Aggregate'].shift(24)
    df_full_context['lag_168hr'] = df_full_context['Aggregate'].shift(168)
    df_full_context.dropna(inplace=True)
    
    # 6. Apply Scaler (Transform ONLY)
    scale_cols = ['Aggregate'] + [f'Appliance{i}' for i in range(1,10)]
    
    # Ensure valid columns exist before scaling
    valid_scale_cols = [c for c in scale_cols if c in df_full_context.columns]
    df_full_context[valid_scale_cols] = scaler_obj.transform(df_full_context[valid_scale_cols])
    
    # 7. Select Final Features
    final_features = valid_scale_cols + ['hour', 'dayofweek', 'month', 'is_weekend', 'lag_1hr', 'lag_24hr', 'lag_168hr']
    
    # Return the slice for the requested period
    df_final_input = df_full_context[final_features]
    df_validation_slice = df_final_input.loc[start_date:end_date]
    
    return df_final_input, df_validation_slice

def inverse_scale_predictions(predictions_scaled, scaler_obj):
    """Helper to convert prediction array back to original units."""
    n_scaler_features = scaler_obj.n_features_in_
    dummy_array = np.zeros((len(predictions_scaled), n_scaler_features))
    dummy_array[:, 0] = predictions_scaled
    return scaler_obj.inverse_transform(dummy_array)[:, 0]