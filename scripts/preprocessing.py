import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import ast
from sklearn.preprocessing import RobustScaler
import pickle

EMOTIONS_FILE = INPUT_DIR = Path(__file__).parent.parent / Path("data/processed/emotions.csv")
CPS_FILE = INPUT_DIR = Path(__file__).parent.parent / Path("data/processed/emotion_change_points.csv")
METADATA_FILE = INPUT_DIR = Path(__file__).parent.parent / Path("data/raw/metadata.csv")
OUTPUT_FILE = INPUT_DIR = Path(__file__).parent.parent / Path("data/processed/dataset.csv")

def change_point_mask(change_points, length=100):
    """
    Converts change point indices to a fixed length binary list

    Args:
        change_points (list): list of change points indices

    Returns:
        list: fixed length binary list
    """
    change_points = ast.literal_eval(change_points)
    mask = np.zeros(length, dtype=np.int8)
    for idx in change_points:
        if 0 <= idx < length:
            mask[idx] = 1
    return mask.tolist()

if __name__ == "__main__":
    emotions_df = pd.read_csv(EMOTIONS_FILE)
    change_points_df = pd.read_csv(CPS_FILE)
    metadata_df = pd.read_csv(METADATA_FILE)
    
    # Add no of change points detected
    change_points_df['num_change_points'] = change_points_df['change_points'].apply(len)
    change_points_df['change_points'] = change_points_df['change_points'].apply(change_point_mask)
    
    # --- Process Metadata ---
    # Keep only selected columns
    metadata_df = metadata_df[[
        'video_id', 'publishDate', 'duration_sec', 'viewCount', 'likeCount',
        'channelViewCount', 'subscriberCount', 'channelVideoCount']]

    # Compute days since published (from 2025-07-01)
    metadata_df['publishDate'] = pd.to_datetime(metadata_df['publishDate'], errors='coerce').dt.tz_localize(None)
    reference_date = datetime(2025, 7, 1)
    metadata_df['days_published'] = (reference_date - metadata_df['publishDate']).dt.days
    metadata_df.drop(columns=['publishDate'], inplace=True)
    
    feature_columns_to_scale = ['channelViewCount', 'subscriberCount', 'channelVideoCount']
    target_columns_to_scale = ['viewCount', 'likeCount']
    
    feature_scaler = RobustScaler()
    feature_scaler.fit(metadata_df[feature_columns_to_scale])
    with open('../models/feature_scaler.pkl', 'wb') as file:
        pickle.dump(feature_scaler, file) 

    target_scaler = RobustScaler()
    target_scaler.fit(metadata_df[target_columns_to_scale])
    with open('../models/target_scaler.pkl', 'wb') as file:
        pickle.dump(target_scaler, file)
        
    metadata_df[feature_columns_to_scale] = feature_scaler.transform(metadata_df[feature_columns_to_scale])
    metadata_df[target_columns_to_scale] = target_scaler.transform(metadata_df[target_columns_to_scale])

    # --- Merge all dataframes on video_id ---
    df = emotions_df.merge(change_points_df, on="video_id", how="inner")
    df = df.merge(metadata_df, on="video_id", how="inner")