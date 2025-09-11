# scripts/EngagementDataset.py
from pathlib import Path
import numpy as np
import ast
import pickle
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).parent.parent
CACHE_PATH = ROOT / "data/processed/engagement_dataset_cache.npz"
TARGET_COLUMNS = ["viewCount", "likeCount"]
TARGET_SCALER_PATH = ROOT / "models/target_scaler.pkl"

class EngagementDataset(Dataset):
    def __init__(self, dataframe):
        emotion_columns = [
            'admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire',
            'disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy',
            'love','nervousness','optimism','pride','realization','relief','remorse','sadness','surprise','neutral','change_points'
        ]

        # Use cache if present
        if CACHE_PATH.exists():
            cache = np.load(CACHE_PATH)
            self.X_seq = cache['X_seq']
            self.X_static = cache['X_static']
            self.y = cache['y']
            return

        # 1) Build emotion sequences (T x E)
        emotion_sequences = []
        for _, row in dataframe.iterrows():
            vecs = []
            for col in emotion_columns:
                parsed = ast.literal_eval(row[col])
                vecs.append(parsed)
            emotion_sequences.append(np.array(vecs).T)  # (T, E)

        self.X_seq = np.array(emotion_sequences, dtype=np.float32)

        # 2) Build static features: drop id/targets/emotion cols, then keep numeric only
        cols_to_drop = ["video_id"] + TARGET_COLUMNS + emotion_columns
        static_df = dataframe.drop(columns=cols_to_drop, errors="ignore")

        # keep numeric only (drops channelId and any stray strings)
        static_df = static_df.select_dtypes(include=[np.number]).copy()

        # safety: fill NaNs
        static_df = static_df.fillna(0.0)

        self.X_static = static_df.values.astype(np.float32)

        # 3) Targets: scaled-log (log1p -> RobustScaler)
        with open(TARGET_SCALER_PATH, "rb") as f:
            target_scaler = pickle.load(f)
        log_y = np.log1p(dataframe[TARGET_COLUMNS].values.astype(np.float32))
        self.y = target_scaler.transform(log_y).astype(np.float32)

        # Cache
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE_PATH, X_seq=self.X_seq, X_static=self.X_static, y=self.y)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_seq[idx]),
            torch.tensor(self.X_static[idx]),
            torch.tensor(self.y[idx])
        )
