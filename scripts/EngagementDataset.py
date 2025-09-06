from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import ast
import torch

CACHE_PATH = Path(__file__).parent.parent / Path("data/processed/engagement_dataset_cache.npz")
TARGET_COLUMNS = ["viewCount", "likeCount"]

# --- Dataset Class ---
class EngagementDataset(Dataset):
    def __init__(self, dataframe):
        emotion_columns = [
            'admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire',
            'disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy',
            'love','nervousness','optimism','pride','realization','relief','remorse','sadness','surprise','neutral','change_points'
        ]

        if CACHE_PATH.exists():
            cache = np.load(CACHE_PATH)
            self.X_seq = cache['X_seq']
            self.X_static = cache['X_static']
            self.y = cache['y']
            return

        static_features = dataframe.drop(columns=["video_id"] + TARGET_COLUMNS + emotion_columns)
        emotion_sequences = []

        for _, row in dataframe.iterrows():
            emotion_vector = []
            for col in emotion_columns:
                parsed = ast.literal_eval(row[col])
                emotion_vector.append(parsed)
            emotion_vector = np.array(emotion_vector).T
            emotion_sequences.append(emotion_vector)

        self.X_seq = np.array(emotion_sequences).astype(np.float32)
        self.X_static = static_features.values.astype(np.float32)
        self.y = np.log1p(dataframe[TARGET_COLUMNS].values.astype(np.float32))

        np.savez(CACHE_PATH, X_seq=self.X_seq, X_static=self.X_static, y=self.y)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_seq[idx]),
            torch.tensor(self.X_static[idx]),
            torch.tensor(self.y[idx])
        )
