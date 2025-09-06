import pandas as pd
import numpy as np
import ruptures as rpt
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import RobustScaler

# --- CONFIG ---
INPUT_FILE = Path(__file__).parent.parent / Path("data/processed/emotions.csv")
OUTPUT_FILE =Path(__file__).parent.parent / Path("data/processed/emotion_change_points.csv")
EMOTION_COLUMNS = [
    'admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire',
    'disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy',
    'love','nervousness','optimism','pride','realization','relief','remorse','sadness','surprise','neutral'
]
    
# --- Change Point Detection Function ---
def detect_change_points_multivariate(emotion_matrix, model="rbf", pen=10):
    """
    Detect change points in multivariate emotion time series data using the PELT algorithm.

    Args:
        emotion_matrix (array-like, shape (n_timepoints, n_emotions)): Matrix containing emotion values over time
        model (str, optional): The cost function model to use for change point detection. Defaults to "rbf".
        pen (int, optional): Penalty parameter that controls the trade-off between model fit and number of 
        change points. Defaults to 10.

    Returns:
        list: List of change point indices (time steps)
    """
    emotion_matrix = RobustScaler().fit_transform(emotion_matrix)
    algo = rpt.Pelt(model=model, jump=2).fit(emotion_matrix)
    cps = algo.predict(pen=pen)
    # Filter short segments
    filtered_cps = [cps[0]] if cps else []
    for i in range(1, len(cps)):
        if cps[i] - cps[i-1] > 5:
            filtered_cps.append(cps[i])
    return filtered_cps

def detect_change_point(emotions_df):
    """
    Detect change points for multiple videos in an emotion dataset

    Args:
        emotions_df (pandas.DataFrame): DataFrame containing emotion data

    Returns:
        list: List of dictionaries containing the change points
    """
    change_points_list = []

    for _, row in tqdm(emotions_df.iterrows(), total=len(emotions_df), desc="Detecting change points"):
        video_id = row['video_id']
        emotion_matrix = np.stack([row[emo] for emo in EMOTION_COLUMNS], axis=1)  # shape: (100, 28)
        try:
            cps = detect_change_points_multivariate(emotion_matrix, model="rbf", pen=1)
            change_points_list.append({"video_id": video_id, "change_points": cps})
        except Exception as e:
            print(f"Error for video_id {video_id}: {e}")
            change_points_list.append({"video_id": video_id, "change_points": []})
            
    return change_points_list

if __name__ == "__main__":
    # --- Load Data ---
    emotions_df = pd.read_csv(INPUT_FILE)
    
    # --- Preprocess List Columns ---
    for emo in EMOTION_COLUMNS:
        emotions_df[emo] = emotions_df[emo].apply(lambda x: np.array(eval(x)))
    
    change_points_list = detect_change_point(emotions_df)
    change_points_emotions_df = pd.DataFrame(change_points_list)

    change_points_emotions_df.to_csv(OUTPUT_FILE, index=False)