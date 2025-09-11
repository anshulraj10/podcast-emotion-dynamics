import ast
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# ---------- Paths ----------
ROOT = Path(__file__).parent.parent
EMOTIONS_FILE = ROOT / "data/processed/emotions.csv"
CPS_FILE      = ROOT / "data/processed/emotion_change_points.csv"
METADATA_FILE = ROOT / "data/raw/metadata.csv"
OUTPUT_FILE   = ROOT / "data/processed/dataset.csv"

FEATURE_SCALER_PATH = ROOT / "models/feature_scaler.pkl"
TARGET_SCALER_PATH  = ROOT / "models/target_scaler.pkl"

# ---------- Helpers ----------
def change_point_mask(change_points_str, length: int = 100):
    cps = ast.literal_eval(change_points_str)
    mask = np.zeros(length, dtype=np.int8)
    for idx in cps:
        if 0 <= idx < length:
            mask[idx] = 1
    return mask.tolist()

def pick_col(df: pd.DataFrame, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of the candidate columns found: {candidates}. Available: {list(df.columns)}")
    return None

if __name__ == "__main__":
    # ----- Load inputs -----
    emotions_df = pd.read_csv(EMOTIONS_FILE)
    change_points_df = pd.read_csv(CPS_FILE)
    metadata_df = pd.read_csv(METADATA_FILE)

    # Resolve key columns (supports snake_case or camelCase)
    def has(name): return name in metadata_df.columns
    video_id_col    = pick_col(metadata_df, [c for c in ["video_id", "videoId"] if has(c)])
    channel_id_col  = pick_col(metadata_df, [c for c in ["channel_id", "channelId"] if has(c)])
    publish_col     = pick_col(metadata_df, [c for c in ["publish_date", "publishDate"] if has(c)])
    duration_col    = pick_col(metadata_df, [c for c in ["duration_sec", "durationSec", "duration_seconds"] if has(c)])
    view_count_col  = pick_col(metadata_df, [c for c in ["view_count", "viewCount", "views"] if has(c)])
    like_count_col  = pick_col(metadata_df, [c for c in ["like_count", "likeCount", "likes"] if has(c)])
    ch_view_col     = pick_col(metadata_df, [c for c in ["channel_view_count", "channelViewCount"] if has(c)], required=False)
    sub_count_col   = pick_col(metadata_df, [c for c in ["subscriber_count", "subscriberCount"] if has(c)], required=False)
    ch_video_col    = pick_col(metadata_df, [c for c in ["channel_video_count", "channelVideoCount"] if has(c)], required=False)

    # ----- Change point features -----
    change_points_df["num_change_points"] = change_points_df["change_points"].apply(
        lambda s: len(ast.literal_eval(s))
    )
    change_points_df["change_points"] = change_points_df["change_points"].apply(change_point_mask)

    # ----- Select & standardize metadata -----
    keep_cols = [video_id_col, channel_id_col, publish_col, duration_col, view_count_col, like_count_col]
    if ch_view_col:   keep_cols.append(ch_view_col)
    if sub_count_col: keep_cols.append(sub_count_col)
    if ch_video_col:  keep_cols.append(ch_video_col)

    md = metadata_df[keep_cols].copy()

    rename_map = {
        video_id_col:   "video_id",
        channel_id_col: "channelId",
        publish_col:    "publishDate",
        duration_col:   "duration_sec",
        view_count_col: "viewCount",
        like_count_col: "likeCount",
    }
    if ch_view_col:   rename_map[ch_view_col]   = "channelViewCount"
    if sub_count_col: rename_map[sub_count_col] = "subscriberCount"
    if ch_video_col:  rename_map[ch_video_col]  = "channelVideoCount"
    md.rename(columns=rename_map, inplace=True)

    # Ensure optional cols exist
    for opt in ["channelViewCount", "subscriberCount", "channelVideoCount"]:
        if opt not in md.columns:
            md[opt] = 0

    # ----- Dates & age features -----
    md["publishDate"] = pd.to_datetime(md["publishDate"], errors="coerce").dt.tz_localize(None)
    reference_date = datetime(2025, 7, 1)
    md["days_published"] = (reference_date - md["publishDate"]).dt.days
    md.drop(columns=["publishDate"], inplace=True)
    md["days_published"] = md["days_published"].fillna(1).clip(lower=1)

    md["log_age"] = np.log1p(md["days_published"])
    md["views_per_day"] = md["viewCount"] / md["days_published"]
    md["likes_per_day"] = md["likeCount"]  / md["days_published"]
    md["engagement_rate"] = md["likeCount"] / np.clip(md["viewCount"], 1, None)

    # ----- Channel baseline controls (leave-one-out) -----
    grp = md.groupby("channelId", dropna=False)
    cnt = grp["video_id"].transform("count").astype(float)

    sum_views = grp["viewCount"].transform("sum")
    sum_likes = grp["likeCount"].transform("sum")

    md["__er__"] = md["likeCount"] / np.clip(md["viewCount"], 1, None)
    sum_er = grp["__er__"].transform("sum")

    denom = np.clip(cnt - 1.0, 1.0, None)
    md["channel_avg_views_loo"] = ((sum_views - md["viewCount"]) / denom).fillna(0)
    md["channel_avg_likes_loo"] = ((sum_likes - md["likeCount"]) / denom).fillna(0)
    md["channel_avg_er_loo"]    = ((sum_er   - md["__er__"])   / denom).fillna(0)
    md.drop(columns="__er__", inplace=True)

    # ----- RELATIVE-TO-CHANNEL features (now that LOO cols exist) -----
    # Safe fallbacks if any LOO is zero -> use 1 to avoid divide-by-zero
    md["views_per_day_rel"] = md["views_per_day"] / np.clip(md["channel_avg_views_loo"], 1, None)
    md["likes_per_day_rel"] = md["likes_per_day"] / np.clip(md["channel_avg_likes_loo"], 1, None)

    # ----- Scale features; save scalers -----
    feature_columns_to_scale = [
        "channelViewCount", "subscriberCount", "channelVideoCount",
        "views_per_day", "likes_per_day", "engagement_rate",
        "channel_avg_views_loo", "channel_avg_likes_loo", "channel_avg_er_loo",
        "days_published", "log_age",
        "views_per_day_rel", "likes_per_day_rel"
    ]
    md[feature_columns_to_scale] = md[feature_columns_to_scale].fillna(0)

    feature_scaler = RobustScaler()
    feature_scaler.fit(md[feature_columns_to_scale])
    FEATURE_SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_SCALER_PATH, "wb") as f:
        pickle.dump(feature_scaler, f)

    md[feature_columns_to_scale] = feature_scaler.transform(md[feature_columns_to_scale])

    # ----- Target scaler (fit on log targets) -----
    target_columns = ["viewCount", "likeCount"]
    log_targets = np.log1p(md[target_columns].values.astype(np.float32))
    target_scaler = RobustScaler()
    target_scaler.fit(log_targets)
    with open(TARGET_SCALER_PATH, "wb") as f:
        pickle.dump(target_scaler, f)

    # ----- Merge & Save -----
    df = emotions_df.merge(change_points_df, on="video_id", how="inner")
    df = df.merge(md, on="video_id", how="inner")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved processed dataset → {OUTPUT_FILE} (rows={len(df)})")
