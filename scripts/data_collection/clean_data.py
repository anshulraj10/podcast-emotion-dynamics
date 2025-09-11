# Clean metadata by removing entries with missing or invalid transcript
import pandas as pd
from pathlib import Path
import os

# Paths
METADATA_FILE = Path(__file__).parent.parent.parent / "data/raw/metadata.csv"
TRANSCRIPT_DIR = Path(__file__).parent.parent.parent / "data/raw/transcripts"

# Load metadata
df = pd.read_csv(METADATA_FILE)

# Step 1: Remove rows where corresponding transcript is missing
existing_transcripts = {f.stem for f in TRANSCRIPT_DIR.glob("*.csv")}
df = df[df["videoId"].isin(existing_transcripts)]

# Step 2: Remove rows with nulls in critical columns and delete associated transcripts
required_columns = [
    "videoId", "viewCount", "likeCount", "commentCount",
    "channelViewCount", "subscriberCount", "channelVideoCount", "duration_sec"
]
before = len(df)
df_clean = df.dropna(subset=required_columns)
removed = df[~df.index.isin(df_clean.index)]

# Delete corresponding transcript files for removed rows
for vid in removed["videoId"]:
    transcript_path = TRANSCRIPT_DIR / f"{vid}.csv"
    if transcript_path.exists():
        os.remove(transcript_path)

# Save cleaned metadata
df_clean.to_csv(METADATA_FILE, index=False)
print(f"Cleaned metadata saved. Rows before: {before}, after: {len(df_clean)}")
