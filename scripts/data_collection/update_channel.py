import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from googleapiclient.discovery import build

# Load API key
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

# Load original metadata
METADATA_CSV = Path(__file__).parent.parent.parent / Path("data/raw/metadata.csv")
df = pd.read_csv(METADATA_CSV)

# Get unique channel IDs
channel_ids = df["channelId"].dropna().unique().tolist()

# Fetch in batches of 50
channel_data = []

for i in tqdm(range(0, len(channel_ids), 50), desc="Fetching channel stats"):
    batch = channel_ids[i:i+50]
    request = youtube.channels().list(
        part="statistics",
        id=",".join(batch)
    )
    response = request.execute()

    for item in response.get("items", []):
        stats = item["statistics"]
        channel_data.append({
            "channelId": item["id"],
            "channelViewCount": int(stats.get("viewCount", 0)),
            "subscriberCount": int(stats.get("subscriberCount", 0)) if not stats.get("hiddenSubscriberCount") else None,
            "hiddenSubscriberCount": stats.get("hiddenSubscriberCount", "false") == "true",
            "channelVideoCount": int(stats.get("videoCount", 0))
        })

# Convert and merge
channel_df = pd.DataFrame(channel_data)
merged_df = df.merge(channel_df, on="channelId", how="left")

# Save to new CSV
merged_df.to_csv(METADATA_CSV, index=False)
print("Updated metadata saved to data/raw/metadata_with_channels.csv")
