import os
import isodate
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise ValueError("Missing YOUTUBE_API_KEY in .env file")

# Initialize YouTube API client
YOUTUBE = build("youtube", "v3", developerKey=API_KEY)

# Configurations
PUBLISHED_AFTER = "2024-01-01T00:00:00Z"
MIN_DURATION = 1800  # 30 minutes
MAX_DURATION = 7200  # 120 minutes
REGION_CODE = "US"
RESULTS_PER_QUERY = 250

# File paths
BATCH_FILE = Path(__file__).parent / "active_batch.txt"
TRACK_FILE = Path(__file__).parent / "query_tracker.json"
OUTPUT_CSV = Path(__file__).parent.parent.parent / Path("data/raw/metadata.csv")

# Load current batch of queries
with open(BATCH_FILE, 'r') as f:
    queries = [line.strip() for line in f if line.strip()]

def search_video_ids(query, max_results=RESULTS_PER_QUERY):
    video_ids = []
    next_page_token = None

    while len(video_ids) < max_results:
        request = YOUTUBE.search().list(
            q=query,
            part="id",
            type="video",
            maxResults=50,
            publishedAfter=PUBLISHED_AFTER,
            regionCode=REGION_CODE,
            videoDuration="long",
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response.get("items", []):
            video_ids.append(item["id"]["videoId"])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return video_ids[:max_results]

def fetch_channel_metadata(channel_ids):
    channel_data = []
    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i:i+50]
        request = YOUTUBE.channels().list(
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
    return pd.DataFrame(channel_data)

def fetch_video_metadata(video_ids):
    all_data = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        request = YOUTUBE.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(batch)
        )
        response = request.execute()

        for item in response.get("items", []):
            try:
                duration = isodate.parse_duration(item["contentDetails"]["duration"]).total_seconds()
                if MIN_DURATION <= duration <= MAX_DURATION:
                    snippet = item["snippet"]
                    stats = item.get("statistics", {})

                    all_data.append({
                        "videoId": item["id"],
                        "videoUrl": f"https://www.youtube.com/watch?v={item['id']}",
                        "title": snippet.get("title"),
                        "description": snippet.get("description", ""),
                        "publishDate": snippet.get("publishedAt"),
                        "channelTitle": snippet.get("channelTitle"),
                        "channelId": snippet.get("channelId"),
                        "tags": snippet.get("tags", []),
                        "categoryId": snippet.get("categoryId", None),
                        "liveBroadcastContent": snippet.get("liveBroadcastContent", ""),
                        "defaultAudioLanguage": snippet.get("defaultAudioLanguage", ""),
                        "duration_sec": int(duration),
                        "viewCount": int(stats.get("viewCount", 0)),
                        "likeCount": int(stats.get("likeCount", 0)),
                        "commentCount": int(stats.get("commentCount", 0)),
                    })
            except Exception:
                continue

    video_df = pd.DataFrame(all_data)

    # Enrich with channel metadata
    if not video_df.empty:
        unique_channels = video_df["channelId"].dropna().unique().tolist()
        channel_df = fetch_channel_metadata(unique_channels)
        video_df = video_df.merge(channel_df, on="channelId", how="left")

    return video_df

def main():
    all_video_ids = set()

    for query in tqdm(queries, desc="Running queries"):
        ids = search_video_ids(query)
        all_video_ids.update(ids)

    print(f"Collected {len(all_video_ids)} unique video IDs")

    metadata_df = fetch_video_metadata(list(all_video_ids))
    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)

    if OUTPUT_CSV.exists():
        existing_df = pd.read_csv(OUTPUT_CSV)
        combined_df = pd.concat([existing_df, metadata_df]).drop_duplicates(subset=["videoId"])
    else:
        combined_df = metadata_df

    combined_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved metadata for {len(metadata_df)} new videos to {OUTPUT_CSV}")

    # Update tracker
    if TRACK_FILE.exists():
        with open(TRACK_FILE, 'r') as f:
            tracker = json.load(f)
    else:
        tracker = {"completed": []}

    tracker["completed"].extend(queries)
    tracker["completed"] = list(set(tracker["completed"]))

    with open(TRACK_FILE, 'w') as f:
        json.dump(tracker, f, indent=2)

if __name__ == "__main__":
    main()
