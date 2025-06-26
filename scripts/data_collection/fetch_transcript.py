import os
import re
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from youtube_transcript_api.proxies import WebshareProxyConfig
from multiprocessing import Pool

# Paths
METADATA_FILE = Path(__file__).parent.parent.parent / "data/raw/metadata.csv"
TRANSCRIPT_DIR = Path(__file__).parent.parent.parent / "data/raw/transcripts"
FAILED_LOG = Path(__file__).parent.parent.parent / "data/raw/failed_transcripts.json"
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# Load metadata and get video IDs
metadata_df = pd.read_csv(METADATA_FILE)
existing_ids = [f.stem for f in TRANSCRIPT_DIR.glob("*.csv")]

# Load previously failed logs (if exists)
if FAILED_LOG.exists():
    with open(FAILED_LOG, 'r') as f:
        failed_log = json.load(f)
else:
    failed_log = {}

# Extract failed video IDs
failed_ids = list(failed_log.keys())

# Exclude IDs that are in existing_ids or failed_ids
video_ids = [vid for vid in metadata_df["videoId"].dropna().unique().tolist()
             if vid not in existing_ids and vid not in failed_ids]

# Initialize YouTubeTranscriptApi with WebshareProxyConfig
ytt_api = YouTubeTranscriptApi(
    # proxy_config=WebshareProxyConfig(
    #     proxy_username="",
    #     proxy_password="",
    # )
)

# --- Text Cleaning Function ---
def clean_transcript_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-z0-9.,!?\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Worker Function ---
def process_video(video_id):
    output_file = TRANSCRIPT_DIR / f"{video_id}.csv"
    if output_file.exists():
        return None  # already done

    try:
        transcript = ytt_api.fetch(video_id)
        df = pd.DataFrame(transcript)
        df['clean_text'] = df['text'].apply(clean_transcript_text)
        df = df[['start', 'duration', 'text', 'clean_text']]
        df.to_csv(output_file, index=False)
        return None
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        return (video_id, str(e))
    except Exception as e:
        msg = str(e)
        if "YouTube is blocking requests from your IP" in msg:
            raise RuntimeError(f"IP-related error encountered: {msg}")
        return (video_id, f"Unexpected error: {msg}")

# --- Safe Process Wrapper ---
def safe_process(video_id):
    try:
        return process_video(video_id)
    except RuntimeError as e:
        return "IP_BLOCK", str(e)

if __name__ == "__main__":
    from multiprocessing import Pool

    failed_log = {}
    save_counter = 0

    def safe_process(video_id):
        try:
            return process_video(video_id)
        except RuntimeError as e:
            return "IP_BLOCK", str(e)

    with Pool(12) as pool:
        try:
            for result in tqdm(pool.imap_unordered(safe_process, video_ids), total=len(video_ids), desc="Fetching transcripts"):
                if isinstance(result, tuple):
                    vid, reason = result
                    if vid == "IP_BLOCK":
                        print(reason)
                        pool.terminate()
                        break
                    failed_log[vid] = reason

                save_counter += 1
                if save_counter % 100 == 0:
                    with open(FAILED_LOG, 'w') as f:
                        json.dump(failed_log, f, indent=2)

        except Exception as e:
            print(f"Unexpected error: {e}")

        finally:
            with open(FAILED_LOG, 'w') as f:
                json.dump(failed_log, f, indent=2)

            print(f"\nFinished fetching transcripts. {len(failed_log)} failures logged.")
