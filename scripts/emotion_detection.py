from multiprocessing import Pool
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURATION ---
TRANSCRIPT_DIR = Path(__file__).parent.parent / Path("data/raw/transcripts")
OUTPUT_DIR = Path(__file__).parent.parent / Path("data/processed/emotions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Segmentation params
MAX_DURATION = 45.0
MIN_DURATION = 15.0
MIN_WORDS = 8

# Load model and tokenizer
MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)

id2label = model.config.id2label
threshold = 0.3

# --- FUNCTIONS ---
def segment_transcript(df):
    segments = []
    current_text = ""
    segment_start = df.iloc[0]["start"]
    segment_end = df.iloc[0]["start"]

    for i, row in df.iterrows():
        line = str(row["clean_text"]).strip()
        if not line:
            continue

        segment_end = row["start"] + row["duration"]
        current_text += " " + line
        current_duration = segment_end - segment_start

        should_end = (
            line.endswith(('.', '!', '?')) and
            (current_duration >= MIN_DURATION) and
            (len(current_text.split()) >= MIN_WORDS)
        )

        if should_end or current_duration >= MAX_DURATION:
            segments.append({
                "start": segment_start,
                "end": segment_end,
                "text": current_text.strip()
            })
            current_text = ""
            if i + 1 < len(df):
                segment_start = df.iloc[i + 1]["start"]

    if current_text.strip():
        segments.append({
            "start": segment_start,
            "end": segment_end,
            "text": current_text.strip()
        })

    return pd.DataFrame(segments)

def predict_emotions(texts):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        labels = [id2label[i] for i, p in enumerate(probs) if p >= threshold]
        results.append((labels, probs.tolist()))
    return results

# --- PROCESSING LOOP ---
def process_file(file):
    try:
        video_id = file.stem
        output_file = OUTPUT_DIR / f"{video_id}_emotions.csv"
        if output_file.exists():
            return

        df = pd.read_csv(file)
        segmented_df = segment_transcript(df)
        preds = predict_emotions(segmented_df["text"])

        segmented_df["emotions"] = [p[0] for p in preds]
        segmented_df["scores"] = [p[1] for p in preds]
        segmented_df["time"] = (segmented_df["start"] + segmented_df["end"]) / 2

        segmented_df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error processing {file.name}: {e}")

# --- PARALLEL EXECUTION ---
if __name__ == "__main__":
    transcript_files = list(TRANSCRIPT_DIR.glob("*.csv"))
    with Pool(6) as pool:
        list(tqdm(pool.imap_unordered(process_file, transcript_files), total=len(transcript_files), desc="Processing transcripts"))

    print("\nParallel batch emotion detection completed.")
