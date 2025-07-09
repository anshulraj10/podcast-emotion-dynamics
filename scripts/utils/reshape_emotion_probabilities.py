# Script to reshape emotion columns in per-video CSVs (parallelized)
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count
from transformers import AutoModelForSequenceClassification

# --- CONFIG ---
INPUT_DIR = Path(__file__).parent.parent.parent / Path("data/processed/emotions")
OUTPUT_DIR = Path(__file__).parent.parent.parent / Path("data/processed/emotions_flat")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load model label mapping ---
model = AutoModelForSequenceClassification.from_pretrained("joeddav/distilbert-base-uncased-go-emotions-student")
id2label = model.config.id2label
label_columns = [id2label[i] for i in range(len(id2label))]

# --- PROCESSING FUNCTION ---
def reshape_file(file_path):
    try:
        df = pd.read_csv(file_path)
        all_scores = df['scores'].apply(eval).tolist()
        probs_df = pd.DataFrame(all_scores, columns=label_columns)

        df.drop(columns=["emotions", "scores"], inplace=True)
        df = pd.concat([df, probs_df], axis=1)

        output_file = OUTPUT_DIR / file_path.name
        df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")

# --- PARALLEL EXECUTION ---
if __name__ == "__main__":
    files = list(INPUT_DIR.glob("*.csv"))
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(reshape_file, files), total=len(files), desc="Reshaping emotion CSVs"))

    print("\nReshape complete. Output saved to:", OUTPUT_DIR)
