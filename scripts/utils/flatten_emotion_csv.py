import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
INPUT_DIR = Path(__file__).parent.parent.parent / Path("data/processed/emotions_flat")
OUTPUT_FILE = Path(__file__).parent.parent.parent / Path("data/processed/all_emotions.csv")

# --- Merge ---
all_files = list(INPUT_DIR.glob("*.csv"))
df_list = []

for file in tqdm(all_files, desc="Merging emotion files"):
    df = pd.read_csv(file)
    video_id = file.stem.replace('_emotions', '')
    df['video_id'] = video_id
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)
combined_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nCombined CSV saved to: {OUTPUT_FILE}")
