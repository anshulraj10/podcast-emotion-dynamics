from pathlib import Path


BATCH_FILE = Path(__file__).parent.parent / "active_batch.txt"
OUTPUT_CSV = Path(__file__).parent.parent.parent / Path("data/raw/metadata.csv")

print(OUTPUT_CSV)