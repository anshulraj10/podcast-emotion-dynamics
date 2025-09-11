import json
from pathlib import Path

# Configuration
BATCH_SIZE = 13
BATCH_FILE = Path(__file__).parent / "active_batch.txt"
TRACK_FILE = Path(__file__).parent / "query_tracker.json"
QUERY_FILE = Path(__file__).parent / "search_queries.txt"

# Load all queries
with open(QUERY_FILE, 'r') as f:
    all_queries = [q.strip() for q in f.readlines() if q.strip()]

# Initialize tracker file if it doesn't exist
if not TRACK_FILE.exists():
    with open(TRACK_FILE, 'w') as f:
        json.dump({"completed": []}, f)

# Load completed queries
with open(TRACK_FILE, 'r') as f:
    completed = set(json.load(f).get("completed", []))

# Filter remaining queries
remaining = [q for q in all_queries if q not in completed]

# Select next batch
next_batch = remaining[:BATCH_SIZE]

# Output to temporary batch fileBATCH_FILE
BATCH_FILE.write_text("\n".join(next_batch))

print(f"Prepared batch of {len(next_batch)} queries.")
print(f"Remaining after this batch: {len(remaining) - len(next_batch)}")
