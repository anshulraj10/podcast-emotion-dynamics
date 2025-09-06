import pickle
from matplotlib import pyplot as plt
import torch
from random import randint
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from EngagementModel import EngagementModel
from EngagementDataset import EngagementDataset
from torch.serialization import add_safe_globals
from torch.utils.data.dataset import Subset
from utils import save_scatter_graph, save_graph

# --- CONFIG ---
add_safe_globals([Subset])
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = Path(__file__).parent.parent / Path("models/engagement_model.pt")
TARGET_SCALER_PATH = Path(__file__).parent.parent / Path("models/target_scaler.pkl")
TEST_DATASET_PATH = Path(__file__).parent.parent / Path("data/processed/test_dataset.pt")
IMAGES_DIR = Path(__file__).parent.parent / Path("images")
BATCH_SIZE = 128

# Set the model architecture args (must match training script)
SEQ_INPUT_DIM = 29  # adjust based on dataset shape
STATIC_INPUT_DIM = 6  # adjust based on your static features
HIDDEN_SIZE = 128
TRANSFORMER_HEADS = 4
DROPOUT = 0.15

model = EngagementModel(seq_input_dim=SEQ_INPUT_DIM, static_input_dim=STATIC_INPUT_DIM,
                        hidden_size=HIDDEN_SIZE, transformer_heads=TRANSFORMER_HEADS, dropout=DROPOUT).to(DEVICE)

def inverse_transform_and_clip(preds, scaler, percentile_bounds=(5, 95)):
    preds_np = np.expm1(preds)
    preds_unscaled = scaler.inverse_transform(preds_np)
    clipped = np.clip(
        preds_unscaled,
        np.percentile(preds_unscaled, percentile_bounds[0], axis=0),
        np.percentile(preds_unscaled, percentile_bounds[1], axis=0)
    )
    return clipped

# --- Manual Testing ---
with torch.no_grad():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    test_ds = torch.load(TEST_DATASET_PATH, weights_only=False)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Initialize lists to store all predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    
    # Process all batches in the test dataloader
    for xb_seq, xb_static, yb in test_dl:
        out = model(xb_seq.to(DEVICE), xb_static.to(DEVICE))
        pred = torch.cat([out['views_mean'], out['likes_mean']], dim=1).cpu().numpy()
        y_true = yb.numpy()
        
        all_predictions.append(pred)
        all_ground_truth.append(y_true)
    
# Concatenate all batches
all_predictions = np.concatenate(all_predictions, axis=0)
all_ground_truth = np.concatenate(all_ground_truth, axis=0)

# Load scaler and inverse transform
with open(TARGET_SCALER_PATH, 'rb') as file:
    loaded_scaler = pickle.load(file)

pred_unscaled = inverse_transform_and_clip(all_predictions, loaded_scaler)
y_unscaled = loaded_scaler.inverse_transform(np.expm1(all_ground_truth))

# Calculate metrics on entire test set
rmse = np.sqrt(np.mean((pred_unscaled - y_unscaled) ** 2, axis=0))
mae = np.mean(np.abs(pred_unscaled - y_unscaled), axis=0)

print(f"Total test samples: {len(all_predictions)}")
print(f"ViewCount  →  RMSE = {rmse[0]:,.2f}  |  MAE = {mae[0]:,.2f}")
print(f"LikeCount  →  RMSE = {rmse[1]:,.2f}  |  MAE = {mae[1]:,.2f}")

# for idx in [randint(1, len(pred_unscaled)-1) for _ in range(10)]:
#     print(f"Predicted: Views={pred_unscaled[idx][0]:,.0f}, Likes={pred_unscaled[idx][1]:,.0f} | Actual: Views={y_unscaled[idx][0]:,.0f}, Likes={y_unscaled[idx][1]:,.0f}")
    

save_scatter_graph(
    filename="views_scatter_predicted_vs_actual",
    actual=y_unscaled[:, 0],
    predicted=pred_unscaled[:, 0],
    x_label="Actual Views",
    y_label="Predicted Views",
    title="Views: Predicted vs Actual (Scatter Plot)"
)

save_scatter_graph(
    filename="likes_scatter_predicted_vs_actual",
    actual=y_unscaled[:, 1],
    predicted=pred_unscaled[:, 1],
    x_label="Actual Likes",
    y_label="Predicted Likes",
    title="Likes: Predicted vs Actual (Scatter Plot)"
)

views_data = [
    (y_unscaled[:, 0], "Actual Views"),
    (pred_unscaled[:, 0], "Predicted Views")
]

save_graph(
    filename="views_predicted_vs_actual",
    data=views_data,
    x_label="Sample Index",
    y_label="View Count",
    title="Predicted vs Actual Views",
    figsize=(12, 8)
)

# Create predicted vs actual line graph for Likes
likes_data = [
    (y_unscaled[:, 1], "Actual Likes"),
    (pred_unscaled[:, 1], "Predicted Likes")
]

save_graph(
    filename="likes_predicted_vs_actual",
    data=likes_data,
    x_label="Sample Index",
    y_label="Like Count",
    title="Predicted vs Actual Likes",
    figsize=(12, 8)
)