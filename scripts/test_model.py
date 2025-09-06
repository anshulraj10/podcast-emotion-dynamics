# scripts/test_model.py
import pickle
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.serialization import add_safe_globals
from torch.utils.data.dataset import Subset

from EngagementModel import EngagementModel
from utils import save_scatter_graph, save_graph

# --- Torch safety for loading saved Subset objects ---
add_safe_globals([Subset])

# --- CONFIG ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT = Path(__file__).parent.parent

MODEL_PATH = ROOT / "models/engagement_model.pt"
TARGET_SCALER_PATH = ROOT / "models/target_scaler.pkl"
TEST_DATASET_PATH = ROOT / "data/processed/test_dataset.pt"
IMAGES_DIR = ROOT / "images"

# Must match training-time architecture
SEQ_INPUT_DIM = 29          # number of emotion features per timestep
STATIC_INPUT_DIM = 6        # number of static features
HIDDEN_SIZE = 128
TRANSFORMER_HEADS = 4
DROPOUT = 0.15
BATCH_SIZE = 128

def main():
    # --- Build model and load weights ---
    model = EngagementModel(
        seq_input_dim=SEQ_INPUT_DIM,
        static_input_dim=STATIC_INPUT_DIM,
        hidden_size=HIDDEN_SIZE,
        transformer_heads=TRANSFORMER_HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- Load test dataset subset saved during training ---
    test_ds = torch.load(TEST_DATASET_PATH, weights_only=False)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # --- Forward pass over test set ---
    all_pred_scaled_log = []
    all_true_scaled_log = []

    with torch.no_grad():
        for xb_seq, xb_static, yb in test_dl:
            out = model(xb_seq.to(DEVICE), xb_static.to(DEVICE))
            # Concatenate the two point-estimate heads in the same order as targets: [views, likes]
            pred_batch = torch.cat([out["views_mean"], out["likes_mean"]], dim=1).cpu().numpy()
            y_true_batch = yb.cpu().numpy()

            all_pred_scaled_log.append(pred_batch)
            all_true_scaled_log.append(y_true_batch)

    pred_scaled_log = np.concatenate(all_pred_scaled_log, axis=0)
    y_scaled_log = np.concatenate(all_true_scaled_log, axis=0)

    # --- Invert scaling correctly: scaled-log -> log -> counts ---
    with open(TARGET_SCALER_PATH, "rb") as f:
        target_scaler = pickle.load(f)

    pred_log = target_scaler.inverse_transform(pred_scaled_log)
    y_log = target_scaler.inverse_transform(y_scaled_log)

    pred_counts = np.expm1(pred_log)
    y_counts = np.expm1(y_log)

    # --- Metrics (count space) ---
    rmse = np.sqrt(np.mean((pred_counts - y_counts) ** 2, axis=0))
    mae = np.mean(np.abs(pred_counts - y_counts), axis=0)

    print(f"Total test samples: {len(pred_counts)}")
    print(f"ViewCount  →  RMSE = {rmse[0]:,.2f}  |  MAE = {mae[0]:,.2f}")
    print(f"LikeCount  →  RMSE = {rmse[1]:,.2f}  |  MAE = {mae[1]:,.2f}")

    # (Optional) Diagnostics on the log scale
    rmse_log = np.sqrt(np.mean((pred_log - y_log) ** 2, axis=0))
    mae_log = np.mean(np.abs(pred_log - y_log), axis=0)
    print(f"(Log space) Views RMSE={rmse_log[0]:.4f}, Likes RMSE={rmse_log[1]:.4f}")
    print(f"(Log space) Views MAE={mae_log[0]:.4f}, Likes MAE={mae_log[1]:.4f}")

    # --- Plots ---
    # Scatter: Predicted vs Actual (counts)
    save_scatter_graph(
        filename="views_scatter_predicted_vs_actual",
        actual=y_counts[:, 0],
        predicted=pred_counts[:, 0],
        x_label="Actual Views",
        y_label="Predicted Views",
        title="Views: Predicted vs Actual (Scatter Plot)",
    )
    save_scatter_graph(
        filename="likes_scatter_predicted_vs_actual",
        actual=y_counts[:, 1],
        predicted=pred_counts[:, 1],
        x_label="Actual Likes",
        y_label="Predicted Likes",
        title="Likes: Predicted vs Actual (Scatter Plot)",
    )

    # Line plots over sample index (counts)
    save_graph(
        filename="views_predicted_vs_actual",
        data=[(y_counts[:, 0], "Actual Views"), (pred_counts[:, 0], "Predicted Views")],
        x_label="Sample Index",
        y_label="View Count",
        title="Predicted vs Actual Views",
        figsize=(12, 8),
    )
    save_graph(
        filename="likes_predicted_vs_actual",
        data=[(y_counts[:, 1], "Actual Likes"), (pred_counts[:, 1], "Predicted Likes")],
        x_label="Sample Index",
        y_label="Like Count",
        title="Predicted vs Actual Likes",
        figsize=(12, 8),
    )

if __name__ == "__main__":
    main()
