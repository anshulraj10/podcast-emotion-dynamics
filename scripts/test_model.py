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

# allow loading torch Subset
add_safe_globals([Subset])

# ---------- Config ----------
ROOT = Path(__file__).parent.parent
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MODEL_PATH = ROOT / "models" / "engagement_model.pt"
TARGET_SCALER_PATH = ROOT / "models" / "target_scaler.pkl"
TEST_DATASET_PATH = ROOT / "data/processed/test_dataset.pt"

HIDDEN_SIZE = 128
TRANSFORMER_HEADS = 4
DROPOUT = 0.15
BATCH_SIZE = 128

# --- helper: inverse-transform a single target column from scaled-log space ---
def inverse_single_col(scaled_col_1d, scaler, col_idx: int):
    """
    scaled_col_1d: shape (N, 1) scaled-log values for a single target
    scaler: RobustScaler fitted on 2 columns [views, likes]
    col_idx: 0 for views, 1 for likes
    """
    n = scaled_col_1d.shape[0]
    tmp = np.zeros((n, 2), dtype=float)
    tmp[:, col_idx] = scaled_col_1d[:, 0]
    inv = scaler.inverse_transform(tmp)
    return inv[:, col_idx].reshape(-1, 1)


def main():
    # ----- Load test subset and peek one batch to infer input dims -----
    test_ds = torch.load(TEST_DATASET_PATH, weights_only=False)
    probe_dl = DataLoader(test_ds, batch_size=1)  # small probe
    xb_seq, xb_static, _ = next(iter(probe_dl))
    seq_input_dim = xb_seq.shape[2]
    static_input_dim = xb_static.shape[1]

    # ----- Build model with inferred dims and load weights -----
    model = EngagementModel(
        seq_input_dim=seq_input_dim,
        static_input_dim=static_input_dim,
        hidden_size=HIDDEN_SIZE,
        transformer_heads=TRANSFORMER_HEADS,
        dropout=DROPOUT
    ).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Full dataloader
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # ----- Inference -----
    all_pred_scaled_log = []
    all_true_scaled_log = []

    # for quantile coverage
    all_views_p10, all_views_p90 = [], []
    all_likes_p10, all_likes_p90 = [], []

    with torch.no_grad():
        for xb_seq, xb_static, yb in test_dl:
            xb_seq = xb_seq.to(DEVICE)
            xb_static = xb_static.to(DEVICE)

            out = model(xb_seq, xb_static)

            mean_pred = torch.cat([out["views_mean"], out["likes_mean"]], dim=1).cpu().numpy()
            y_true = yb.cpu().numpy()

            all_pred_scaled_log.append(mean_pred)
            all_true_scaled_log.append(y_true)

            all_views_p10.append(out["views_p10"].cpu().numpy())
            all_views_p90.append(out["views_p90"].cpu().numpy())
            all_likes_p10.append(out["likes_p10"].cpu().numpy())
            all_likes_p90.append(out["likes_p90"].cpu().numpy())

    pred_scaled_log = np.concatenate(all_pred_scaled_log, axis=0)
    y_scaled_log    = np.concatenate(all_true_scaled_log, axis=0)

    # ----- Inverse transforms: scaled-log -> log -> counts -----
    with open(TARGET_SCALER_PATH, "rb") as f:
        target_scaler = pickle.load(f)

    pred_log = target_scaler.inverse_transform(pred_scaled_log)
    y_log    = target_scaler.inverse_transform(y_scaled_log)

    pred_counts = np.expm1(pred_log)
    y_counts    = np.expm1(y_log)

    # ----- Metrics -----
    rmse = np.sqrt(np.mean((pred_counts - y_counts) ** 2, axis=0))
    mae  = np.mean(np.abs(pred_counts - y_counts), axis=0)

    print(f"Total test samples: {len(pred_counts)}")
    print(f"ViewCount  →  RMSE = {rmse[0]:,.2f}  |  MAE = {mae[0]:,.2f}")
    print(f"LikeCount  →  RMSE = {rmse[1]:,.2f}  |  MAE = {mae[1]:,.2f}")

    # Optional: log-space diagnostics
    rmse_log = np.sqrt(np.mean((pred_log - y_log) ** 2, axis=0))
    mae_log  = np.mean(np.abs(pred_log - y_log) ** 1, axis=0)  # L1 in log space
    print(f"(Log space) Views RMSE={rmse_log[0]:.4f}, Likes RMSE={rmse_log[1]:.4f}")
    print(f"(Log space) Views MAE={mae_log[0]:.4f}, Likes MAE={mae_log[1]:.4f}")

    # ----- Quantile coverage (p10–p90) -----
    v10_scaled = np.concatenate(all_views_p10, axis=0)  # (N,1)
    v90_scaled = np.concatenate(all_views_p90, axis=0)  # (N,1)
    l10_scaled = np.concatenate(all_likes_p10, axis=0)  # (N,1)
    l90_scaled = np.concatenate(all_likes_p90, axis=0)  # (N,1)

    # invert each column separately (scaled-log -> log)
    v10_log = inverse_single_col(v10_scaled, target_scaler, col_idx=0)
    v90_log = inverse_single_col(v90_scaled, target_scaler, col_idx=0)
    l10_log = inverse_single_col(l10_scaled, target_scaler, col_idx=1)
    l90_log = inverse_single_col(l90_scaled, target_scaler, col_idx=1)

    # log -> counts
    v10 = np.expm1(v10_log).ravel()
    v90 = np.expm1(v90_log).ravel()
    l10 = np.expm1(l10_log).ravel()
    l90 = np.expm1(l90_log).ravel()

    views_cover = np.mean((y_counts[:, 0] >= v10) & (y_counts[:, 0] <= v90))
    likes_cover = np.mean((y_counts[:, 1] >= l10) & (y_counts[:, 1] <= l90))
    print(f"Coverage (p10–p90): Views={views_cover:.3f}, Likes={likes_cover:.3f}")

    # ----- Plots -----
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
