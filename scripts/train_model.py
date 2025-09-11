# Deep Learning Model to Predict View and Like Count from Features (with quantiles)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from EngagementModel import EngagementModel
from EngagementDataset import EngagementDataset
from utils import save_graph
import math
import torch.nn.functional as F

# ---------- Config ----------
ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "data/processed/dataset.csv"
TEST_DATASET_PATH = ROOT / "data/processed/test_dataset.pt"
MODEL_PATH = ROOT / "models/engagement_model.pt"

TARGET_COLUMNS = ["viewCount", "likeCount"]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 5e-4
TRANSFORMER_HEADS = 4
HIDDEN_SIZE = 128
DROPOUT = 0.15

# ---------- Losses ----------
def tilted_loss(q, y, f):
    e = y - f
    return torch.mean(torch.maximum(q * e, (q - 1) * e))

LAMBDA_Q = 0.25     # weight for quantile loss
LAMBDA_MONO = 0.05  # monotonicity penalty p10<=mean<=p90

def log_cosh_loss(y_pred, y_true):
    # stable: logcosh(x) = |x| + softplus(-2|x|) - log(2)
    x = y_pred - y_true
    z = torch.abs(x)
    return torch.mean(z + F.softplus(-2.0 * z) - math.log(2.0))

if __name__ == "__main__":
    # ----- Load Data -----
    df = pd.read_csv(DATA_FILE)
    dataset = EngagementDataset(df)

    val_size = test_size = int(0.05 * len(dataset))
    train_size = len(dataset) - val_size - test_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    torch.save(test_ds, TEST_DATASET_PATH)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # ----- Model -----
    seq_input_dim = dataset.X_seq.shape[2]
    static_input_dim = dataset.X_static.shape[1]
    model = EngagementModel(
        seq_input_dim=seq_input_dim,
        static_input_dim=static_input_dim,
        hidden_size=HIDDEN_SIZE,
        transformer_heads=TRANSFORMER_HEADS,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = log_cosh_loss
    mae_criterion = nn.L1Loss()
    writer = SummaryWriter(log_dir=str(ROOT / "models" / "runs"))

    # ----- Train Loop -----
    train_losses, val_losses, train_maes, val_maes = [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_train_loss, epoch_train_mae = [], []

        for xb_seq, xb_static, yb in train_dl:
            xb_seq, xb_static, yb = xb_seq.to(DEVICE), xb_static.to(DEVICE), yb.to(DEVICE)
            out = model(xb_seq, xb_static)

            mean_pred = torch.cat([out["views_mean"], out["likes_mean"]], dim=1)
            loss_point = criterion(mean_pred, yb)
            mae_point = mae_criterion(mean_pred, yb)

            # split targets
            y_views = yb[:, 0:1]
            y_likes = yb[:, 1:2]

            qloss = (
                tilted_loss(0.10, y_views, out["views_p10"]) +
                tilted_loss(0.90, y_views, out["views_p90"]) +
                tilted_loss(0.10, y_likes, out["likes_p10"]) +
                tilted_loss(0.90, y_likes, out["likes_p90"])
            )

            mono_pen = (
                torch.relu(out["views_p10"] - out["views_mean"]).mean() +
                torch.relu(out["views_mean"] - out["views_p90"]).mean() +
                torch.relu(out["likes_p10"] - out["likes_mean"]).mean() +
                torch.relu(out["likes_mean"] - out["likes_p90"]).mean()
            )

            loss = loss_point + LAMBDA_Q * qloss + LAMBDA_MONO * mono_pen

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss.append(loss.item())
            epoch_train_mae.append(mae_point.item())

        # ----- Validation -----
        model.eval()
        epoch_val_loss, epoch_val_mae = [], []
        with torch.no_grad():
            for xb_seq, xb_static, yb in val_dl:
                xb_seq, xb_static, yb = xb_seq.to(DEVICE), xb_static.to(DEVICE), yb.to(DEVICE)
                out = model(xb_seq, xb_static)

                mean_pred = torch.cat([out["views_mean"], out["likes_mean"]], dim=1)
                loss_point = criterion(mean_pred, yb)

                y_views = yb[:, 0:1]
                y_likes = yb[:, 1:2]
                qloss = (
                    tilted_loss(0.10, y_views, out["views_p10"]) +
                    tilted_loss(0.90, y_views, out["views_p90"]) +
                    tilted_loss(0.10, y_likes, out["likes_p10"]) +
                    tilted_loss(0.90, y_likes, out["likes_p90"])
                )
                mono_pen = (
                    torch.relu(out["views_p10"] - out["views_mean"]).mean() +
                    torch.relu(out["views_mean"] - out["views_p90"]).mean() +
                    torch.relu(out["likes_p10"] - out["likes_mean"]).mean() +
                    torch.relu(out["likes_mean"] - out["likes_p90"]).mean()
                )
                loss = loss_point + LAMBDA_Q * qloss + LAMBDA_MONO * mono_pen

                mae_point = mae_criterion(mean_pred, yb)
                epoch_val_loss.append(loss.item())
                epoch_val_mae.append(mae_point.item())

        train_losses.append(np.mean(epoch_train_loss))
        val_losses.append(np.mean(epoch_val_loss))
        train_maes.append(np.mean(epoch_train_mae))
        val_maes.append(np.mean(epoch_val_mae))

        writer.add_scalars("Loss", {"Train": train_losses[-1], "Val": val_losses[-1]}, epoch)
        writer.add_scalars("MAE", {"Train": train_maes[-1], "Val": val_maes[-1]}, epoch)
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss={train_losses[-1]:.4f}  Val Loss={val_losses[-1]:.4f}  "
            f"Train MAE={train_maes[-1]:.4f}  Val MAE={val_maes[-1]:.4f}"
        )

    # ----- Plots -----
    save_graph(
        filename="train-vs-val-loss",
        data=[(train_losses, "Train Loss"), (val_losses, "Val Loss")],
        x_label="Epoch",
        y_label="Loss",
        title="Train vs Val Loss"
    )
    save_graph(
        filename="train-vs-val-mae",
        data=[(train_maes, "Train MAE"), (val_maes, "Val MAE")],
        x_label="Epoch",
        y_label="MAE",
        title="Train vs Val MAE"
    )

    # ----- Save Model -----
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")
