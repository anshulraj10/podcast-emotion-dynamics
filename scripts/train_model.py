# Deep Learning Model to Predict View and Like Count from Features

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import ast
from EngagementModel import EngagementModel
from EngagementDataset import EngagementDataset
from utils import save_graph

# --- CONFIG ---
DATA_FILE = Path(__file__).parent.parent / Path("data/processed/dataset.csv")
TEST_DATASET_PATH = Path(__file__).parent.parent / Path("data/processed/test_dataset.pt")
MODEL_PATH = Path(__file__).parent.parent / Path("models/engagement_model.pt")
TARGET_COLUMNS = ["viewCount", "likeCount"]

# --- Quantile Loss ---
def tilted_loss(q, y, f):
    e = y - f
    return torch.mean(torch.max(q * e, (q - 1) * e))
       
# --- Postprocessing Utility ---
def inverse_transform_and_clip(preds, scaler, percentile_bounds=(5, 95)):
    preds_np = np.expm1(preds)
    preds_unscaled = scaler.inverse_transform(preds_np)
    clipped = np.clip(
        preds_unscaled,
        np.percentile(preds_unscaled, percentile_bounds[0], axis=0),
        np.percentile(preds_unscaled, percentile_bounds[1], axis=0)
    )
    return clipped

if __name__ == "__main__":
    # --- Model Config ---
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 5e-4
    TRANSFORMER_HEADS = 4
    HIDDEN_SIZE = 128
    DROPOUT = 0.15
    
    # --- Load Data ---
    df = pd.read_csv(DATA_FILE)
    dataset = EngagementDataset(df)

    val_size = test_size = int(0.05 * len(dataset))
    train_size = len(dataset) - val_size - test_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    torch.save(test_ds, TEST_DATASET_PATH)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # --- Initialize Model ---
    seq_input_dim = dataset.X_seq.shape[2]
    static_input_dim = dataset.X_static.shape[1]
    model = EngagementModel(seq_input_dim=seq_input_dim, static_input_dim=static_input_dim,
                            hidden_size=HIDDEN_SIZE, transformer_heads=TRANSFORMER_HEADS, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.HuberLoss()
    mae_criterion = nn.L1Loss()
    writer = SummaryWriter(log_dir="../models/runs")

    # --- Training Loop ---
    train_losses, val_losses, train_maes, val_maes = [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_train_loss, epoch_train_mae = [], []
        for xb_seq, xb_static, yb in train_dl:
            xb_seq, xb_static, yb = xb_seq.to(DEVICE), xb_static.to(DEVICE), yb.to(DEVICE)
            out = model(xb_seq, xb_static)
            pred = torch.cat([out['views_mean'], out['likes_mean']], dim=1)
            loss = criterion(pred, yb)
            mae = mae_criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())
            epoch_train_mae.append(mae.item())

        model.eval()
        epoch_val_loss, epoch_val_mae = [], []
        with torch.no_grad():
            for xb_seq, xb_static, yb in val_dl:
                xb_seq, xb_static, yb = xb_seq.to(DEVICE), xb_static.to(DEVICE), yb.to(DEVICE)
                out = model(xb_seq, xb_static)
                pred = torch.cat([out['views_mean'], out['likes_mean']], dim=1)
                loss = criterion(pred, yb)
                mae = mae_criterion(pred, yb)
                epoch_val_loss.append(loss.item())
                epoch_val_mae.append(mae.item())

        train_losses.append(np.mean(epoch_train_loss))
        val_losses.append(np.mean(epoch_val_loss))
        train_maes.append(np.mean(epoch_train_mae))
        val_maes.append(np.mean(epoch_val_mae))

        writer.add_scalars("Loss", {"Train": train_losses[-1], "Val": val_losses[-1]}, epoch)
        writer.add_scalars("MAE", {"Train": train_maes[-1], "Val": val_maes[-1]}, epoch)
        print(f"Epoch {epoch}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, Train MAE={train_maes[-1]:.4f}, Val MAE={val_maes[-1]:.4f}")
        
    
    save_graph(filename="train-vs-val-loss",
               data=[(train_losses, "Train Loss"), (val_losses, "Val Loss")],
               x_label="Epoch",
               y_label="Loss",
               title="Train vs Val Loss"
               )
    
    save_graph(filename="train-vs-val-mae",
               data=[(train_losses, "Train MAE"), (val_losses, "Val MAE")],
               x_label="Epoch",
               y_label="MAE",
               title="Train vs Val MAE"
               )
    
    # --- Save the Trained Model ---
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")