import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

IMAGES_DIR = Path(__file__).parent.parent / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def save_graph(filename, data, x_label="x-axis", y_label="y-axis", title="", figsize=(10, 6)):
    filepath = IMAGES_DIR / f"{filename}.png"
    plt.figure(figsize=figsize)
    for data_points, label in data:
        plt.plot(data_points, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(filepath)
    plt.close()

def save_scatter_graph(filename, actual, predicted, x_label="Actual", y_label="Predicted", title="", figsize=(10, 8)):
    filepath = IMAGES_DIR / f"{filename}.png"
    plt.figure(figsize=figsize)
    plt.scatter(actual, predicted, alpha=0.6, s=20)
    min_val = min(float(np.min(actual)), float(np.min(predicted)))
    max_val = max(float(np.max(actual)), float(np.max(predicted)))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filepath)
    plt.close()

