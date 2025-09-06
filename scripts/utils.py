import matplotlib.pyplot as plt
from pathlib import Path

IMAGES_DIR = Path(__file__).parent.parent / Path("images")

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
    
def save_scatter_graph(filename, actual, predicted, x_label="Actual", y_label="Predicted", title="", figsize=(10, 8)):
    filepath = IMAGES_DIR / f"{filename}.png"
    plt.figure(figsize=figsize)
    
    # Scatter plot
    plt.scatter(actual, predicted, alpha=0.6, s=20)
    
    # Perfect prediction line (y=x)
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filepath)
    plt.close()