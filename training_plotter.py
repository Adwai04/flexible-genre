import numpy as np
import matplotlib.pyplot as plt

def load_weights(npz_path):
    data = np.load(npz_path)
    w = data["w"]       # shape: (N, 2)
    x_vals = w[:, 0]    # take first component x
    return x_vals

def plot_distribution(x_vals, title, bins=np.arange(0, 1.1, 0.1), fname="plots"):
    hist, edges = np.histogram(x_vals, bins=bins)
    hist_normalized = hist / hist.sum()       # normalize
    centers = (edges[:-1] + edges[1:]) / 2    # bin centers

    plt.figure(figsize=(6, 4))
    plt.bar(centers, hist_normalized, width=0.08, edgecolor='black')
    plt.title(f"Weight Distribution – {title}")
    plt.xlabel("weight_1")
    plt.ylabel("Normalized count")
    plt.ylim(0, max(hist_normalized) * 1.2)
    # plt.grid(alpha=0.3)
    plt.savefig(f"results/training_distribution/{name}.png")
    # plt.show()


# ---- Load and Plot Each Dataset Separately ----

datasets = {
    "squares": "training_pairs_squares.npz",
    "circles": "training_pairs_circles.npz",
    "moons":   "training_pairs_moons.npz"
}

for name, path in datasets.items():
    x_vals = load_weights(path)
    plot_distribution(x_vals, name, fname="training_pairs_"+name)
