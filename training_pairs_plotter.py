# plot_training_pairs.py
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_pairs(file_path="training_pairs.npz", save_dir="plots", n_samples=None):
    data = np.load(file_path)
    x_minus = data["x_minus"]
    x_plus = data["x_plus"]

    print(len(x_minus))

    # Limit number of pairs to plot
    if n_samples is not None:
        x_minus = x_minus[:n_samples]
        x_plus = x_plus[:n_samples]

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"training_pairs_{len(x_minus)}.png")

    plt.figure(figsize=(8, 8))
    plt.scatter(x_minus[:, 0], x_minus[:, 1], color='red', label='x-', alpha=0.6)
    plt.scatter(x_plus[:, 0], x_plus[:, 1], color='blue', label='x+', alpha=0.6)

    # Draw lines connecting each pair
    for xm, xp in zip(x_minus, x_plus):
        plt.plot([xm[0], xp[0]], [xm[1], xp[1]], color='gray', alpha=0.3)

    plt.title(f"Visualization of (x−, x+) Training Pairs (first {len(x_minus)})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {save_path}")

# Example usage
if __name__ == "__main__":
    plot_pairs("training_pairs.npz", n_samples=50)
