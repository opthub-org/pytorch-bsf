import matplotlib.pyplot as plt
import numpy as np

from torch_bsf.model_selection.elastic_net_grid import elastic_net_grid


def project_simplex(points):
    # Project 3D simplex points to 2D
    # (1,0,0) -> (0,0)
    # (0,1,0) -> (1,0)
    # (0,0,1) -> (0.5, np.sqrt(3)/2)
    y = points[:, 1]
    z = points[:, 2]

    px = y + 0.5 * z
    py = np.sqrt(3) / 2 * z
    return px, py


def plot_grid(base, ax, title):
    grid = elastic_net_grid(n_lambdas=20, n_alphas=10, base=base)
    px, py = project_simplex(grid)
    ax.scatter(px, py, s=10, alpha=0.6, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_title(title)

    # Draw simplex boundary
    boundary = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(boundary[:, 0], boundary[:, 1], "k-", lw=1, alpha=0.3)

    # Label vertices
    ax.text(-0.05, -0.05, "(1,0,0)")
    ax.text(0.95, -0.05, "(0,1,0)")
    ax.text(0.45, np.sqrt(3) / 2 + 0.02, "(0,0,1)")
    ax.axis("off")


fig, axes = plt.subplots(1, 4, figsize=(16, 4))
plot_grid(1, axes[0], "base=1 (Linear)")
plot_grid(10, axes[1], "base=10")
plot_grid(100, axes[2], "base=100")
plot_grid(1000, axes[3], "base=1000")

plt.tight_layout()
plt.savefig("docs/_static/elastic_net_grid_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
