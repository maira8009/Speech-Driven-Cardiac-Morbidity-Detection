
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from config import paths
from utils_io import save_figure

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

def permutation_importance(model, X, y, metric_fn, n_repeats=5, random_state=1337):
    rng = np.random.RandomState(random_state)
    base = metric_fn(model, X, y)
    n_features = X.shape[1]
    importances = np.zeros(n_features)
    for j in range(n_features):
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            s = metric_fn(model, Xp, y)
            scores.append(base - s)
        importances[j] = np.mean(scores)
    return importances

def main():
    # This script provides hooks and plots for interpretability.
    # For baselines, you can load a trained sklearn model and compute permutation importances.
    # For diffusion, if torch is available, gradient saliency through denoiser can be visualized.
    print("interpretability hooks ready. integrate with your trained objects.")

if __name__ == "__main__":
    main()
