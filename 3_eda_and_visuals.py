
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from config import paths, files, settings
from utils_io import save_figure

def safe_load_npy(name):
    path = os.path.join(paths.cache_dir, name)
    if not os.path.exists(path):
        print(f"missing cache: {path}")
        return None
    return np.load(path)

def plot_histograms(df: pd.DataFrame, prefix: str):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols[:30]:
        fig, ax = plt.subplots()
        ax.hist(df[c].dropna().values, bins=50)
        ax.set_title(f"Histogram {c}")
        ax.set_xlabel(c)
        ax.set_ylabel("count")
        save_figure(fig, f"{prefix}_hist_{c}")
        plt.close(fig)

def plot_correlation(df: pd.DataFrame, name: str):
    corr = df.select_dtypes(include=[np.number]).corr().fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=6)
    ax.set_title("Correlation matrix")
    fig.colorbar(cax)
    save_figure(fig, name)
    plt.close(fig)

def plot_pca_tsne(X, y=None, prefix="emb"):
    if X is None:
        return
    pca = PCA(n_components=2, random_state=settings.random_seed)
    Xp = pca.fit_transform(X)
    fig, ax = plt.subplots()
    if y is None:
        ax.scatter(Xp[:,0], Xp[:,1], s=6)
    else:
        sc = ax.scatter(Xp[:,0], Xp[:,1], c=y, s=6)
        fig.colorbar(sc)
    ax.set_title("PCA two components")
    save_figure(fig, f"{prefix}_pca")
    plt.close(fig)

    tsne = TSNE(n_components=2, init="random", random_state=settings.random_seed, perplexity=30)
    Xt = tsne.fit_transform(X[: min(3000, X.shape[0])])
    fig, ax = plt.subplots()
    if y is None:
        ax.scatter(Xt[:,0], Xt[:,1], s=6)
    else:
        sc = ax.scatter(Xt[:,0], Xt[:,1], c=y[:Xt.shape[0]], s=6)
        fig.colorbar(sc)
    ax.set_title("tSNE two components")
    save_figure(fig, f"{prefix}_tsne")
    plt.close(fig)

def main():
    # Reload raw frames for correlation plots
    speech_path = files.speech_features_final
    ecg_path = files.ecg_features
    if os.path.exists(speech_path):
        speech = pd.read_csv(speech_path)
        plot_histograms(speech, "speech")
        plot_correlation(speech, "speech_correlation")
    if os.path.exists(ecg_path):
        ecg = pd.read_csv(ecg_path)
        plot_histograms(ecg, "ecg")
        plot_correlation(ecg, "ecg_correlation")

    X = safe_load_npy("X_merged.npy")
    y_path = os.path.join(paths.cache_dir, "y.npy")
    y = np.load(y_path) if os.path.exists(y_path) else None
    plot_pca_tsne(X, y, prefix="merged")

if __name__ == "__main__":
    main()
