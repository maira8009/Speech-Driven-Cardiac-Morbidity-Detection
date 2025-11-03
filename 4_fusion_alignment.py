
import os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from config import files, paths, settings
from utils_io import save_table

def main():
    X_path = os.path.join(paths.cache_dir, "X_merged.npy")
    y_path = os.path.join(paths.cache_dir, "y.npy")
    if not os.path.exists(X_path):
        print("missing X cache")
        return
    X = np.load(X_path)
    y = np.load(y_path) if os.path.exists(y_path) else np.zeros(X.shape[0], dtype=int)

    # Create reproducible folds
    skf = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_seed) if len(np.unique(y))>1 else None
    fold_rows = []
    if skf:
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            for idx in tr:
                fold_rows.append({"index": int(idx), "fold": int(fold), "split": "train"})
            for idx in va:
                fold_rows.append({"index": int(idx), "fold": int(fold), "split": "valid"})
    else:
        # Unlabeled case: simple split
        n = X.shape[0]
        cut = int(0.8*n)
        tr_idx = np.arange(0, cut)
        va_idx = np.arange(cut, n)
        for idx in tr_idx:
            fold_rows.append({"index": int(idx), "fold": 0, "split": "train"})
        for idx in va_idx:
            fold_rows.append({"index": int(idx), "fold": 0, "split": "valid"})
    folds_df = pd.DataFrame(fold_rows)
    save_table(folds_df, "cv_folds")
    print("fold assignments saved")

if __name__ == "__main__":
    main()
