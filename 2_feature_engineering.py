
import os, numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from config import files, paths, settings
from utils_io import save_table

np.random.seed(settings.random_seed)

def load_df(path):
    if not os.path.exists(path):
        print(f"missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

def build_preprocessor(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop", n_jobs=settings.n_jobs)
    return pre, numeric_cols, cat_cols

def main():
    speech = load_df(files.speech_features_final)
    ecg = load_df(files.ecg_features)

    if speech.empty or ecg.empty:
        print("warning missing datasets for preprocessing")
    # Save original column lists
    pd.Series(speech.columns).to_csv(os.path.join(paths.tables_dir, "speech_columns.csv"), index=False)
    pd.Series(ecg.columns).to_csv(os.path.join(paths.tables_dir, "ecg_columns.csv"), index=False)

    # Identify label and id if present
    y = None
    if settings.target_column in ecg.columns:
        y = ecg[settings.target_column].values
    elif settings.target_column in speech.columns:
        y = speech[settings.target_column].values

    # Align on subject id if available
    if settings.id_column in speech.columns and settings.id_column in ecg.columns:
        merged = pd.merge(speech, ecg, on=settings.id_column, suffixes=("_speech", "_ecg"))
    else:
        # fallback to cartesian alignment by index
        min_len = min(len(speech), len(ecg))
        merged = pd.concat([speech.iloc[:min_len].reset_index(drop=True),
                            ecg.iloc[:min_len].reset_index(drop=True)], axis=1)
    merged.to_csv(os.path.join(paths.tables_dir, "merged_preview.csv"), index=False)

    pre, num_cols, cat_cols = build_preprocessor(merged)
    X = pre.fit_transform(merged)
    np.save(os.path.join(paths.cache_dir, "X_merged.npy"), X)
    if y is not None:
        np.save(os.path.join(paths.cache_dir, "y.npy"), y)

    meta = pd.DataFrame({
        "numeric_cols": pd.Series(num_cols),
        "categorical_cols": pd.Series(cat_cols)
    })
    save_table(pd.DataFrame({"n_samples":[X.shape[0]], "n_features":[X.shape[1]]}), "feature_matrix_shape")
    print("preprocessing complete")

if __name__ == "__main__":
    main()
