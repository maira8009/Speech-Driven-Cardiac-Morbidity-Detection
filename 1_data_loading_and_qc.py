
import os, pandas as pd, numpy as np, json
from config import files, paths, settings
from utils_io import save_table
from typing import Dict, Any

np.random.seed(settings.random_seed)

def load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"missing file: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"loaded {path} shape {df.shape}")
    return df

def audit_dataframe(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    if df.empty:
        return { "name": name, "rows": 0, "cols": 0 }
    summary = {
        "name": name,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing_count": {k: int(df[k].isna().sum()) for k in df.columns},
        "missing_pct": {k: float(df[k].isna().mean()) for k in df.columns},
        "describe_numeric": df.select_dtypes(include=[np.number]).describe().to_dict(),
        "head": df.head(settings.max_rows).to_dict(orient="list"),
    }
    return summary

def main():
    speech_final = load_csv_safe(files.speech_features_final)
    speech_proc = load_csv_safe(files.speech_features_processed)
    ecg = load_csv_safe(files.ecg_features)

    audits = []
    audits.append(audit_dataframe(speech_final, "final_speech_features"))
    audits.append(audit_dataframe(speech_proc, "processed_speech_features"))
    audits.append(audit_dataframe(ecg, "ECG_features_clean"))

    audit_df = pd.DataFrame([
        {"dataset": a["name"], "rows": a.get("rows", 0), "cols": a.get("cols", 0)}
        for a in audits
    ])
    save_table(audit_df, "dataset_dimensions")

    # Schema intersection and join keys
    common_cols = set(speech_final.columns).intersection(set(ecg.columns))
    key_candidates = [settings.id_column, settings.time_column, settings.target_column]
    keys_present = [k for k in key_candidates if k in common_cols]
    pd.DataFrame({"common_columns": list(common_cols)}).to_csv(
        os.path.join(paths.tables_dir, "common_columns.csv"), index=False
    )
    pd.DataFrame({"key_candidates_present": keys_present}).to_csv(
        os.path.join(paths.tables_dir, "key_candidates_present.csv"), index=False
    )
    print("common columns saved")

    # Save audits as json
    with open(os.path.join(paths.tables_dir, "audits.json"), "w", encoding="utf8") as f:
        json.dump(audits, f, indent=2)

if __name__ == "__main__":
    main()
