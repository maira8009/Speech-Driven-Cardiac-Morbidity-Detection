
import os, sys, json, time, contextlib
from datetime import datetime
from typing import Optional
from config import paths, settings

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

@contextlib.contextmanager
def logged_stdout(log_name: str):
    os.makedirs(paths.logs_dir, exist_ok=True)
    log_path = os.path.join(paths.logs_dir, f"{log_name}_{timestamp()}.log")
    with open(log_path, "w", encoding="utf8") as f, contextlib.redirect_stdout(f):
        print(f"[{datetime.now().isoformat()}] start {log_name}")
        yield
        print(f"[{datetime.now().isoformat()}] end {log_name}")

def save_figure(fig, name: str):
    path = os.path.join(paths.figures_dir, f"{name}.png")
    fig.savefig(path, dpi=settings.figure_dpi, bbox_inches="tight")
    print(f"saved figure: {path}")
    return path

def save_table(df, name: str):
    csv_path = os.path.join(paths.tables_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"saved table: {csv_path}")
    return csv_path
