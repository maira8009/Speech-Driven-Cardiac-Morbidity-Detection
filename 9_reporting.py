
import os, pandas as pd
from config import paths

def main():
    tables = []
    for name in ["baseline_metrics.csv", "diffusion_mse.csv", "dataset_dimensions.csv", "feature_matrix_shape.csv"]:
        p = os.path.join(paths.tables_dir, name)
        if os.path.exists(p):
            tables.append(p)
    print("ready to package the following tables:")
    for t in tables:
        print(t)

if __name__ == "__main__":
    main()
