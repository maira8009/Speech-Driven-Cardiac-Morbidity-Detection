
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from config import paths
from utils_io import save_figure, save_table

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from 6_diffusion_model import SimpleDDPM  # noqa

def main():
    # Baselines are trained in 5_models_baselines.py
    # Evaluate diffusion reconstruction fidelity if torch is available
    if not TORCH_AVAILABLE:
        print("torch not available, skip diffusion eval")
        return

    # Load cached merged features to rebuild blocks like in 6_diffusion_model
    merged_path = os.path.join(paths.tables_dir, "merged_preview.csv")
    if not os.path.exists(merged_path):
        print("missing merged_preview.csv")
        return
    merged = pd.read_csv(merged_path).select_dtypes(include=[float, int]).fillna(0.0)
    speech_cols = [c for c in merged.columns if "speech" in c.lower()]
    ecg_cols = [c for c in merged.columns if "ecg" in c.lower()]
    if not speech_cols or not ecg_cols:
        mid = merged.shape[1] // 2
        ecg_block = merged.iloc[:, :mid].values
        speech_block = merged.iloc[:, mid:].values
    else:
        ecg_block = merged[ecg_cols].values
        speech_block = merged[speech_cols].values

    speech_block = (speech_block - speech_block.mean(axis=0)) / (speech_block.std(axis=0)+1e-8)
    ecg_block = (ecg_block - ecg_block.mean(axis=0)) / (ecg_block.std(axis=0)+1e-8)

    device = "cpu"
    X_target = torch.tensor(ecg_block, dtype=torch.float32, device=device)
    X_cond = torch.tensor(speech_block, dtype=torch.float32, device=device)

    dim = X_target.shape[1]
    cond_dim = X_cond.shape[1]
    ddpm = SimpleDDPM(dim, cond_dim, timesteps=200, device=device)
    model_path = os.path.join(paths.models_dir, "ddpm_denoiser.pt")
    if not os.path.exists(model_path):
        print("trained denoiser not found. run 6_diffusion_model.py first")
        return
    ddpm.model.load_state_dict(torch.load(model_path, map_location=device))

    with torch.no_grad():
        recon = ddpm.sample(n=X_cond.shape[0], cond=X_cond).cpu().numpy()

    mse = mean_squared_error(X_target.cpu().numpy().ravel(), recon.ravel())
    pd.DataFrame([{"metric":"mse_reconstruction", "value":mse}]).to_csv(
        os.path.join(paths.tables_dir, "diffusion_mse.csv"), index=False
    )

    # Plot sample real versus generated for first 3 features
    for i in range(min(3, recon.shape[1])):
        fig, ax = plt.subplots()
        ax.plot(X_target[:300, i].cpu().numpy(), label="real")
        ax.plot(recon[:300, i], label="generated")
        ax.set_title(f"ECG feature index {i} real versus generated")
        ax.legend()
        save_figure(fig, f"recon_feature_{i}")
        plt.close(fig)

if __name__ == "__main__":
    main()
