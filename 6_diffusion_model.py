
import os, numpy as np, pandas as pd, math
from typing import Tuple
from config import paths, settings

# Optional torch import guarded to allow earlier steps to run without torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception as e:
    print("torch is not available, diffusion model will not run")
    TORCH_AVAILABLE = False

def get_data():
    X = np.load(os.path.join(paths.cache_dir, "X_merged.npy"))
    y_path = os.path.join(paths.cache_dir, "y.npy")
    y = np.load(y_path) if os.path.exists(y_path) else None
    return X, y

# Simple U Net like 1D block for denoising over feature vectors interpreted as 1D sequences
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
    def forward(self, x):
        return x + self.net(x)

class Denoiser(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.fc_in = nn.Linear(dim + cond_dim, dim)
        self.blocks = nn.Sequential(
            ResidualBlock(dim),
            ResidualBlock(dim),
            ResidualBlock(dim),
        )
        self.fc_out = nn.Linear(dim, dim)
    def forward(self, x_noisy, cond):
        h = torch.cat([x_noisy, cond], dim=-1)
        h = self.fc_in(h)
        h = self.blocks(h)
        return self.fc_out(h)

class SimpleDDPM:
    def __init__(self, dim, cond_dim, timesteps=1000, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)
        self.model = Denoiser(dim, cond_dim).to(device)
        self.dim = dim
        self.cond_dim = cond_dim

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = torch.sqrt(self.alpha_bars[t]).view(-1,1)
        sqrt_one_minus = torch.sqrt(1.0 - self.alpha_bars[t]).view(-1,1)
        return sqrt_ab * x0 + sqrt_one_minus * noise

    def p_losses(self, x0, cond, t):
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        pred_noise = self.model(x_noisy, cond)
        return F.mse_loss(pred_noise, noise)

    def train_loop(self, X_target, X_cond, steps=2000, lr=1e-3, batch_size=64):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        n = X_target.shape[0]
        for step in range(steps):
            idx = torch.randint(0, n, (batch_size,), device=self.device)
            xt = X_target[idx]
            ct = X_cond[idx]
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
            loss = self.p_losses(xt, ct, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 0:
                print(f"step {step} loss {loss.item():.6f}")

    def sample(self, n, cond):
        x = torch.randn(n, self.dim, device=self.device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.tensor([t]*n, device=self.device)
            # predict noise
            noise_pred = self.model(x, cond)
            alpha_bar_t = self.alpha_bars[t]
            x = (x - (1 - alpha_bar_t).sqrt() * noise_pred) / (alpha_bar_t.sqrt() + 1e-8)
            if t > 0:
                x = x + torch.randn_like(x) * (1 - alpha_bar_t).sqrt()
        return x

def main():
    if not TORCH_AVAILABLE:
        print("skip diffusion training since torch is not available")
        return
    X, y = get_data()
    # Split into condition speech and target ecg feature blocks by simple column rules
    # Here we assume speech feature names include 'speech' and ecg include 'ecg' after merging in step 2
    merged_cols_path = os.path.join(paths.tables_dir, "merged_preview.csv")
    if not os.path.exists(merged_cols_path):
        print("merged preview not found")
        return
    merged = pd.read_csv(merged_cols_path)
    speech_cols = [c for c in merged.columns if "speech" in c.lower()]
    ecg_cols = [c for c in merged.columns if "ecg" in c.lower()]
    if not speech_cols or not ecg_cols:
        # fallback split
        mid = X.shape[1] // 2
        ecg_block = X[:, :mid]
        speech_block = X[:, mid:]
    else:
        # use the learned preprocessing order to map back is complex
        # naive fallback: use raw columns from merged with imputation free of pipelines
        mf = merged.select_dtypes(include=[np.number]).fillna(merged.select_dtypes(include=[np.number]).median())
        speech_block = mf[speech_cols].values
        ecg_block = mf[ecg_cols].values

    # standardize blocks
    speech_block = (speech_block - speech_block.mean(axis=0)) / (speech_block.std(axis=0)+1e-8)
    ecg_block = (ecg_block - ecg_block.mean(axis=0)) / (ecg_block.std(axis=0)+1e-8)

    device = "cpu"
    X_target = torch.tensor(ecg_block, dtype=torch.float32, device=device)
    X_cond = torch.tensor(speech_block, dtype=torch.float32, device=device)
    dim = X_target.shape[1]
    cond_dim = X_cond.shape[1]

    ddpm = SimpleDDPM(dim, cond_dim, timesteps=200, device=device)
    ddpm.train_loop(X_target, X_cond, steps=1000, lr=1e-3, batch_size=64)

    torch.save(ddpm.model.state_dict(), os.path.join(paths.models_dir, "ddpm_denoiser.pt"))
    print("saved diffusion denoiser")

if __name__ == "__main__":
    main()
