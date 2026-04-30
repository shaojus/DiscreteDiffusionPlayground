import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RoPEAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = (d_model // nhead) ** -0.5

    def forward(self, x, cos, sin):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.nhead, D // self.nhead).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(x)

class SamplingTracker:
    def __init__(self, steps, batch_size, seq_len, device):
        self.x_history = torch.empty(steps + 1, batch_size, seq_len, dtype=torch.long, device=device)
    def record(self, step_idx, xt):
        self.x_history[step_idx].copy_(xt)
    def finalize(self, mask_token):
        xh, mh = self.x_history, (self.x_history == mask_token)
        S1, B, L = xh.shape
        first_unmask, num_unmasks, num_remasks = torch.full((B, L), -1, device=xh.device), torch.zeros((B, L), device=xh.device), torch.zeros((B, L), device=xh.device)
        finalization_step = torch.full((B, L), S1 - 1, device=xh.device)
        for s in range(1, S1):
            became_unmasked, became_masked = mh[s-1] & (~mh[s]), (~mh[s-1]) & mh[s]
            first_unmask[(first_unmask < 0) & became_unmasked] = s
            num_unmasks += became_unmasked.float(); num_remasks += became_masked.float()
            finalization_step[xh[s-1] != xh[s]] = s
        return {"x_history": xh, "mask_history": mh, "first_unmask_step": first_unmask, "num_unmasks": num_unmasks, "num_remasks": num_remasks, "finalization_step": finalization_step}

class ReMDM(nn.Module):
    def __init__(self, d_model=128, nhead=4, d_hid=512, n_layers=6, dropout=0.1, max_len=16, mask_token=2, steps=16, eta=0.5, sampler="remdm-rescale", t_eps=1e-3, **kwargs):
        super().__init__()
        self.mask_token, self.steps, self.eta, self.sampler = mask_token, steps, eta, sampler
        self.t_eps = float(t_eps)
        self.emb = nn.Embedding(3, d_model)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model // nhead, 2).float() / (d_model // nhead)))
        self.register_buffer("inv_freq", inv_freq)
        self.layers = nn.ModuleList([nn.ModuleDict({'attn': RoPEAttention(d_model, nhead, dropout), 'norm1': nn.LayerNorm(d_model), 'mlp': nn.Sequential(nn.Linear(d_model, d_hid), nn.GELU(), nn.Linear(d_hid, d_model), nn.Dropout(dropout)), 'norm2': nn.LayerNorm(d_model)}) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        h = self.emb(x)
        cos, sin = self._get_rope(h)
        for layer in self.layers:
            h = h + layer['attn'](layer['norm1'](h), cos, sin)
            h = h + layer['mlp'](layer['norm2'](h))
        return self.fc(h)

    def _get_rope(self, x):
        t = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0).unsqueeze(1)
        return emb.cos(), emb.sin()

    def training_loss(self, x0):
        B, L = x0.shape
        eps = self.t_eps
        t = torch.rand(B, device=x0.device) * (1.0 - eps) + eps         # t in [eps, 1]
        mask = torch.rand((B, L), device=x0.device) < t.view(B, 1)
        xt = x0.clone(); xt[mask] = self.mask_token
        logits = self(xt)
        ce = F.cross_entropy(logits.transpose(1, 2), x0, reduction='none')   # (B, L)
        # MDLM ELBO: mean_b mean_l [ mask_bl * CE_bl / t_b ]
        return ((ce * mask.float()) / t.view(B, 1)).mean()

    @torch.no_grad()
    def sample(self, N, L, device):
        return self._run_remdm(N, L, device, track=False)

    @torch.no_grad()
    def sample_tracked(self, N, L, device):
        return self._run_remdm(N, L, device, track=True)

    def _run_remdm(self, N, L, device, track=False):
        xt = torch.full((N, L), self.mask_token, device=device, dtype=torch.long)
        grid = torch.linspace(1.0, 0.0, self.steps + 1, device=device)
        tracker = SamplingTracker(self.steps, N, L, device) if track else None
        if tracker: tracker.record(0, xt)

        for k in range(self.steps):
            t, s = grid[k], grid[k+1]
            probs = self(xt).softmax(-1) # [N, L, 2]
            
            alpha_t, alpha_s = 1.0 - t, 1.0 - s
            sigma_max = min(1.0, (1.0 - alpha_s) / alpha_t) if alpha_t > 0 else 1.0
            sigma = self.eta * sigma_max if self.sampler == "remdm-rescale" else min(self.eta, sigma_max)

            q = torch.zeros(N, L, 3, device=device)
            is_masked = (xt == self.mask_token)
            denom = (1.0 - alpha_t).clamp_min(1e-8)
            
            # --- Handle Masked Tokens (Reveal step) ---
            if is_masked.any():
                q_masked = q[is_masked] # [M, 3]
                q_masked[:, :2] = probs[is_masked] * (alpha_s - (1.0 - sigma) * alpha_t) / denom
                q_masked[:, 2] = (1.0 - alpha_s - sigma * alpha_t) / denom
                q[is_masked] = q_masked

            # --- Handle Clean Tokens (Remasking / Stay step) ---
            is_clean = ~is_masked
            if is_clean.any():
                q_clean = q[is_clean] # [C, 3]
                q_clean[:, 2] = sigma
                # Map the (1 - sigma) probability directly to the held token index
                # This replaces the scatter_ call that caused the dimension error
                row_idx = torch.arange(q_clean.size(0), device=device)
                col_idx = xt[is_clean]
                q_clean[row_idx, col_idx] = 1.0 - sigma
                q[is_clean] = q_clean

            xt = torch.distributions.Categorical(probs=q.clamp_min(0)).sample()
            if tracker: tracker.record(k + 1, xt)
            
        return tracker.finalize(self.mask_token) if track else xt.clamp_max(1)