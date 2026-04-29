import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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

class MDLM(nn.Module):
    def __init__(self, d_model=128, nhead=4, d_hid=512, n_layers=6, dropout=0.1, max_len=16, mask_token=2, steps=16, **kwargs):
        super().__init__()
        self.mask_token = mask_token
        self.steps = steps
        self.emb = nn.Embedding(3, d_model)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model // nhead, 2).float() / (d_model // nhead)))
        self.register_buffer("inv_freq", inv_freq)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': RoPEAttention(d_model, nhead, dropout),
                'norm1': nn.LayerNorm(d_model),
                'mlp': nn.Sequential(nn.Linear(d_model, d_hid), nn.GELU(), nn.Linear(d_hid, d_model), nn.Dropout(dropout)),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, 2)

    def _get_rope(self, x):
        t = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0).unsqueeze(1)
        return emb.cos(), emb.sin()

    def forward(self, x):
        h = self.emb(x)
        cos, sin = self._get_rope(h)
        for layer in self.layers:
            h = h + layer['attn'](layer['norm1'](h), cos, sin)
            h = h + layer['mlp'](layer['norm2'](h))
        return self.fc(h)

    def training_loss(self, x0):
        B, L = x0.shape
        t = torch.rand(B, device=x0.device)
        mask = torch.rand((B, L), device=x0.device) < t.view(B, 1)
        xt = x0.clone()
        xt[mask] = self.mask_token
        logits = self(xt)
        loss = F.cross_entropy(logits.transpose(1, 2), x0, reduction='none')
        return (loss * mask.float()).sum() / (B * L * t.view(B, 1)).mean().clamp_min(1e-6)

    @torch.no_grad()
    def sample(self, N, L, device):
        xt = torch.full((N, L), self.mask_token, device=device)
        ts = torch.linspace(1.0, 0.0, self.steps + 1, device=device)
        for i in range(self.steps):
            probs = self(xt).softmax(-1)
            p_unmask = (ts[i] - ts[i+1]) / ts[i].clamp_min(1e-8)
            mask_pos = (xt == self.mask_token)
            to_unmask = (torch.rand((N, L), device=device) < p_unmask) & mask_pos
            if to_unmask.any():
                xt[to_unmask] = torch.distributions.Categorical(probs=probs).sample()[to_unmask]
        return xt