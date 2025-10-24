import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RADDEncoder(nn.Module):
    def __init__(
        self,
        V: int = 3,
        C: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        d_ff: int = 512,
        n_layers: int = 6,
        max_len: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.V = int(V)
        self.C = int(C)
        self.mask_id = self.V - 1

        # token emb
        self.tok_emb = nn.Embedding(self.V, d_model)

        # fixed sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        # vanilla TransformerEncoder (batch_first=True)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)

        # head to {0,1}
        self.fc = nn.Linear(d_model, self.C)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        x: (B, L) ints in {0,1,mask_id}
        returns: logits (B, L, C=2) for {0,1}
        """
        B, L = x.shape
        h = self.tok_emb(x) + self.pe[:, :L, :]
        h = self.enc(h)              # (B, L, d)
        logits = self.fc(h)          # (B, L, 2)
        return logits



def _forward_process_radd(x0: torch.LongTensor, mask_id: int, eps: float = 1e-3):
    """
    x0: (B,L) in {0,1}
    Returns:
      noisy:     (B,L) in {0,1,mask_id}
      mask_idx:  (B,L) bool
      p_mask:    (B,L) float per-token mask prob
      t:         (B,)   U[0,1]
    """
    B, L = x0.shape
    device = x0.device
    t = torch.rand(B, device=device)                         # t ~ U[0,1]
    p = ((1.0 - eps) * t + eps).unsqueeze(1).expand(B, L)    # per-token prob
    mask_idx = (torch.rand(B, L, device=device) < p)
    noisy = torch.where(mask_idx, torch.full_like(x0, mask_id), x0)
    return noisy, mask_idx, p, t


def _radd_loss_from_logits(
    logits: torch.Tensor,     # (B,L,2)
    x0: torch.LongTensor,     # (B,L) in {0,1}
    mask_idx: torch.Tensor,   # (B,L) bool
    p_mask: torch.Tensor      # (B,L) float
) -> torch.Tensor:
    """
    CE on masked positions only; each term divided by p_mask (importance), mean over all tokens.
    """
    B, L, C = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, C), x0.reshape(-1), reduction='none').reshape(B, L)
    per_token = (ce / p_mask) * mask_idx.float()
    loss = per_token.sum() / (B * L)
    return loss



@torch.no_grad()
def radd_sample(
    model: RADDEncoder,
    B: int,
    L: int,
    *,
    steps: int = 64,
    device: Optional[torch.device] = None,
    strategy: str = "vanilla",     # "vanilla" or "topk"
    k: Optional[int] = None,       # fixed K per sample per step (for "topk")
    k_frac: Optional[float] = None,# reveal ceil(k_frac * #masked) per step
    match_expected: bool = True,   # if no k/k_frac: K ≈ expected vanilla reveals
    temperature: float = 1.0,      # token sampling temperature
    topk_token: Optional[int] = None,  # token-level top-k truncation (None = off)
) -> torch.LongTensor:
    """
    Returns: (B, L) in {0,1}
    """
    device = device or next(model.parameters()).device
    MASK = model.mask_id
    xt = torch.full((B, L), MASK, dtype=torch.long, device=device)

    t_grid = torch.linspace(1.0, 0.0, steps + 1, device=device)

    def _sample_values(probs_slice: torch.Tensor) -> torch.LongTensor:
        # probs_slice: (N, 2)
        if temperature != 1.0:
            logits = torch.log(probs_slice.clamp_min(1e-12)) / temperature
            probs_t = torch.softmax(logits, dim=-1)
        else:
            probs_t = probs_slice

        if topk_token is not None and topk_token < probs_t.size(-1):
            topk_vals, topk_idx = torch.topk(probs_t, topk_token, dim=-1)
            probs_t = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
            picks_topk = torch.multinomial(probs_t, num_samples=1).squeeze(1)
            return topk_idx.gather(1, picks_topk.unsqueeze(1)).squeeze(1)
        else:
            return torch.multinomial(probs_t, num_samples=1).squeeze(1)

    for s in range(steps):
        t_s, t_s1 = t_grid[s], t_grid[s + 1]
        logits = model(xt)            # (B, L, 2)
        probs  = logits.softmax(-1)   # (B, L, 2)
        masked = (xt == MASK)         # (B, L)

        if not masked.any():
            continue

        if strategy == "vanilla":
            preveal = float((t_s - t_s1) / (t_s + 1e-12))
            Bidx, Lidx = masked.nonzero(as_tuple=True)
            flags = torch.bernoulli(torch.full((Bidx.numel(),), preveal, device=device)).bool()
            if flags.any():
                Br, Lr = Bidx[flags], Lidx[flags]
                picks = _sample_values(probs[Br, Lr])
                xt[Br, Lr] = picks

        elif strategy == "topk":
            preveal = float((t_s - t_s1) / (t_s + 1e-12))  # used when match_expected=True
            cert = probs.max(dim=-1).values  # certainty = max prob ∈ [0,1], shape (B,L)
            for b in range(B):
                masked_b = masked[b]                           # (L,)
                n_masked = int(masked_b.sum().item())
                if n_masked == 0:
                    continue

                # choose K
                if k is not None:
                    K = min(k, n_masked)
                elif k_frac is not None:
                    K = max(1, min(n_masked, int(torch.ceil(torch.tensor(k_frac * n_masked)).item())))
                elif match_expected:
                    K = max(1, min(n_masked, int(torch.ceil(torch.tensor(preveal * n_masked)).item())))
                else:
                    K = 1

                # pick K lowest certainty (most uncertain)
                cert_b = cert[b]
                cert_masked = cert_b.clone()
                cert_masked[~masked_b] = float('inf')

                if K == n_masked:
                    choose_idx = torch.nonzero(masked_b, as_tuple=False).squeeze(1)
                else:
                    noise = torch.empty_like(cert_masked).uniform_(0, 1e-6)
                    cert_noisy = cert_masked + noise
                    choose_idx = torch.topk(cert_noisy, k=K, largest=False).indices  # (K,)

                picks = _sample_values(probs[b, choose_idx])     # (K,)
                xt[b, choose_idx] = picks
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # final cleanup (rare): fill any leftover MASKs by argmax
    left = (xt == MASK)
    if left.any():
        xt = torch.where(left, model(xt).argmax(-1), xt)

    return xt.clamp_max(1)  # ensure bits ∈ {0,1}


class RADD(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        d_ff: int = 512,
        n_layers: int = 6,
        dropout: float = 0.0,
        max_len: int = 16,
        # forward masking schedule
        eps: float = 1e-3,
        # sampling controls
        samp_steps: int = 64,
        samp_strategy: str = "topk",          # "vanilla" or "topk"
        samp_k: Optional[int] = None,
        samp_k_frac: Optional[float] = 0.2,
        samp_match_expected: bool = True,
        samp_temperature: float = 1.0,
        samp_topk_token: Optional[int] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__()
        self.encoder = RADDEncoder(
            V=3, C=2, d_model=d_model, nhead=nhead, d_ff=d_ff,
            n_layers=n_layers, max_len=max_len, dropout=dropout
        )
        self.mask_id = self.encoder.mask_id

        # schedule + sampler cfg
        self.eps = float(eps)
        self.samp_steps = int(samp_steps)
        self.samp_strategy = str(samp_strategy)
        self.samp_k = samp_k
        self.samp_k_frac = samp_k_frac
        self.samp_match_expected = bool(samp_match_expected)
        self.samp_temperature = float(samp_temperature)
        self.samp_topk_token = samp_topk_token

        if device is not None:
            self.to(device)

    def training_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """
        batch: (B,L) bits {0,1}
        """
        noisy, mask_idx, p_mask, _ = _forward_process_radd(batch, self.mask_id, eps=self.eps)
        logits = self.encoder(noisy)
        loss = _radd_loss_from_logits(logits, batch, mask_idx, p_mask)
        return loss

    @torch.no_grad()
    def sample(self, N: int, L: int, device: torch.device) -> torch.Tensor:
        xt = radd_sample(
            self.encoder, B=N, L=L, device=device,
            steps=self.samp_steps, strategy=self.samp_strategy,
            k=self.samp_k, k_frac=self.samp_k_frac,
            match_expected=self.samp_match_expected,
            temperature=self.samp_temperature,
            topk_token=self.samp_topk_token,
        )
        return xt
