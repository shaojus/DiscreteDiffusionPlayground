import math, torch, torch.nn as nn, torch.nn.functional as F

def _sinusoidal_positional_encoding(max_len: int, d_model: int, device=None):
    pe = torch.zeros(max_len, d_model, device=device)
    pos = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

class _PosEnc(nn.Module):
    def __init__(self, V: int, d_model: int, max_len: int):
        super().__init__()
        self.proj = nn.Linear(V, d_model, bias=False)
        self.register_buffer("pe", _sinusoidal_positional_encoding(max_len=max_len, d_model=d_model))
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        T = X.size(0); H = self.proj(X); return H + self.pe[:T, :].unsqueeze(1)

def _md4_alpha_q_w(t: torch.Tensor):
    one_minus_t = (1.0 - t)
    alpha = 1.0 - torch.cos(0.5 * math.pi * one_minus_t)
    dalpha = 0.5 * math.pi * torch.sin(0.5 * math.pi * one_minus_t)
    denom = (1.0 - alpha).clamp_min(1e-8)
    w = dalpha.abs() / denom
    q = denom
    return alpha, q, w

class MD4(nn.Module):
    def __init__(self, d_model=128, nhead=4, d_hid=512, n_layers=6, dropout=0.2, max_len=16, use_time_cond: bool=False, steps: int=64, mask_token: int=2, device=None):
        super().__init__()
        self.V = 3; self.C = 2
        self.mask_token = int(mask_token)
        self.steps = int(steps)
        self.use_time_cond = bool(use_time_cond)
        self.register_buffer("oh", torch.eye(self.V, device=device))
        self.pos = _PosEnc(self.V, d_model, max_len=max_len)
        if self.use_time_cond:
            self.time_mlp = nn.Sequential(nn.Linear(256, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=False)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.fc = nn.Linear(d_model, self.C)

    def _time_embed(self, t: torch.Tensor, d_model: int):
        half = 256 // 2
        freqs = torch.exp(torch.linspace(0, -math.log(10000), steps=half, device=t.device))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if 256 % 2 == 1: emb = F.pad(emb, (0,1))
        return self.time_mlp(emb)

    def forward(self, tokens: torch.LongTensor, t: torch.Tensor | None = None):
        h = self.pos(self.oh[tokens])
        if self.use_time_cond and t is not None:
            t_emb = self._time_embed(t, h.size(-1))
            h = h + t_emb[None, :, :]
        h = self.enc(h)
        return self.fc(h)

    def training_loss(self, batch: torch.Tensor) -> torch.Tensor:
        loss, _ = self._compute_loss_md4(batch)
        return loss

    @torch.no_grad()
    def sample(self, N: int, L: int, device: torch.device) -> torch.Tensor:
        return self._reverse_sample(L=L, B=N, steps=self.steps, device=device)

    def _compute_loss_md4(self, x0_bt: torch.LongTensor, *, t_eps: float = 0.07, add_recon_const: bool = True):
        device = x0_bt.device
        B, T = x0_bt.shape
        t = (1.0 - 2*t_eps) * torch.rand(B, device=device) + t_eps
        alpha, q, w_t = _md4_alpha_q_w(t)
        mask = (torch.rand(B, T, device=device) < q.unsqueeze(1))
        x_t = x0_bt.clone(); x_t[mask] = self.mask_token
        logits_tbc = self(x_t.t(), t=None)
        logits_btc = logits_tbc.transpose(0, 1).contiguous()
        p1 = logits_btc.log_softmax(dim=-1).exp()[..., 1]
        if not mask.any():
            zero = logits_btc.sum() * 0.0
            return zero, {'mean_loss': 0.0, 'mask_frac': 0.0}
        targets = x0_bt.float()
        bce = torch.nn.functional.binary_cross_entropy(p1[mask], targets[mask], reduction='none')
        w_tok = w_t.unsqueeze(1).expand_as(mask)[mask]
        loss_main = (w_tok * bce).mean()
        if add_recon_const:
            seq_len = float(B * T); alpha0 = 1.0
            recon = seq_len * (1 - alpha0) * math.log(2)
            loss = loss_main / math.log(2) + (recon / seq_len)
        else:
            loss = loss_main
        return loss, {}

    @torch.no_grad()
    def _reverse_sample(self, *, L: int, B: int, steps: int, device: torch.device) -> torch.LongTensor:
        xt = torch.full((B, L), self.mask_token, device=device, dtype=torch.long)
        grid = torch.linspace(0, 1, steps + 1, device=device)
        for i in range(steps, 0, -1):
            ti = torch.full((B,), grid[i], device=device)
            si = torch.full((B,), grid[i - 1], device=device)
            logits_tbc = self(xt.t(), t=None)
            logits_btc = logits_tbc.transpose(0, 1)
            p1 = logits_btc.log_softmax(dim=-1).exp()[..., 1]
            alphat, _, _ = _md4_alpha_q_w(ti); alphas, _, _ = _md4_alpha_q_w(si)
            p_unmask = ((alphas - alphat) / (1.0 - alphat + 1e-8)).unsqueeze(1).expand(B, L)
            masked_now = (xt == self.mask_token)
            to_unmask = (torch.rand(B, L, device=device) < p_unmask) & masked_now
            new_bits = (torch.rand_like(p1) < p1).long()
            xt[to_unmask] = new_bits[to_unmask]
        xt[xt == self.mask_token] = 0
        return xt
