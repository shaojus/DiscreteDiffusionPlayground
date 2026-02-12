import math, torch, torch.nn as nn, torch.nn.functional as F

def _sinusoidal_positional_encoding(max_len: int, d_model: int, device=None):
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class PosEnc(nn.Module):
    def __init__(self, V: int, d_model: int, max_len: int):
        super().__init__()
        self.proj = nn.Linear(V, d_model, bias=False)
        self.register_buffer("pe", _sinusoidal_positional_encoding(max_len=max_len, d_model=d_model))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        T = X.size(0)
        H = self.proj(X)
        return H + self.pe[:T, :].unsqueeze(1)

class RandomMaskingTransformer(nn.Module):
    def __init__(
        self,
        d_model=128,
        nhead=4,
        d_hid=512,
        n_layers=6,
        dropout=0.2,
        max_len=16,
        mask_prob: float = 0.3,
        device=None,
    ):
        super().__init__()
        self.MASK = 2
        self.V = 3
        self.DATA_LEN = max_len
        self.mask_prob = float(mask_prob)

        self.register_buffer("oh", torch.eye(self.V, device=device))
        self.pos = PosEnc(self.V, d_model, max_len=max_len)

        layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=False)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.fc = nn.Linear(d_model, self.V)

    def forward(self, tok: torch.Tensor):
        emb = self.pos(self.oh[tok])          # [L, B, d_model]
        out = self.enc(emb)                   # [L, B, d_model]
        return self.fc(out)                   # [L, B, V]

    def training_loss(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: [B, L] with tokens in {0,1}
        B, L = batch.shape
        tok = batch.transpose(0, 1).contiguous()                     # [L, B]
        m = (torch.rand_like(tok, dtype=torch.float32) < self.mask_prob)  # [L, B] bool

        # ensure we always have at least one masked position for a valid loss
        if not m.any():
            # mask at least one random position
            ridx = torch.randint(0, L, (1,), device=tok.device)
            cidx = torch.randint(0, B, (1,), device=tok.device)
            m[ridx, cidx] = True

        tok_masked = tok.clone()
        tok_masked[m] = self.MASK

        logits = self.forward(tok_masked)                          # [L, B, V]

        # --- robust masking: flatten, then index ---
        V = logits.size(-1)
        logits_flat = logits.view(-1, V)                           # [L*B, V]
        tok_flat    = tok.view(-1)                                 # [L*B]
        m_flat      = m.view(-1)                                   # [L*B]

        pred   = logits_flat[m_flat][:, :2]                        # [M, 2] classes {0,1}
        target = tok_flat[m_flat].long()                           # [M] in {0,1}

        return F.cross_entropy(pred, target)

    @torch.no_grad()
    def sample(self, N: int, L: int, device: torch.device) -> torch.Tensor:
        return self.generate(batch=N, top_k=1).transpose(0, 1).contiguous()

    def generate(
        self,
        batch: int = 1,
        temp: float = 1.0,
        max_steps: int | None = None,
        record_order: bool = False,
        parallel: bool = False,
        top_k: int = 1,
    ):
        self.eval()
        SEQ_LEN = self.DATA_LEN
        if max_steps is None:
            max_steps = SEQ_LEN
        k = max(1, int(top_k))

        seq = torch.full((SEQ_LEN, batch), self.MASK, dtype=torch.long, device=self.oh.device)
        if record_order:
            order = torch.full_like(seq, -1)

        if parallel:
            logits = self(seq); logits[:, :, self.MASK] = -float("inf")
            probs = (logits / temp).softmax(-1)
            sampled = torch.multinomial(probs.reshape(-1, self.V), 1).view(SEQ_LEN, batch)
            seq = sampled
            if record_order: order[:] = 0
            return (seq, order) if record_order else seq

        for t in range(max_steps):
            masked = (seq == self.MASK)
            if not masked.any():
                break
            logits = self(seq); logits[:, :, self.MASK] = -float("inf")
            probs = (logits / temp).softmax(-1)
            conf, _ = probs.max(-1)
            conf = conf.masked_fill(~masked, -1.0)
            _, row_idx = conf.topk(k, dim=0)
            col_idx = torch.arange(batch, device=seq.device).repeat(k, 1)
            row_idx_flat = row_idx.flatten(); col_idx_flat = col_idx.flatten()
            keep = masked[row_idx_flat, col_idx_flat]
            if not keep.any():
                break
            row_idx_flat = row_idx_flat[keep]; col_idx_flat = col_idx_flat[keep]
            sel_probs = probs[row_idx_flat, col_idx_flat]
            sampled = torch.multinomial(sel_probs, 1).squeeze(-1)
            seq[row_idx_flat, col_idx_flat] = sampled
            if record_order: order[row_idx_flat, col_idx_flat] = t
        return (seq, order) if record_order else seq
