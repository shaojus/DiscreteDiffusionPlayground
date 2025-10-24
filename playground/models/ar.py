import torch, torch.nn as nn, torch.nn.functional as F

class AutoregressiveTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, d_hid=512, n_layers=6, dropout=0.0, max_len=32, device=None):
        super().__init__()
        self.V = 2
        self.tok = nn.Embedding(self.V, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.proj = nn.Linear(d_model, self.V)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        h = self.enc(h, mask)
        return self.proj(h)

    def training_loss(self, batch: torch.Tensor) -> torch.Tensor:
        logits = self.forward(batch[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))

    @torch.no_grad()
    def sample(self, N: int, L: int, device: torch.device) -> torch.Tensor:
        x = torch.zeros(N, L, dtype=torch.long, device=device)
        for t in range(L):
            logits = self.forward(x[:, : t+1])[:, -1, :]
            probs = logits.softmax(-1)
            x[:, t] = torch.multinomial(probs, 1).squeeze(-1)
        return x
