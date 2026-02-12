import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PosEnc(nn.Module):
    def __init__(self, V, d_model, max_len):
        super().__init__()
        self.proj = nn.Linear(V, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(pos * div)
        pe[:, 0, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, onehot_TBV):  # (T,B,V)
        return self.proj(onehot_TBV) + self.pe[: onehot_TBV.size(0)]


class AutoregressiveTransformer(nn.Module):
    """
      - forward:   x0, x1, x2, ...
      - reverse:   x_{L-1}, ..., x1, x0
      - interleave (assumes original layout is [x0..x_{n-1}, y0..y_{n-1}]):
                   x0, y0, x1, y1, ...
    """
    def __init__(
        self,
        d_model=128,
        nhead=4,
        d_hid=512,
        n_layers=6,
        dropout=0.0,
        max_len=33,   # for L tokens, set max_len = L+1 (BOS + L)
        V=3,
        BOS=2,
        order: str = "forward",  # "forward" | "reverse" | "interleave"
        **kwargs
    ):
        super().__init__()
        self.V = V
        self.BOS = BOS
        self.order = order

        self.register_buffer("oh", torch.eye(self.V))  # (V,V)
        self.pos = PosEnc(V=self.V, d_model=d_model, max_len=max_len)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid,
            dropout=dropout, batch_first=False
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.fc  = nn.Linear(d_model, self.V)

    @staticmethod
    def _causal_mask(T, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, tok_TB):  # ints in {0,1,2}, shape (T,B)
        onehot_TBV = self.oh[tok_TB]           # (T,B,V)
        h = self.pos(onehot_TBV)               # (T,B,D)
        T, B, D = h.shape
        mem = torch.zeros(1, B, D, device=h.device, dtype=h.dtype)  # dummy memory
        out = self.dec(h, mem, tgt_mask=self._causal_mask(T, h.device))  # (T,B,D)
        return self.fc(out)                    # (T,B,V)

    @staticmethod
    def _perm_indices(L: int, order: str, device: torch.device) -> torch.Tensor:
        """
        Returns permmuted indices p of shape (L,), so x_perm = x[:, p].
        """
        if order == "forward":
            p = torch.arange(L, device=device)
        elif order == "reverse":
            p = torch.arange(L - 1, -1, -1, device=device)
        elif order == "interleave":
            if L % 2 != 0:
                raise ValueError(f"interleave requires even L, got L={L}")
            n = L // 2
            p = torch.empty(L, dtype=torch.long, device=device)
            p[0::2] = torch.arange(0, n, device=device)         # x part
            p[1::2] = torch.arange(n, 2 * n, device=device)     # y part
        else:
            raise ValueError(f"Unknown order={order!r}. Use 'forward','reverse','interleave'.")
        return p

    @staticmethod
    def _invert_perm(p: torch.Tensor) -> torch.Tensor:
        """
        Given p where x_perm = x[:, p], returns inv such that x = x_perm[:, inv].
        """
        inv = torch.empty_like(p)
        inv[p] = torch.arange(p.numel(), device=p.device)
        return inv


    def training_loss(self, batch_bits: torch.Tensor) -> torch.Tensor:
        """
        batch_bits: (B,L) with targets in {0,1} (or generally < V, excluding BOS).
        """
        B, L = batch_bits.shape
        dev = batch_bits.device

        p = self._perm_indices(L, self.order, dev)
        tgt = batch_bits[:, p]  # (B,L) in the chosen order

        # inputs with BOS + shifted targets
        x_in = torch.full((B, 1), self.BOS, dtype=torch.long, device=dev)
        x_in = torch.cat([x_in, tgt[:, :-1]], dim=1)  # (B,L)

        logits_TBV = self.forward(x_in.transpose(0, 1).contiguous())  # (L,B,V)
        logits_BTV = logits_TBV.transpose(0, 1).contiguous()          # (B,L,V)

        loss = F.cross_entropy(
            logits_BTV.reshape(-1, self.V),
            tgt.reshape(-1),
            reduction="mean",
        )
        return loss

    @torch.no_grad()
    def sample(self, N: int, L: int, device: torch.device) -> torch.Tensor:
        self.eval()

        p = self._perm_indices(L, self.order, device)
        inv = self._invert_perm(p)

        # generate in permuted order
        seq_TB = torch.full((1, N), self.BOS, dtype=torch.long, device=device)  # (1,N)
        for _ in range(L):
            logits_TBV = self.forward(seq_TB)      # (t,N,V)
            logits_last_NV = logits_TBV[-1]        # (N,V)

            logits_last_NV = logits_last_NV.clone()
            logits_last_NV[:, self.BOS] = -float("inf")  # forbid BOS

            probs = logits_last_NV.softmax(-1)
            nxt = torch.multinomial(probs, 1).squeeze(-1)  # (N,)
            seq_TB = torch.cat([seq_TB, nxt.unsqueeze(0)], dim=0)

        # drop BOS -> (L,N) -> (N,L) permuted layout
        perm_samples = seq_TB[1:, :].transpose(0, 1).contiguous()  # (N,L)

        # unpermute back to original layout
        samples = perm_samples[:, inv].contiguous()
        return samples
