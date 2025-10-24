import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.graph_lib import Absorbing  

def _sinusoidal_positional_encoding(max_len: int, d_model: int, device=None):
    pe = torch.zeros(max_len, d_model, device=device)
    pos = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class PosEnc(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int):
        super().__init__()
        self.proj = nn.Linear(vocab_size, d_model, bias=False)
        self.register_buffer("pe", _sinusoidal_positional_encoding(max_len, d_model))

    def forward(self, onehot_TBV: torch.Tensor) -> torch.Tensor:
        # onehot_TBV: [T,B,V]
        T = onehot_TBV.size(0)
        h = self.proj(onehot_TBV)  # [T,B,D]
        return h + self.pe[:T, :].unsqueeze(1)  # [T,B,D]

class LogLinearNoise(nn.Module):
    """
    sigma(t) = -log(1 - (1 - eps) * t)
    d sigma / dt = (1 - eps) / (1 - (1 - eps) * t)
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = float(eps)
        self._anchor = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def rate_noise(self, t):
        return (1.0 - self.eps) / (1.0 - (1.0 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1.0 - self.eps) * t)


class SigmaCond(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, sigma_b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.log1p(sigma_b).unsqueeze(-1))  # [B,1] -> [B,D]


class SEDDEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model=128, nhead=4, d_ff=512, n_layers=6, max_len=16, dropout=0.0):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pos = PosEnc(self.vocab_size, d_model, max_len)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, batch_first=False)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.fc = nn.Linear(d_model, self.vocab_size)
        self.sig_emb = SigmaCond(d_model)

    def forward(self, tokens_BT: torch.LongTensor, sigma_B: torch.Tensor):
        # tokens_BT: [B,T] with ids in [0, V-1]
        onehot = F.one_hot(tokens_BT, num_classes=self.vocab_size).float()     # [B,T,V]
        x = self.pos(onehot.transpose(0, 1).contiguous())                      # [T,B,D]
        s = self.sig_emb(sigma_B).unsqueeze(0)                                 # [1,B,D]
        h = self.enc(x + s)                                                    # [T,B,D]
        return self.fc(h).transpose(0, 1).contiguous()                         # [B,T,V]


def sedd_absorb_loss(graph: Absorbing,
                     model: SEDDEncoder,
                     x0_bt: torch.LongTensor,
                     noise: LogLinearNoise,
                     t_eps: float = 1e-3):
    device = x0_bt.device
    B, T = x0_bt.shape

    # sample times
    t = (1.0 - t_eps) * torch.rand(B, device=device)
    sigma_b = noise.total_noise(t)     # [B]
    dsigma_b = noise.rate_noise(t)     # [B]

    # graph in [T,B]
    x0_tb = x0_bt.transpose(0, 1).contiguous()                  # [T,B]
    x_t_tb = graph.sample_transition(x0_tb, sigma_b)            # [T,B] in {0,1,mask}
    x_t_bt = x_t_tb.transpose(0, 1).contiguous()                # [B,T] for encoder

    logits_btv = model(x_t_bt, sigma_b)                         # [B,T,V]
    logits_tbv = logits_btv.transpose(0, 1).contiguous()        # [T,B,V]

    # graph.score_entropy returns per-token loss [T,B]
    loss_tb = graph.score_entropy(
        score=logits_tbv,
        sigma=sigma_b.unsqueeze(0),  # [1,B] -> broadcasts over T
        x=x_t_tb,
        x0=x0_tb
    )  # [T,B]

    loss_bt = loss_tb.transpose(0, 1).contiguous()              # [B,T]
    loss_b = (dsigma_b[:, None] * loss_bt).sum(dim=-1)          # [B]
    return loss_b.mean(), {
        "mask_frac": float((x_t_bt == getattr(graph, "mask_id", int(getattr(graph, "dim", 3) - 1))).float().mean().item()),
        "t_mean": float(t.mean().item()),
    }


@torch.no_grad()
def euler_absorb_sampler(model: SEDDEncoder,
                         graph: Absorbing,
                         T: int,
                         B: int = 64,
                         device: Optional[torch.device] = None,
                         t_steps: int = 128,
                         temperature: float = 1.0,
                         rate_scale: Optional[float] = None,
                         noise: Optional[LogLinearNoise] = None) -> torch.LongTensor:
    """
    Reverse CTMC via Euler steps in t ∈ [1→0]; operates in graph space [T,B],
    uses encoder in [B,T]. Final return is bits in {0,1} of shape [B,T].
    """
    device = device or next(model.parameters()).device
    vocab_size = int(getattr(graph, "dim", 3))
    mask_id = int(getattr(graph, "mask_id", vocab_size - 1))
    m = vocab_size - 1
    rate_scale = float(rate_scale) if rate_scale is not None else float(T)
    noise = (noise or LogLinearNoise()).to(device)

    x_tb = graph.sample_limit(T, B).to(device)

    t_grid = torch.linspace(1.0, 0.0, t_steps + 1, device=device)

    for k in range(t_steps):
        t_hi, t_lo = t_grid[k], t_grid[k + 1]
        dt = float(t_hi - t_lo)

        sigma = noise.total_noise(torch.full((B,), t_hi, device=device))  # [B]

        # Encoder in [B,T]
        x_bt = x_tb.transpose(0, 1).contiguous()
        logits_btv = model(x_bt, sigma)                                   # [B,T,V]
        probs = torch.softmax(logits_btv[..., :m] / max(1e-8, float(temperature)), dim=-1)  # [B,T,m]

        # certainty -> rate
        H = -(probs * (probs.clamp_min(1e-12)).log()).sum(-1)             # [B,T]
        r_tb = (rate_scale * (1.0 - H).clamp_min(0.0)).transpose(0, 1).contiguous()  # [T,B]

        masked_tb = (x_tb == mask_id)
        if masked_tb.any():
            p_flip_tb = (-(r_tb * dt).clamp(min=-50.0)).exp().neg().add(1.0)        # 1 - exp(-r*dt)
            flips_tb = (torch.rand_like(p_flip_tb) < p_flip_tb) & masked_tb
            if flips_tb.any():
                probs_tb = probs.transpose(0, 1).contiguous()                        # [T,B,m]
                sampled = torch.distributions.Categorical(probs_tb[flips_tb]).sample()
                x_tb = x_tb.clone()
                x_tb[flips_tb] = sampled

        if (x_tb != mask_id).all():
            break

    # final fill
    left_tb = (x_tb == mask_id)
    if left_tb.any():
        sigma0 = noise.total_noise(torch.zeros(B, device=device))
        p0 = torch.softmax(model(x_tb.transpose(0, 1).contiguous(), sigma0)[..., :m], dim=-1)  # [B,T,m]
        p0_tb = p0.transpose(0, 1).contiguous()
        x_tb[left_tb] = p0_tb[left_tb].argmax(-1)

    return x_tb.transpose(0, 1).contiguous().clamp_max(1)  # [B,T] in {0,1}


class SEDD(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 nhead: int = 4,
                 d_ff: int = 512,
                 n_layers: int = 6,
                 dropout: float = 0.0,
                 max_len: int = 16,
                 noise_eps: float = 1e-3,
                 t_eps: float = 1e-3,
                 t_steps: int = 128,
                 temperature: float = 1.0,
                 rate_scale: Optional[float] = None,
                 device: Optional[torch.device] = None,
                 **kwargs):
        super().__init__()
        self.graph = Absorbing(dim=3)  # {0,1,MASK} as in your code
        self.vocab_size = int(getattr(self.graph, "dim", 3))
        self.mask_id = int(getattr(self.graph, "mask_id", self.vocab_size - 1))

        self.model = SEDDEncoder(vocab_size=self.vocab_size,
                                 d_model=d_model, nhead=nhead, d_ff=d_ff,
                                 n_layers=n_layers, max_len=max_len, dropout=dropout)

        self.noise = LogLinearNoise(eps=noise_eps)
        self.t_eps = float(t_eps)
        self.t_steps = int(t_steps)
        self.temperature = float(temperature)
        self.rate_scale = rate_scale

        if device is not None:
            self.to(device)

    def training_loss(self, batch: torch.Tensor) -> torch.Tensor:
        loss, _ = sedd_absorb_loss(self.graph, self.model, batch, self.noise, t_eps=self.t_eps)
        return loss

    @torch.no_grad()
    def sample(self, N: int, L: int, device: torch.device) -> torch.Tensor:
        toks = euler_absorb_sampler(self.model, self.graph, T=L, B=N, device=device,
                                    t_steps=self.t_steps, temperature=self.temperature,
                                    rate_scale=self.rate_scale, noise=self.noise)
        return toks.clamp_max(1)

