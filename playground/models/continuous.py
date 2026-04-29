import math, torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def cosine_betas(T=1000, s=0.008, device=None):
    t = torch.arange(T+1, dtype=torch.float32, device=device)
    f = torch.cos((t/T + s) / (1+s) * math.pi/2)**2
    a_bar = (f / f[0])[1:]
    betas = 1 - (a_bar[1:] / a_bar[:-1])
    betas = torch.clamp(betas, 1e-5, 0.999)
    betas = torch.cat([betas, betas[-1:]], dim=0)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

def bits_to_onehot(x_bits):
    return F.one_hot(x_bits.long(), num_classes=2).float()


class DiffusionDDPM:
    def __init__(self, T=1000, device="cpu"):
        self.T = int(T)
        self.device = torch.device(device)
        b, a, ab = cosine_betas(self.T, device=self.device)
        self.betas = b
        self.alphas = a
        self.alpha_bars = ab
        
        # Precompute DDPM coefficients
        self.sqrt_alpha_bars = torch.sqrt(ab)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - ab)
        self.sqrt_alphas = torch.sqrt(a)
        self.sqrt_betas = torch.sqrt(b)

    def _ab(self, t):
        return self.alpha_bars[t].view(-1, 1, 1)

    @torch.no_grad()
    def q_sample(self, x0, t, noise=None):
        """Forward process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise

    @torch.no_grad()
    def p_step(self, model, x_t, step):
        """
        DDPM reverse step: p(x_{t-1} | x_t)
        
        mean = (1/sqrt(α_t)) * (x_t - (β_t / sqrt(1 - αbar_t)) * ε_hat)
        x_{t-1} = mean + sqrt(β_t) * z   (for t > 0)
        """
        B = x_t.size(0)
        device = x_t.device
        t = torch.full((B,), step, device=device, dtype=torch.long)

        # Model predicts noise
        eps_hat, _ = model(x_t, t)

        # Get coefficients
        alpha_t = self.alphas[step]
        beta_t = self.betas[step]
        sqrt_alpha_t = self.sqrt_alphas[step]
        sqrt_one_minus_ab_t = self.sqrt_one_minus_alpha_bars[step]

        # Compute mean
        # μ = (1/sqrt(α_t)) * (x_t - (β_t / sqrt(1 - αbar_t)) * ε_hat)
        coef = beta_t / sqrt_one_minus_ab_t
        mean = (1.0 / sqrt_alpha_t) * (x_t - coef * eps_hat)

        if step == 0:
            return mean

        # Add noise for t > 0
        z = torch.randn_like(x_t)
        sigma = self.sqrt_betas[step]
        x_prev = mean + sigma * z
        return x_prev

    @torch.no_grad()
    def sample_probs(self, model, n, seq_len, device=None):
        """
        Run full DDPM reverse process, return softmax probabilities.
        """
        device = device or next(model.parameters()).device
        x = torch.randn(n, seq_len, 2, device=device)

        for t in reversed(range(self.T)):
            x = self.p_step(model, x, t)

        return F.softmax(x, dim=-1)


class TimeEmbed(nn.Module):
    def __init__(self, T, d=64):
        super().__init__()
        self.emb = nn.Embedding(T, d)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):
        return self.proj(self.emb(t))


class TinyTransformer2D(nn.Module):
    """
    Now predicts (eps_hat, logits) for ε-prediction training.
    """
    def __init__(self, seq_len, T, d_model=256, nhead=4, nlayers=6, d_ff=512):
        super().__init__()
        self.time = TimeEmbed(T, d=64)
        self.inp = nn.Linear(2 + 64, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, activation="gelu", batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc, nlayers)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
        self.out_eps = nn.Linear(d_model, 2)      # predict noise
        self.out_logits = nn.Linear(d_model, 2)   # aux CE head

    def forward(self, xt, t):
        B, L, _ = xt.shape
        ht = self.time(t).unsqueeze(1).expand(B, L, -1)
        h = self.inp(torch.cat([xt, ht], dim=-1)) + self.pos[:, :L, :]
        h = self.enc(h)
        return self.out_eps(h), self.out_logits(h)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = [p.detach().clone().to(p.device) for p in model.parameters()]
        self.params = list(model.parameters())

    def update(self):
        for s, p in zip(self.shadow, self.params):
            if s.device != p.device:
                s.data = s.data.to(p.device)
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model_out):
        for p, s in zip(model_out.parameters(), self.shadow):
            p.data.copy_(s.data)


class ContinuousDiffusion(nn.Module):
    """
    Standard DDPM with ε-prediction.
    
    Unified-trainer compatible:
      - training_loss(batch_bits) -> scalar loss
      - sample(N, L, device) -> (N, L) bits in {0,1}
    """
    def __init__(
        self,
        T: int = 1000,
        d_model: int = 256,
        nhead: int = 4,
        n_layers: int = 6,
        d_hid: int = 512,
        ce_weight: float = 0.5,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        **kwargs,
    ):
        super().__init__()
        self.T = int(T)
        self.ce_weight = float(ce_weight)
        self.use_ema = bool(use_ema)

        self._max_len = kwargs.get("max_len", 1024)

        self.diff = None
        self.net = TinyTransformer2D(
            seq_len=self._max_len,
            T=self.T,
            d_model=d_model,
            nhead=nhead,
            nlayers=n_layers,
            d_ff=d_hid,
        )
        self._ema = EMA(self.net, decay=ema_decay) if use_ema else None

    def _ensure_diff(self, device):
        if (self.diff is None) or (self.diff.device != device):
            self.diff = DiffusionDDPM(T=self.T, device=device)

    def training_loss(self, batch_bits):
        """
        Standard DDPM ε-prediction loss: E[||ε - ε_hat||²]
        
        batch_bits: (B, L) integers in {0,1}
        """
        device = batch_bits.device
        B, L = batch_bits.shape
        self._ensure_diff(device)

        x0 = bits_to_onehot(batch_bits)  # (B, L, 2)

        # Uniform t sampling (standard DDPM)
        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)

        # Sample noise and create x_t
        eps = torch.randn_like(x0)
        xt = self.diff.q_sample(x0, t, noise=eps)

        # Predict noise
        eps_hat, logits = self.net(xt[:, :self._max_len], t)
        eps_hat = eps_hat[:, :L, :]
        logits = logits[:, :L, :]

        # Simple MSE loss on noise prediction
        loss_eps = F.mse_loss(eps_hat, eps)

        ab = self.diff._ab(t)
        with torch.no_grad():
            mask_b = (ab.squeeze(-1).squeeze(-1) > 0.5).float()
        ce = F.cross_entropy(
            logits.reshape(-1, 2),
            x0.argmax(dim=-1).reshape(-1),
            reduction='none'
        ).reshape(B, L)
        ce = (ce * mask_b.unsqueeze(1)).sum() / (mask_b.sum() * L + 1e-8)

        loss = loss_eps + self.ce_weight * ce

        if self._ema is not None and self.training:
            self._ema.update()

        return loss

    @torch.no_grad()
    def sample(self, N, L, device):
        """
        Run full DDPM reverse process, return hard bits {0,1} of shape (N, L).
        """
        self.eval()
        device = torch.device(device)
        self._ensure_diff(device)

        net = self.net
        if self._ema is not None:
            tmp = copy.deepcopy(self.net).to(device)
            tmp.eval()
            self._ema.copy_to(tmp)
            net = tmp

        probs = self.diff.sample_probs(net, n=N, seq_len=L, device=device)
        bits = probs.argmax(dim=-1)
        return bits