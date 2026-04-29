import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sample_categorical(probs: torch.Tensor) -> torch.LongTensor:
    """Sample from probs along the last dimension."""
    probs = probs.clamp_min(0.0)
    denom = probs.sum(dim=-1, keepdim=True)
    bad = denom <= 0
    if bad.any():
        probs = probs.clone()
        probs[bad.expand_as(probs)] = 0.0
        probs[..., -1] = torch.where(bad.squeeze(-1), torch.ones_like(probs[..., -1]), probs[..., -1])
        denom = probs.sum(dim=-1, keepdim=True)
    probs = probs / denom.clamp_min(1e-12)
    flat = probs.reshape(-1, probs.shape[-1])
    samp = torch.distributions.Categorical(probs=flat).sample()
    return samp.view(*probs.shape[:-1])


class AbsorbingGraph:
    """
    Minimal absorbing graph matching the official SEDD graph_lib behavior.

    Pass dim=<number of non-mask classes>. For binary bits, dim=2 so the total
    state space is {0, 1, MASK}.
    """

    def __init__(self, dim: int):
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim + 1

    @property
    def absorb(self) -> bool:
        return True

    @property
    def mask_id(self) -> int:
        return self.dim - 1

    def transp_rate(self, i: torch.LongTensor) -> torch.Tensor:
        edge = -F.one_hot(i, num_classes=self.dim).to(torch.float32)
        edge = edge.to(dtype=torch.float32)
        mask = i == self.mask_id
        if mask.any():
            edge[mask] += 1.0
        return edge

    def reverse_rate(self, i: torch.LongTensor, score: torch.Tensor) -> torch.Tensor:
        normalized_rate = self.transp_rate(i).to(score) * score
        normalized_rate = normalized_rate.scatter(-1, i[..., None], torch.zeros_like(normalized_rate[..., :1]))
        normalized_rate = normalized_rate.scatter(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def transp_transition(self, i: torch.LongTensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma[:, None, None]
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim).to(sigma)
        add_mask = torch.where(i == self.mask_id, 1.0 - (-sigma.squeeze(-1)).exp(), 0.0)[..., None]
        edge = edge + add_mask
        return edge

    def sample_transition(self, i: torch.LongTensor, sigma: torch.Tensor) -> torch.LongTensor:
        # sigma: [B,1] or [B]
        if sigma.ndim == 1:
            sigma = sigma[:, None]
        move_chance = 1.0 - (-sigma).exp()
        move_indices = torch.rand_like(i, dtype=torch.float32) < move_chance
        return torch.where(move_indices, torch.full_like(i, self.mask_id), i)

    def staggered_score(self, score: torch.Tensor, dsigma: torch.Tensor) -> torch.Tensor:
        # Official SEDD graph_lib behavior for absorbing graphs.
        score = score.clone()
        extra_const = (1.0 - dsigma.exp())[:, None] * score.sum(dim=-1)
        score = score * dsigma.exp()[:, None, None]
        score[..., -1] += extra_const
        return score

    def sample_rate(self, i: torch.LongTensor, rate: torch.Tensor) -> torch.LongTensor:
        probs = F.one_hot(i, num_classes=self.dim).to(rate) + rate
        return _sample_categorical(probs)

    def sample_limit(self, *batch_dims: int) -> torch.LongTensor:
        return torch.full(batch_dims, self.mask_id, dtype=torch.int64)

    def score_entropy(
        self,
        score: torch.Tensor,
        sigma: torch.Tensor,
        x: torch.LongTensor,
        x0: torch.LongTensor,
    ) -> torch.Tensor:
        # Direct port of the official Absorbing.score_entropy formula.
        if sigma.ndim == 1:
            sigma = sigma[:, None]
        rel_ind = x == self.mask_id
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1.0)
        ratio = 1.0 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)
        const = ratio * (ratio.log() - 1.0)

        entropy = torch.zeros_like(x, dtype=score.dtype)
        entropy[rel_ind] = pos_term - neg_term + const
        return entropy


class LogLinearNoise(nn.Module):
    """Official SEDD log-linear schedule."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = float(eps)

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.log1p(-(1.0 - self.eps) * t)

    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.eps) / (1.0 - (1.0 - self.eps) * t)

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.total_noise(t), self.rate_noise(t)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = int(dim)
        self.max_period = int(max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / max(half, 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class SigmaCond(nn.Module):
    def __init__(self, d_model: int, emb_dim: int = 256):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(emb_dim)
        self.net = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, sigma_b: torch.Tensor) -> torch.Tensor:
        return self.net(self.time(sigma_b))


class SEDDEncoder(nn.Module):
    """
    Simplified encoder-only SEDD network.

    This is not the repo's DDiT backbone, but it preserves the important SEDD
    output semantics:
      1) outputs are log-scores, not categorical logits,
      2) for absorbing diffusion they are scaled by log(expm1(sigma)),
      3) the score at the current token is set to 0.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        d_hid: int = 512,
        n_layers: int = 6,
        max_len: int = 16,
        dropout: float = 0.0,
        scale_by_sigma: bool = True,
        absorb: bool = True,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.scale_by_sigma = bool(scale_by_sigma)
        self.absorb = bool(absorb)

        self.tok_emb = nn.Embedding(self.vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        self.sig_emb = SigmaCond(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, self.vocab_size)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, tokens_bt: torch.LongTensor, sigma_b: torch.Tensor) -> torch.Tensor:
        B, T = tokens_bt.shape
        x = self.tok_emb(tokens_bt) + self.pos_emb[:, :T, :]
        c = F.silu(self.sig_emb(sigma_b))
        x = x + c[:, None, :]
        x = self.enc(x)
        x = self.out(x)

        if self.scale_by_sigma:
            assert self.absorb, "scale_by_sigma here is configured only for absorbing SEDD."
            esigm1_log = torch.where(
                sigma_b < 0.5,
                torch.expm1(sigma_b),
                sigma_b.exp() - 1.0,
            ).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - math.log(self.vocab_size - 1)

        x = torch.scatter(x, -1, tokens_bt[..., None], torch.zeros_like(x[..., :1]))
        return x


@dataclass
class SamplerConfig:
    predictor: str = "analytic"  # {analytic, euler}
    steps: int = 128
    denoise: bool = True
    eps: float = 1e-5


class _SamplingWrapper:
    def __init__(self, model: nn.Module):
        self.model = model

    def get_score(self, x: torch.LongTensor, sigma: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            log_score = self.model(x, sigma.reshape(-1))
            return log_score.exp()


def sedd_absorb_loss(
    graph: AbsorbingGraph,
    model: SEDDEncoder,
    x0_bt: torch.LongTensor,
    noise: LogLinearNoise,
    t_eps: float = 1e-3,
) -> Tuple[torch.Tensor, dict]:
    B, _ = x0_bt.shape
    device = x0_bt.device

    # Official repo samples t in [eps, 1].
    t = (1.0 - t_eps) * torch.rand(B, device=device) + t_eps
    sigma_b, dsigma_b = noise(t)
    x_t_bt = graph.sample_transition(x0_bt, sigma_b[:, None])
    log_score = model(x_t_bt, sigma_b)

    loss_bt = graph.score_entropy(log_score, sigma_b[:, None], x_t_bt, x0_bt)
    loss_b = (dsigma_b[:, None] * loss_bt).sum(dim=-1)
    stats = {
        "mask_frac": float((x_t_bt == graph.mask_id).float().mean().item()),
        "t_mean": float(t.mean().item()),
        "sigma_mean": float(sigma_b.mean().item()),
    }
    return loss_b.mean(), stats


@torch.no_grad()
def _euler_step(
    graph: AbsorbingGraph,
    model_wrap: _SamplingWrapper,
    noise: LogLinearNoise,
    x: torch.LongTensor,
    t: torch.Tensor,
    step_size: float,
) -> torch.LongTensor:
    sigma, dsigma = noise(t)
    score = model_wrap.get_score(x, sigma)
    rev_rate = step_size * dsigma[:, None, None] * graph.reverse_rate(x, score)
    return graph.sample_rate(x, rev_rate)


@torch.no_grad()
def _analytic_step(
    graph: AbsorbingGraph,
    model_wrap: _SamplingWrapper,
    noise: LogLinearNoise,
    x: torch.LongTensor,
    t: torch.Tensor,
    step_size: float,
) -> torch.LongTensor:
    curr_sigma = noise.total_noise(t)
    next_t = (t - step_size).clamp_min(0.0)
    next_sigma = noise.total_noise(next_t)
    dsigma = curr_sigma - next_sigma
    score = model_wrap.get_score(x, curr_sigma)
    stag_score = graph.staggered_score(score, dsigma)
    probs = stag_score * graph.transp_transition(x, dsigma)
    return _sample_categorical(probs)


@torch.no_grad()
def _denoise(
    graph: AbsorbingGraph,
    model_wrap: _SamplingWrapper,
    noise: LogLinearNoise,
    x: torch.LongTensor,
    t: torch.Tensor,
) -> torch.LongTensor:
    sigma = noise.total_noise(t)
    score = model_wrap.get_score(x, sigma)
    stag_score = graph.staggered_score(score, sigma)
    probs = stag_score * graph.transp_transition(x, sigma)
    probs = probs[..., :-1]
    return _sample_categorical(probs)


@torch.no_grad()
def sedd_sample(
    model: SEDDEncoder,
    graph: AbsorbingGraph,
    noise: LogLinearNoise,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    sampler: SamplerConfig,
) -> torch.LongTensor:
    model_wrap = _SamplingWrapper(model)
    x = graph.sample_limit(batch_size, seq_len).to(device)

    timesteps = torch.linspace(1.0, sampler.eps, sampler.steps + 1, device=device)
    dt = (1.0 - sampler.eps) / sampler.steps

    for i in range(sampler.steps):
        t = timesteps[i].expand(batch_size)
        if sampler.predictor == "analytic":
            x = _analytic_step(graph, model_wrap, noise, x, t, dt)
        elif sampler.predictor == "euler":
            x = _euler_step(graph, model_wrap, noise, x, t, dt)
        else:
            raise ValueError(f"Unknown predictor: {sampler.predictor}")

    if sampler.denoise:
        t = timesteps[-1].expand(batch_size)
        x = _denoise(graph, model_wrap, noise, x, t)

    return x.clamp_max(graph.mask_id - 1)


class SEDD(nn.Module):
    """
    Drop-in SEDD wrapper for small discrete playgrounds.

    Key differences from your old file:
      - uses the correct absorbing graph size for binary data,
      - trains log-scores with score entropy,
      - applies absorbing sigma scaling and zeroes the current token score,
      - samples with the actual SEDD reverse process (analytic or Euler),
        not a softmax-confidence heuristic.
    """

    def __init__(
        self,
        data_classes: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        d_hid: int = 512,
        n_layers: int = 6,
        dropout: float = 0.0,
        max_len: int = 16,
        noise_eps: float = 1e-3,
        t_eps: float = 1e-3,
        t_steps: int = 128,
        predictor: str = "analytic",
        denoise: bool = True,
        sampling_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__()
        self.graph = AbsorbingGraph(dim=data_classes)
        self.vocab_size = self.graph.dim
        self.mask_id = self.graph.mask_id

        self.model = SEDDEncoder(
            vocab_size=self.vocab_size,
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            n_layers=n_layers,
            max_len=max_len,
            dropout=dropout,
            scale_by_sigma=True,
            absorb=True,
        )

        self.noise = LogLinearNoise(eps=noise_eps)
        self.t_eps = float(t_eps)
        self.sampler = SamplerConfig(
            predictor=predictor,
            steps=t_steps,
            denoise=denoise,
            eps=sampling_eps,
        )

        if device is not None:
            self.to(device)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.training_loss(batch)

    def training_loss(self, batch: torch.Tensor) -> torch.Tensor:
        loss, _ = sedd_absorb_loss(self.graph, self.model, batch.long(), self.noise, t_eps=self.t_eps)
        return loss

    @torch.no_grad()
    def sample(self, N: int, L: int, device: torch.device) -> torch.Tensor:
        return sedd_sample(
            model=self.model,
            graph=self.graph,
            noise=self.noise,
            batch_size=N,
            seq_len=L,
            device=device,
            sampler=self.sampler,
        )
