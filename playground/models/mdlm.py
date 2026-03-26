import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_positional_encoding(max_len: int, d_model: int, device=None):
    pe = torch.zeros(max_len, d_model, device=device)
    pos = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class PosEnc(nn.Module):
    """
    One-hot token projection + sinusoidal positional encoding.

    Input:  [T, B, V]
    Output: [T, B, D]
    """
    def __init__(self, vocab_size: int, d_model: int, max_len: int):
        super().__init__()
        self.proj = nn.Linear(vocab_size, d_model, bias=False)
        self.register_buffer(
            "pe",
            _sinusoidal_positional_encoding(max_len=max_len, d_model=d_model),
        )

    def forward(self, x_tbv: torch.Tensor) -> torch.Tensor:
        T = x_tbv.size(0)
        h = self.proj(x_tbv)
        return h + self.pe[:T, :].unsqueeze(1)


def _mdlm_alpha_q_w(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MD4-style cosine schedule. (see MD4 page 15 )

    Returns:
      alpha(t): cumulative clean-retention surrogate
      q(t):     masking probability  # 1 - alpha_t
      w(t):     training weight     # alpha_t' / (1 - alpha_t)
    """
    one_minus_t = 1.0 - t
    alpha = 1.0 - torch.cos(0.5 * math.pi * one_minus_t)
    dalpha = 0.5 * math.pi * torch.sin(0.5 * math.pi * one_minus_t)
    q = (1.0 - alpha).clamp_min(1e-8)   
    w = dalpha.abs() / q  
    return alpha, q, w


class TimeCond(nn.Module):
    def __init__(self, d_model: int, emb_dim: int = 256):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.net = nn.Sequential(
            nn.Linear(self.emb_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t_b: torch.Tensor) -> torch.Tensor:
        half = self.emb_dim // 2
        freqs = torch.exp(
            torch.linspace(0, -math.log(10000), steps=half, device=t_b.device)
        )
        args = t_b[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.emb_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.net(emb)


class MDLMEncoder(nn.Module):
    """
    Token space: {0,1,MASK} = {0,1,2}
    Predict logits over clean bits {0,1}.
    """
    def __init__(
        self,
        vocab_size: int = 3, # 0, 1, 2
        d_model: int = 128,
        nhead: int = 4,
        d_hid: int = 512,
        n_layers: int = 6,
        max_len: int = 16,
        dropout: float = 0.0,
        use_time_cond: bool = False,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.out_classes = 2
        self.use_time_cond = bool(use_time_cond)

        self.pos = PosEnc(self.vocab_size, d_model, max_len)
        self.time_emb = TimeCond(d_model) if self.use_time_cond else None

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=False,
        )
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.fc = nn.Linear(d_model, self.out_classes)

    def forward(
        self,
        tokens_bt: torch.LongTensor,
        t_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        onehot = F.one_hot(tokens_bt, num_classes=self.vocab_size).float()   # [B,T,V]
        h = self.pos(onehot.transpose(0, 1).contiguous())                    # [T,B,D]

        if self.use_time_cond and t_b is not None:
            h = h + self.time_emb(t_b).unsqueeze(0)

        h = self.enc(h)
        return self.fc(h).transpose(0, 1).contiguous()                       # [B,T,2]


class MDLM(nn.Module):
    """
    Toy binary MDLM.

    Public API:
      - training_loss(batch)
      - sample(N, L, device)
    """
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        d_hid: int = 512,
        n_layers: int = 6,
        dropout: float = 0.0,
        max_len: int = 16,
        mask_token: int = 2,
        use_time_cond: bool = False,
        steps: int = 64,
        t_eps: float = 0.07,
        add_recon_const: bool = False,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__()
        self.V = 3
        self.C = 2
        self.mask_token = int(mask_token)
        self.use_time_cond = bool(use_time_cond)
        self.steps = int(steps)
        self.t_eps = float(t_eps)
        #self.add_recon_const = bool(add_recon_const)
        self.temperature = float(temperature)

        if self.mask_token != 2:
            raise ValueError("This toy MDLM assumes token space {0,1,MASK} with mask_token=2.")

        self.model = MDLMEncoder(
            vocab_size=self.V,
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            n_layers=n_layers,
            max_len=max_len,
            dropout=dropout,
            use_time_cond=use_time_cond,
        )

        if device is not None:
            self.to(device)

    def forward(
        self,
        tokens_bt: torch.LongTensor,
        t_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(tokens_bt, t_b=t_b)

    def q_xt(self, x0_bt: torch.LongTensor, q_b: torch.Tensor) -> torch.LongTensor:
        """
        Independent masking corruption: See Algorithm 1, line 4
          x_t[i] = MASK with prob q(t), else x0[i] # q(t) = 1-alpha
        """
        move_mask = torch.rand_like(x0_bt.float()) < q_b[:, None]
        xt = x0_bt.clone()
        xt[move_mask] = self.mask_token
        return xt

    def training_loss(self, batch: torch.Tensor) -> torch.Tensor:
        loss, _ = self._compute_loss_mdlm(
            batch.long(),
            t_eps=self.t_eps,
        )
        return loss

    @torch.no_grad()
    def sample(self, N: int, L: int, device: torch.device) -> torch.Tensor:
        return self._reverse_sample(L=L, B=N, steps=self.steps, device=device)

    def _compute_loss_mdlm(
        self,
        x0_bt: torch.LongTensor,
        *,
        t_eps: float = 0.07,
    ):
        device = x0_bt.device
        B, T = x0_bt.shape

        if x0_bt.dim() != 2:
            raise ValueError(f"Expected [B,T], got {tuple(x0_bt.shape)}")
        if not ((x0_bt == 0) | (x0_bt == 1)).all():
            raise ValueError("MDLM.training_loss expects clean binary inputs in {0,1} only.")

        # ------------------------------------------------------------
        # 1) sample continuous time t ~ Uniform[t_eps, 1 - t_eps]
        # ------------------------------------------------------------
        t_b = (1.0 - 2.0 * t_eps) * torch.rand(B, device=device) + t_eps

        # alpha(t), q(t)=1-alpha(t), w(t)=|alpha'(t)|/(1-alpha(t))
        alpha_b, q_b, w_b = _mdlm_alpha_q_w(t_b)

        # ------------------------------------------------------------
        # 2) sample z_t ~ q(z_t | x0) by masking each token independently
        # ------------------------------------------------------------
        x_t_bt = self.q_xt(x0_bt, q_b)
        mask_bt = (x_t_bt == self.mask_token)

        # Rare degenerate edge case: no masked tokens in the whole batch
        if not mask_bt.any():
            zero = x0_bt.float().sum() * 0.0
            stats = {
                "loss": 0.0,
                "diffusion_loss": 0.0,
                "reconstruction_loss": 0.0,
                "mask_frac": 0.0,
                "t_mean": float(t_b.mean().item()),
                "q_mean": float(q_b.mean().item()),
                "alpha_mean": float(alpha_b.mean().item()),
            }
            return zero, stats

        # ------------------------------------------------------------
        # 3) model predicts logits over clean bits {0,1}
        # ------------------------------------------------------------
        logits_btc = self(
            x_t_bt,
            t_b=t_b if self.use_time_cond else None,
        )  # [B,T,2]
        log_probs_btc = F.log_softmax(logits_btc, dim=-1)

        # ------------------------------------------------------------
        # 4) SUBS diffusion loss:
        #    -w(t) * log p_theta(x0 | z_t, t)
        #    only on masked positions
        # ------------------------------------------------------------
        log_p_x0_bt = torch.gather(
            log_probs_btc,
            dim=-1,
            index=x0_bt[..., None],   # [B,T,1]
        ).squeeze(-1)                 # [B,T]

        diffusion_tok_bt = -log_p_x0_bt * w_b[:, None]
        diffusion_loss = diffusion_tok_bt[mask_bt].mean()

        # ------------------------------------------------------------
        # 5) Continuous-time SUBS reconstruction loss is exactly 0
        # ------------------------------------------------------------
        reconstruction_loss = diffusion_loss.new_zeros(())

        # ------------------------------------------------------------
        # 6) Total loss
        # ------------------------------------------------------------
        loss = diffusion_loss + reconstruction_loss

        stats = {
            "loss": float(loss.detach().item()),
            "diffusion_loss": float(diffusion_loss.detach().item()),
            "reconstruction_loss": float(reconstruction_loss.detach().item()),
            "mask_frac": float(mask_bt.float().mean().item()),
            "t_mean": float(t_b.mean().item()),
            "q_mean": float(q_b.mean().item()),
            "alpha_mean": float(alpha_b.mean().item()),
        }
        return loss, stats

    @torch.no_grad()
    def _reverse_sample(
        self,
        *,
        L: int,
        B: int,
        steps: int,
        device: torch.device,
        eps: float = 1e-5,
    ) -> torch.LongTensor:
        """
        Repo-faithful DDPM-style reverse sampler for binary SUBS toy MDLM.
        """
        xt = torch.full((B, L), self.mask_token, device=device, dtype=torch.long)
        timesteps = torch.linspace(1.0, eps, steps + 1, device=device)

        for i in range(steps):
            t_i = torch.full((B,), timesteps[i], device=device)
            s_i = torch.full((B,), timesteps[i + 1], device=device)

            # p_theta(x0 | x_t, t)
            logits_btc = self(
                xt,
                t_b=t_i if self.use_time_cond else None,
            )  # [B,L,2]
            probs_btc = torch.softmax(
                logits_btc / max(self.temperature, 1e-8),
                dim=-1,
            )  # [B,L,2]

            # q_t, q_s: mask probabilities at times t and s
            _, q_t, _ = _mdlm_alpha_q_w(t_i)
            _, q_s, _ = _mdlm_alpha_q_w(s_i)

            # repo _ddpm_update analogue:
            # clean mass = p_theta(x0|x_t,t) * (q_t - q_s)
            # mask mass  = q_s
            delta_q = (q_t - q_s).clamp_min(0.0)[:, None, None]

            q_xs = torch.zeros(B, L, 3, device=device, dtype=probs_btc.dtype)
            q_xs[..., 0] = probs_btc[..., 0] * delta_q[..., 0]
            q_xs[..., 1] = probs_btc[..., 1] * delta_q[..., 0]
            q_xs[..., 2] = q_s[:, None].expand(B, L)

            q_xs = q_xs / q_xs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            sampled = torch.distributions.Categorical(probs=q_xs).sample()

            # once unmasked, keep fixed
            copy_flag = (xt != self.mask_token)
            xt = torch.where(copy_flag, xt, sampled)

        # repo-style final noise removal
        if (xt == self.mask_token).any():
            t_final = torch.full((B,), timesteps[-1], device=device)
            logits_btc = self(
                xt,
                t_b=t_final if self.use_time_cond else None,
            )
            final_bits = logits_btc.argmax(dim=-1)
            left = (xt == self.mask_token)
            xt[left] = final_bits[left]

        return xt