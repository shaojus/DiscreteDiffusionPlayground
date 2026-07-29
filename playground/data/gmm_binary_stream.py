import torch, torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal
import numpy as np 

from playground.data.encoding import binary_to_gray, gray_to_binary, normalize_encoding


class GMMBinaryStream(IterableDataset):
    def __init__(
        self,
        n_mixes=8,
        R=None,
        log_var_scal=0.0,
        n_bits=8,
        interleave=False,
        reverse=False,
        encoding="binary",
        device="cpu",
        realized_state=None,
    ):
        super().__init__()
        self.device = torch.device("cpu") 
        self.R = float(R) if R is not None else float(n_mixes)
        self.n_bits = int(n_bits)
        self.interleave = bool(interleave)
        self.reverse = bool(reverse)
        self.encoding = normalize_encoding(encoding)

        pi   = torch.ones(n_mixes, device=self.device)
        if realized_state is None:
            loc = (torch.rand(n_mixes, 2, device=self.device) - 0.5) * 2 * n_mixes
        else:
            loc = torch.as_tensor(realized_state["loc"], dtype=torch.float32, device=self.device)
            if tuple(loc.shape) != (int(n_mixes), 2):
                raise ValueError(
                    f"Expected realized GMM locations with shape {(int(n_mixes), 2)}, "
                    f"got {tuple(loc.shape)}"
                )
        self.loc = loc
        logv = torch.ones(n_mixes, 2, device=self.device) * log_var_scal
        scale = torch.diag_embed(F.softplus(logv))
        self.dist = MixtureSameFamily(
            Categorical(pi),
            MultivariateNormal(loc, scale_tril=scale),
            validate_args=False
        )

        self._shifts = torch.arange(self.n_bits - 1, -1, -1, device=self.device)
        self._eps = 2.0 ** (-self.n_bits)

    def realized_state_dict(self):
        return {"loc": self.loc.detach().cpu().clone()}

    def __iter__(self):
        while True:
            # rejection sampling: ensure xy in [-R, R]^2
            while True:
                xy = self.dist.sample()
                if (xy.abs() <= self.R).all():
                    break

            u = (xy + self.R) / (2 * self.R)          # now u is guaranteed in [0,1]
            # keep u strictly < 1 so v is always in [0, 2^n - 1]
            u = torch.clamp(u, 0.0, 1.0 - self._eps)

            v = torch.floor(u * (1 << self.n_bits)).to(torch.long)
            if self.encoding == "gray":
                v = binary_to_gray(v)
            bits = ((v.unsqueeze(-1) >> self._shifts) & 1).to(torch.long)

            if self.interleave:
                seq = torch.stack((bits[0], bits[1]), dim=1).reshape(-1)
            else:
                seq = torch.cat((bits[0], bits[1]), dim=0)

            if self.reverse:
                seq = torch.flip(seq, [0])

            yield seq

    def decode(self, seq):
        seq = np.asarray(seq, dtype=int)

        if self.reverse:
            seq = seq[::-1]

        if self.interleave:
            bx = seq[0::2][:self.n_bits]
            by = seq[1::2][:self.n_bits]
        else:
            bx = seq[:self.n_bits]
            by = seq[self.n_bits:2*self.n_bits]

        vx = 0
        for b in bx: vx = (vx << 1) | int(b)
        vy = 0
        for b in by: vy = (vy << 1) | int(b)
        if self.encoding == "gray":
            vx = gray_to_binary(vx, self.n_bits)
            vy = gray_to_binary(vy, self.n_bits)
        ux = vx / float(1 << self.n_bits)
        uy = vy / float(1 << self.n_bits)
        x = ux * (2 * self.R) - self.R
        y = uy * (2 * self.R) - self.R
        return np.array([x, y], dtype=float)
