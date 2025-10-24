import torch, torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal
import numpy as np 

class GMMBinaryStream(IterableDataset):
    def __init__(self, n_mixes=8, R=None, log_var_scal=0.0, n_bits=8, interleave=False, reverse=False, device="cpu"):
        super().__init__()
        self.device = torch.device("cpu") 
        self.R = float(R) if R is not None else float(n_mixes)
        self.n_bits = int(n_bits)
        self.interleave = bool(interleave)
        self.reverse = bool(reverse)  # Add this line

        pi   = torch.ones(n_mixes, device=self.device)
        loc  = (torch.rand(n_mixes, 2, device=self.device) - 0.5) * 2 * n_mixes
        logv = torch.ones(n_mixes, 2, device=self.device) * log_var_scal
        scale = torch.diag_embed(F.softplus(logv))
        self.dist = MixtureSameFamily(
            Categorical(pi),
            MultivariateNormal(loc, scale_tril=scale),
            validate_args=False
        )

        self._shifts = torch.arange(self.n_bits - 1, -1, -1, device=self.device)
        self._eps = 2.0 ** (-self.n_bits)

    def __iter__(self):
        while True:
            xy = self.dist.sample()
            u = ((xy + self.R) / (2 * self.R)).clamp(0.0, 1.0 - self._eps)
            v = torch.floor(u * (1 << self.n_bits)).to(torch.long)
            bits = ((v.unsqueeze(-1) >> self._shifts) & 1).to(torch.long)
            if self.interleave:
                seq = torch.stack((bits[0], bits[1]), dim=1).reshape(-1)
            else:
                seq = torch.cat((bits[0], bits[1]), dim=0)
            
            if self.reverse:  # Add this block
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
        ux = vx / float(1 << self.n_bits)
        uy = vy / float(1 << self.n_bits)
        x = ux * (2 * self.R) - self.R
        y = uy * (2 * self.R) - self.R
        return np.array([x, y], dtype=float)
