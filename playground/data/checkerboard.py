import torch
from torch.utils.data import IterableDataset
import numpy as np

class CheckerboardBinaryStream(IterableDataset):
    def __init__(
        self,
        n_cells=8,
        R=8.0,
        n_bits=8,
        interleave=False,
        reverse=False,
        p_white=0.0,
        device="cpu",
    ):
        super().__init__()
        self.device = torch.device("cpu")  # keep CPU like your GMM class
        self.R = float(R)
        self.n_bits = int(n_bits)
        self.interleave = bool(interleave)
        self.reverse = bool(reverse)

        self.n_cells = int(n_cells)
        assert self.n_cells > 0
        self.p_white = float(p_white)
        assert self.p_white >= 0.0

        # cell width
        self.cell = (2.0 * self.R) / self.n_cells

        # Precompute all cell indices and their "color"
        # i, j in [0, n_cells-1]
        ii, jj = torch.meshgrid(
            torch.arange(self.n_cells, device=self.device),
            torch.arange(self.n_cells, device=self.device),
            indexing="ij",
        )
        is_black = ((ii + jj) % 2 == 0).reshape(-1)  # (n_cells^2,)
        self._black_ids = torch.nonzero(is_black, as_tuple=False).squeeze(1)
        self._white_ids = torch.nonzero(~is_black, as_tuple=False).squeeze(1)

        # Sampling weights over squares (mixture over cells)
        if self.p_white == 0.0:
            self._cell_ids = self._black_ids
            self._cell_probs = torch.ones(len(self._cell_ids), device=self.device)
            self._cell_probs = self._cell_probs / self._cell_probs.sum()
        else:
            # Mix black and white with relative weight 1 : p_white
            nb = len(self._black_ids)
            nw = len(self._white_ids)
            probs_black = torch.ones(nb, device=self.device) * 1.0
            probs_white = torch.ones(nw, device=self.device) * self.p_white
            self._cell_ids = torch.cat([self._black_ids, self._white_ids], dim=0)
            self._cell_probs = torch.cat([probs_black, probs_white], dim=0)
            self._cell_probs = self._cell_probs / self._cell_probs.sum()

        # Bit helpers (same as your class)
        self._shifts = torch.arange(self.n_bits - 1, -1, -1, device=self.device)
        self._eps = 2.0 ** (-self.n_bits)

    @torch.no_grad()
    def sample_xy(self):
        # Choose a cell id (flattened index into n_cells x n_cells)
        k = torch.multinomial(self._cell_probs, num_samples=1).item()
        flat = int(self._cell_ids[k].item())

        i = flat // self.n_cells
        j = flat % self.n_cells

        x0 = -self.R + i * self.cell
        y0 = -self.R + j * self.cell

        u = torch.rand(2, device=self.device)
        x = x0 + u[0] * self.cell
        y = y0 + u[1] * self.cell

        # clamp to stay within [-R, R) (avoid right edge hitting exactly R)
        x = torch.clamp(x, -self.R, self.R - 1e-12)
        y = torch.clamp(y, -self.R, self.R - 1e-12)
        return torch.stack([x, y], dim=0)

    def __iter__(self):
        while True:
            xy = self.sample_xy()

            u = ((xy + self.R) / (2 * self.R)).clamp(0.0, 1.0 - self._eps)
            v = torch.floor(u * (1 << self.n_bits)).to(torch.long)  # (2,)
            bits = ((v.unsqueeze(-1) >> self._shifts) & 1).to(torch.long)  # (2, n_bits)

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
            by = seq[self.n_bits:2 * self.n_bits]

        vx = 0
        for b in bx:
            vx = (vx << 1) | int(b)
        vy = 0
        for b in by:
            vy = (vy << 1) | int(b)

        ux = vx / float(1 << self.n_bits)
        uy = vy / float(1 << self.n_bits)
        x = ux * (2 * self.R) - self.R
        y = uy * (2 * self.R) - self.R
        return np.array([x, y], dtype=float)
