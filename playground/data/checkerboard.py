import math
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.distributions import Categorical, MixtureSameFamily, Uniform, Independent


class CheckerboardBinaryStream(IterableDataset):
    """
    Checkerboard on [-R, R]^2.

    If rotate_45=False:
        standard axis-aligned checkerboard with n_cells x n_cells cells.

    If rotate_45=True:
        checkerboard defined in randomly rotated coordinates, approximated by a
        fine mixture of small axis-aligned rectangles.
    """

    def __init__(
        self,
        n_cells=8,
        R=8.0,
        n_bits=8,
        interleave=False,
        reverse=False,
        p_white=0.0,
        rotate_45=True,
        approx_subdiv=8,
        random_rotate=True,
        random_shift=True,
        max_shift_frac=1.0,   # max shift as fraction of one checker cell
        device="cpu",
    ):
        super().__init__()
        self.device = torch.device("cpu")
        self.R = float(R)
        self.n_bits = int(n_bits)
        self.interleave = bool(interleave)
        self.reverse = bool(reverse)

        self.n_cells = int(n_cells)
        assert self.n_cells > 0

        self.p_white = float(p_white)
        assert self.p_white >= 0.0

        self.rotate_45 = bool(rotate_45)
        self.approx_subdiv = int(approx_subdiv)
        assert self.approx_subdiv > 0

        self.random_rotate = bool(random_rotate)
        self.random_shift = bool(random_shift)
        self.max_shift_frac = float(max_shift_frac)
        assert self.max_shift_frac >= 0.0

        self._eps = 2.0 ** (-self.n_bits)
        self._shifts = torch.arange(self.n_bits - 1, -1, -1, device=self.device)

        # standard checkerboard cell size in xy when not rotated
        self.cell = (2.0 * self.R) / self.n_cells

        # Geometry is sampled once here, like loc in your GMMBinaryStream.
        # So torch.manual_seed(cfg.seed) before dataset creation makes this deterministic.
        if self.rotate_45:
            if self.random_rotate:
                self.theta = (2.0 * math.pi * torch.rand((), device=self.device)).item()
            else:
                self.theta = math.pi / 4.0

            if self.random_shift:
                max_shift = self.max_shift_frac * self.cell
                self.shift_x = ((2.0 * torch.rand((), device=self.device) - 1.0) * max_shift).item()
                self.shift_y = ((2.0 * torch.rand((), device=self.device) - 1.0) * max_shift).item()
            else:
                self.shift_x = 0.0
                self.shift_y = 0.0
        else:
            self.theta = 0.0
            self.shift_x = 0.0
            self.shift_y = 0.0

        self.dist = self._build_dist()

    def _build_dist(self):
        if not self.rotate_45:
            return self._build_axis_aligned_checkerboard()
        return self._build_rotated_checkerboard_approx()

    def _build_axis_aligned_checkerboard(self):
        ii, jj = torch.meshgrid(
            torch.arange(self.n_cells, device=self.device),
            torch.arange(self.n_cells, device=self.device),
            indexing="ij",
        )
        is_black = ((ii + jj) % 2 == 0).reshape(-1)

        black_ids = torch.nonzero(is_black, as_tuple=False).squeeze(1)
        white_ids = torch.nonzero(~is_black, as_tuple=False).squeeze(1)

        if self.p_white == 0.0:
            cell_ids = black_ids
            probs = torch.ones(len(cell_ids), device=self.device)
        else:
            nb = len(black_ids)
            nw = len(white_ids)
            probs_black = torch.ones(nb, device=self.device)
            probs_white = torch.ones(nw, device=self.device) * self.p_white
            cell_ids = torch.cat([black_ids, white_ids], dim=0)
            probs = torch.cat([probs_black, probs_white], dim=0)

        probs = probs / probs.sum()

        flat = cell_ids.to(torch.long)
        i = flat // self.n_cells
        j = flat % self.n_cells

        lows = torch.stack(
            [-self.R + i * self.cell, -self.R + j * self.cell],
            dim=-1,
        ).to(self.device)
        highs = lows + self.cell

        comp = Independent(Uniform(lows, highs, validate_args=False), 1)
        mix = Categorical(probs=probs)
        return MixtureSameFamily(mix, comp, validate_args=False)

    def _build_rotated_checkerboard_approx(self):
        # Tile [-R, R]^2 with many tiny axis-aligned rectangles.
        # Each tiny rectangle gets assigned black/white based on the parity
        # of its center in rotated+shifted coordinates.
        n_fine = self.n_cells * self.approx_subdiv
        fine = (2.0 * self.R) / n_fine

        xs = torch.linspace(-self.R, self.R - fine, n_fine, device=self.device)
        ys = torch.linspace(-self.R, self.R - fine, n_fine, device=self.device)

        x0, y0 = torch.meshgrid(xs, ys, indexing="ij")
        xc = x0 + 0.5 * fine
        yc = y0 + 0.5 * fine

        # Shift the board before rotating, so the checker pattern itself is translated.
        xcs = xc - self.shift_x
        ycs = yc - self.shift_y

        c = math.cos(self.theta)
        s = math.sin(self.theta)

        # General rotated coordinates:
        # old 45-degree case was a special case of this.
        u = c * xcs + s * ycs
        v = -s * xcs + c * ycs

        # Use the full rotated span that contains the rotated square.
        rot_R = math.sqrt(2.0) * self.R
        rot_cell = (2.0 * rot_R) / self.n_cells

        iu = torch.floor((u + rot_R) / rot_cell).long()
        iv = torch.floor((v + rot_R) / rot_cell).long()

        iu = torch.clamp(iu, 0, self.n_cells - 1)
        iv = torch.clamp(iv, 0, self.n_cells - 1)

        is_black = ((iu + iv) % 2 == 0)

        weights = torch.where(
            is_black,
            torch.ones_like(xc, dtype=torch.float32),
            torch.ones_like(xc, dtype=torch.float32) * self.p_white,
        )

        keep = weights.reshape(-1) > 0
        weights = weights.reshape(-1)[keep]

        lows = torch.stack([x0.reshape(-1), y0.reshape(-1)], dim=-1)[keep]
        highs = lows + fine

        probs = weights / weights.sum()

        comp = Independent(Uniform(lows, highs, validate_args=False), 1)
        mix = Categorical(probs=probs)
        return MixtureSameFamily(mix, comp, validate_args=False)

    @torch.no_grad()
    def sample_xy(self):
        return self.dist.sample()

    def __iter__(self):
        while True:
            xy = self.sample_xy()

            # tokenization stays on [-R, R]^2
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