import math
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.distributions import Categorical, MixtureSameFamily, Uniform, Independent


class _RotatedCheckerboardDist:
    """Rotated, randomly shifted checkerboard on [-R, R]^2.

    Sampling is rejection (accept rate ≥ p_white when p_white > 0, else 0.5).
    log_prob uses a cell-parity test in (u, v) coordinates, normalized by a
    one-shot MC estimate of the support area. cell_mass_grid is computed once
    via a deterministic-seed MC histogram and cached per (Ex, Ey) edges.
    """

    def __init__(self, theta, shift_x, shift_y, R, n_cells, p_white,
                 area_mc_n=1_000_000, mass_mc_n=4_000_000):
        self.theta = float(theta)
        self.shift_x = float(shift_x)
        self.shift_y = float(shift_y)
        self.R = float(R)
        self.n_cells = int(n_cells)
        self.p_white = float(p_white)
        self._c, self._s = math.cos(theta), math.sin(theta)
        self._rot_R = math.sqrt(2.0) * self.R
        self._rot_cell = (2.0 * self._rot_R) / self.n_cells
        self._mass_mc_n = int(mass_mc_n)
        # Total weighted area = ∫_{[-R,R]^2} weight(x,y) dx dy.  Used as the
        # log_prob normalization. Estimated by MC with a fixed seed for
        # determinism (does not perturb global RNG).
        self._twa = self._estimate_total_weighted_area(area_mc_n)
        self._mass_cache = {}

    def _color_weight(self, xy):
        """Weight at each (x,y): 1 inside black cells, p_white inside white,
        0 outside [-R,R]^2 OR outside the n_cells x n_cells rotated grid.

        No iu/iv clamping: the support is exactly the in-bounds rotated cells
        intersected with [-R,R]^2 (matches the main-branch fix's semantics —
        a "n_cells x n_cells rotated checkerboard" really has only n_cells^2
        cells, even when shift makes the rotated grid not cover the visible
        window completely)."""
        in_window = ((xy[..., 0] >= -self.R) & (xy[..., 0] <= self.R)
                   & (xy[..., 1] >= -self.R) & (xy[..., 1] <= self.R))
        xs = xy[..., 0] - self.shift_x
        ys = xy[..., 1] - self.shift_y
        u = self._c * xs + self._s * ys
        v = -self._s * xs + self._c * ys
        iu = torch.floor((u + self._rot_R) / self._rot_cell).long()
        iv = torch.floor((v + self._rot_R) / self._rot_cell).long()
        in_grid = (iu >= 0) & (iu < self.n_cells) & (iv >= 0) & (iv < self.n_cells)
        # parity uses clamped indices only as a lookup-safety guard; out-of-grid
        # points are zeroed by the `in_support` mask below regardless.
        iu_safe = iu.clamp(0, self.n_cells - 1)
        iv_safe = iv.clamp(0, self.n_cells - 1)
        is_black = ((iu_safe + iv_safe) % 2 == 0)
        wt = torch.where(is_black,
                         torch.ones_like(u),
                         torch.full_like(u, self.p_white))
        in_support = in_window & in_grid
        return torch.where(in_support, wt, torch.zeros_like(u))

    @torch.no_grad()
    def debug_sampler_stats(self, n_samples=10000, p_white_override=None):
        """Sanity helper: draw n_samples uniform candidates on [-R,R]^2 and
        report acceptance behavior (rejection breakdown by reason)."""
        n = int(n_samples)
        if n <= 0:
            raise ValueError("n_samples must be > 0")
        p_white = float(self.p_white if p_white_override is None else p_white_override)
        xy = torch.rand(n, 2, dtype=torch.float64) * (2 * self.R) - self.R
        in_window = ((xy[..., 0] >= -self.R) & (xy[..., 0] <= self.R)
                   & (xy[..., 1] >= -self.R) & (xy[..., 1] <= self.R))
        xs = xy[..., 0] - self.shift_x
        ys = xy[..., 1] - self.shift_y
        u = self._c * xs + self._s * ys
        v = -self._s * xs + self._c * ys
        iu = torch.floor((u + self._rot_R) / self._rot_cell).long()
        iv = torch.floor((v + self._rot_R) / self._rot_cell).long()
        in_grid = (iu >= 0) & (iu < self.n_cells) & (iv >= 0) & (iv < self.n_cells)
        iu_safe = iu.clamp(0, self.n_cells - 1)
        iv_safe = iv.clamp(0, self.n_cells - 1)
        parity_black = ((iu_safe + iv_safe) % 2 == 0)
        in_support = in_window & in_grid
        keep_prob = torch.zeros(n, dtype=torch.float64)
        keep_prob[in_support] = torch.where(
            parity_black[in_support],
            torch.ones_like(keep_prob[in_support]),
            torch.full_like(keep_prob[in_support], p_white),
        )
        rnd = torch.rand(n, dtype=torch.float64)
        keep = rnd < keep_prob
        rej_outside_grid = (~in_grid) & (~keep)
        rej_white = in_support & (~parity_black) & (~keep)
        return {
            "n_samples": n,
            "acceptance_rate": float(keep.float().mean().item()),
            "frac_reject_outside_rot_grid": float(rej_outside_grid.float().mean().item()),
            "frac_reject_white": float(rej_white.float().mean().item()),
            "support_area_estimate": float(in_support.float().mean().item()) * (2 * self.R) ** 2,
        }

    def _estimate_total_weighted_area(self, n):
        g = torch.Generator()
        g.manual_seed(0)
        xy = torch.rand(n, 2, generator=g, dtype=torch.float64) * (2 * self.R) - self.R
        wt = self._color_weight(xy)
        return float(wt.mean().item() * (2 * self.R) ** 2)

    def _draw_uniform_then_accept(self, N, generator=None):
        """Rejection step: oversample uniform-on-[-R,R]^2, accept by color
        weight. Returns up to N accepted (x,y) points as a float32 tensor."""
        accept_rate = max(self._twa / ((2 * self.R) ** 2), 1e-6)
        oversample = max(1.5 / accept_rate, 1.5)
        out = torch.empty(N, 2, dtype=torch.float32)
        filled = 0
        while filled < N:
            need = N - filled
            B = max(int(need * oversample), 256)
            if generator is None:
                xy = torch.rand(B, 2, dtype=torch.float64) * (2 * self.R) - self.R
            else:
                xy = torch.rand(B, 2, generator=generator, dtype=torch.float64) \
                     * (2 * self.R) - self.R
            wt = self._color_weight(xy)
            if 0.0 < self.p_white < 1.0:
                if generator is None:
                    u = torch.rand(B, dtype=torch.float64)
                else:
                    u = torch.rand(B, generator=generator, dtype=torch.float64)
                accept = u < wt
            else:
                accept = wt > 0
            kept = xy[accept]
            take = min(len(kept), need)
            out[filled:filled + take] = kept[:take].to(torch.float32)
            filled += take
        return out

    def sample(self, sample_shape=torch.Size()):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        sample_shape = torch.Size(sample_shape)
        N = max(int(np.prod(sample_shape)), 1)
        out = self._draw_uniform_then_accept(N)
        if len(sample_shape) == 0:
            return out[0]
        return out.reshape(*sample_shape, 2)

    def log_prob(self, xy):
        xy = torch.as_tensor(xy)
        if xy.dtype != torch.float64:
            xy = xy.to(torch.float64)
        wt = self._color_weight(xy)
        density = wt / self._twa
        return torch.log(density)

    def cell_mass_grid(self, edges_x, edges_y):
        """MC histogram of bin masses, with a fixed seed and per-edges cache."""
        Ex = np.asarray(edges_x, dtype=np.float64)
        Ey = np.asarray(edges_y, dtype=np.float64)
        key = (Ex.tobytes(), Ey.tobytes())
        if key in self._mass_cache:
            return self._mass_cache[key].copy()

        g = torch.Generator()
        g.manual_seed(42)
        xy = self._draw_uniform_then_accept(self._mass_mc_n, generator=g).numpy()
        H, *_ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[Ex, Ey])
        s_total = H.sum()
        if s_total > 0:
            H = H / s_total
        self._mass_cache[key] = H.copy()
        return H


from playground.data.encoding import binary_to_gray, gray_to_binary, normalize_encoding


class CheckerboardBinaryStream(IterableDataset):
    """
    Checkerboard on [-R, R]^2.

    If rotate_45=False:
        standard axis-aligned checkerboard with n_cells x n_cells cells.

    If rotate_45=True:
        checkerboard defined in randomly rotated and shifted coordinates,
        sampled exactly via a triangle-mixture decomposition (no rejection
        and no fine-grid staircase artifacts).
    """

    def __init__(
        self,
        n_cells=8,
        R=8.0,
        n_bits=8,
        interleave=False,
        reverse=False,
        encoding="binary",
        p_white=0.0,
        rotate_45=True,
        theta=None,
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
        self.encoding = normalize_encoding(encoding)

        self.n_cells = int(n_cells)
        assert self.n_cells > 0

        self.p_white = float(p_white)
        assert self.p_white >= 0.0

        self.rotate_45 = bool(rotate_45)

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
            if theta is not None:
                self.theta = float(theta)
            elif self.random_rotate:
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
        return self._build_rotated_checkerboard_exact()

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

    def _build_rotated_checkerboard_exact(self):
        return _RotatedCheckerboardDist(
            theta=self.theta, shift_x=self.shift_x, shift_y=self.shift_y,
            R=self.R, n_cells=self.n_cells, p_white=self.p_white,
        )

    @torch.no_grad()
    def sample_xy(self):
        # For both axis-aligned and rotated cases, sampling is delegated to self.dist.
        # In rotated mode, self.dist is a _RotatedCheckerboardDist with exact rejection sampling.
        return self.dist.sample()

    def __iter__(self):
        while True:
            xy = self.sample_xy()

            # tokenization stays on [-R, R]^2
            u = ((xy + self.R) / (2 * self.R)).clamp(0.0, 1.0 - self._eps)
            v = torch.floor(u * (1 << self.n_bits)).to(torch.long)  # (2,)
            if self.encoding == "gray":
                v = binary_to_gray(v)
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

        if self.encoding == "gray":
            vx = gray_to_binary(vx, self.n_bits)
            vy = gray_to_binary(vy, self.n_bits)

        ux = vx / float(1 << self.n_bits)
        uy = vy / float(1 << self.n_bits)

        x = ux * (2 * self.R) - self.R
        y = uy * (2 * self.R) - self.R
        return np.array([x, y], dtype=float)
