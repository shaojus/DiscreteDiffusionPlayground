import numpy as np
import torch
from scipy.stats import wasserstein_distance, norm
from hyppo.ksample import MMD, Energy
import ot

def get_true_samples(ds, n=60_000):
    if hasattr(ds, "dist"):
        with torch.no_grad():
            return ds.dist.sample((n,)).cpu().numpy()
    it = iter(ds)
    tokens = [next(it).numpy() for _ in range(n)]
    xy = np.vstack([ds.decode(seq) for seq in tokens])
    return xy

def _true_cell_mass_grid(ds, edges_x, edges_y):
    """
    Returns Pt[i,j] = P( x in [Ex[i],Ex[i+1}), y in [Ey[j],Ey[j+1}) )
    for ds.dist, supporting:
      - Mixture of diagonal Gaussians (MixtureSameFamily(Categorical, MultivariateNormal))
      - Mixture of axis-aligned 2D Uniform rectangles (MixtureSameFamily(Categorical, Independent(Uniform)))
    Output is a mass grid that sums ~ 1.
    """
    mix = ds.dist
    cat = mix.mixture_distribution
    comp = mix.component_distribution
    pis = cat.probs.detach().cpu().numpy()  # (K,)

    Ex = np.asarray(edges_x, dtype=np.float64)
    Ey = np.asarray(edges_y, dtype=np.float64)
    Bx = len(Ex) - 1
    By = len(Ey) - 1
    H = np.zeros((Bx, By), dtype=np.float64)

    if hasattr(comp, "loc") and hasattr(comp, "scale_tril"):
        loc = comp.loc.detach().cpu().numpy()              # (K,2)
        scale = comp.scale_tril.detach().cpu().numpy()     # (K,2,2) diag in your case

        for k in range(len(pis)):
            mx, my = loc[k, 0], loc[k, 1]
            sx, sy = scale[k, 0, 0], scale[k, 1, 1]
            Fx = norm.cdf((Ex - mx) / sx)
            Fy = norm.cdf((Ey - my) / sy)
            Px = (Fx[1:, None] - Fx[:-1, None])            # (Bx,1)
            Py = (Fy[None, 1:] - Fy[None, :-1])            # (1,By)
            H += pis[k] * (Px * Py)
        return H

    # checkerboard case
    base = getattr(comp, "base_dist", None)  # Uniform(...)
    if base is None or not hasattr(base, "low") or not hasattr(base, "high"):
        raise TypeError(
            "Unsupported ds.dist component_distribution type. "
            "Expected Gaussian with (loc, scale_tril) or Independent(Uniform) with (low, high)."
        )

    lows = base.low.detach().cpu().numpy()   # (K,2)
    highs = base.high.detach().cpu().numpy() # (K,2)

    bin_wx = Ex[1:] - Ex[:-1]  # (Bx,)
    bin_wy = Ey[1:] - Ey[:-1]  # (By,)

    for k in range(len(pis)):
        lx, ly = lows[k, 0], lows[k, 1]
        hx, hy = highs[k, 0], highs[k, 1]
        area_k = (hx - lx) * (hy - ly)
        if area_k <= 0:
            continue

        # overlap lengths in x for each x-bin
        ox = np.maximum(0.0, np.minimum(Ex[1:], hx) - np.maximum(Ex[:-1], lx))  # (Bx,)
        oy = np.maximum(0.0, np.minimum(Ey[1:], hy) - np.maximum(Ey[:-1], ly))  # (By,)

        # mass in bin = overlap_area / rect_area
        # outer product gives overlap areas per bin
        H += pis[k] * (ox[:, None] * oy[None, :]) / area_k

    return H

def _gridify_mass(samples, edges_x, edges_y):
    H, *_ = np.histogram2d(samples[:, 0], samples[:, 1], bins=[edges_x, edges_y])
    count_in = float(H.sum())
    Pm = H / (count_in + 1e-12)
    return Pm, count_in

def _apply_dirichlet_smoothing(P, alpha):
    if alpha <= 0:
        return P
    Bx, By = P.shape
    return (P + alpha) / (P.sum() + alpha * Bx * By)

def _tv(Pt, Pm):
    return 0.5 * np.sum(np.abs(Pt - Pm))

def _js(P, Q, eps=1e-12):
    M = 0.5 * (P + Q)
    def _kl(a, b):
        return np.sum(a * np.log((a + eps) / (b + eps)))
    return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)

def _sliced_wasserstein_2d(X, Y, n_proj=64, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    X = np.asarray(X); Y = np.asarray(Y)
    sw = 0.0
    for _ in range(n_proj):
        v = rng.normal(size=2)
        v /= np.linalg.norm(v) + 1e-12
        x = np.sort(X @ v)
        y = np.sort(Y @ v)
        n = min(len(x), len(y))
        sw += np.sqrt(np.mean((x[:n] - y[:n])**2))
    return sw / n_proj

def _energy_stat(X, Y):
    return float(Energy().statistic(X, Y))

def w2_exact(X, Y):
    n = X.shape[0]
    assert X.shape == Y.shape
    C = ot.dist(X, Y, metric="euclidean") ** 2
    a = np.ones(n) / n
    b = np.ones(n) / n
    w2_sq = ot.emd2(a, b, C)
    return float(np.sqrt(w2_sq))

def w2_exact_subsample(X, Y, max_n=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = min(len(X), len(Y), max_n)
    idx1 = rng.choice(len(X), n, replace=False)
    idx2 = rng.choice(len(Y), n, replace=False)
    return w2_exact(X[idx1], Y[idx2])

def divergence_metrics_plus(
    ds,
    gen_xy,
    n_true=60_000,
    grid=200,
    alpha=0.0,
    two_sample_max_n=10_000,
    rng_seed=2025,
):
    rng = np.random.default_rng(rng_seed)
    true_xy = get_true_samples(ds, n_true)

    # 1D W2 (marginals)
    w2_x = wasserstein_distance(gen_xy[:, 0], true_xy[:, 0])
    w2_y = wasserstein_distance(gen_xy[:, 1], true_xy[:, 1])

    # Grid definition
    R = float(ds.R)
    edges = np.linspace(-R, R, grid + 1)
    area = (2 * R) ** 2 / (grid * grid)

    # True cell masses (works for both GMM + checkerboard now)
    Pt = _true_cell_mass_grid(ds, edges, edges)
    Pt = Pt / (Pt.sum() + 1e-12)

    # Model cell masses from samples
    Pm, count_in = _gridify_mass(gen_xy, edges, edges)
    in_bounds_frac = float(count_in / max(1, len(gen_xy)))

    # Optional symmetric smoothing
    Pt = _apply_dirichlet_smoothing(Pt, alpha)
    Pm = _apply_dirichlet_smoothing(Pm, alpha)

    D = Pt - Pm

    max_sq_err = float(np.max(D**2))
    ix, iy = np.unravel_index(np.argmax(D**2), D.shape)
    worst_cell = dict(
        ix=int(ix), iy=int(iy),
        x_interval=(float(edges[ix]), float(edges[ix + 1])),
        y_interval=(float(edges[iy]), float(edges[iy + 1])),
        value=max_sq_err,
    )

    eps = 1e-12
    KL_forward = float(np.sum(Pt * np.log((Pt + eps) / (Pm + eps))))
    KL_reverse = float(np.sum(Pm * np.log((Pm + eps) / (Pt + eps))))

    MSE = float(np.mean(D**2))
    ISE = float(area * np.sum(D**2))  # integrated squared error in continuous units

    out = dict(
        W2_x=float(w2_x),
        W2_y=float(w2_y),

        TV=float(_tv(Pt, Pm)),
        JS=float(_js(Pt, Pm)),
        KL_forward=KL_forward,
        KL_reverse=KL_reverse,
        MSE=MSE,
        ISE=ISE,

        mean_abs_delta=float(np.mean(np.abs(D))),
        MaxSquaredError=max_sq_err,
        MaxSquaredError_DensityContrib=float(np.max(D**2) / area),

        Pt_sum=float(Pt.sum()),
        Pm_sum=float(Pm.sum()),
        in_bounds_frac=in_bounds_frac,
        worst_cell=worst_cell,
    )

    # Two-sample tests (subsample)
    n = min(len(true_xy), len(gen_xy), two_sample_max_n)
    idxT = rng.choice(len(true_xy), n, replace=False)
    idxG = rng.choice(len(gen_xy), n, replace=False)
    X = true_xy[idxT]
    Y = gen_xy[idxG]

    out["EnergyDistance_stat"] = _energy_stat(X, Y)
    out["SlicedWasserstein"] = _sliced_wasserstein_2d(X, Y, n_proj=64, rng=rng)
    out["W2_2D_exact"] = w2_exact_subsample(true_xy, gen_xy, max_n=5000, seed=rng_seed)

    return out
