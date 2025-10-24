import numpy as np
import torch
from numpy.random import default_rng
from scipy.stats import wasserstein_distance
from hyppo.ksample import MMD, Energy
from scipy.stats import norm

def get_true_samples(ds, n=60_000):
    if hasattr(ds, "dist"):
        with torch.no_grad():
            return ds.dist.sample((n,)).cpu().numpy()
    it = iter(ds)
    tokens = [next(it).numpy() for _ in range(n)]
    xy = np.vstack([ds.decode(seq) for seq in tokens])
    return xy

def _true_cell_mass_grid(ds, edges_x, edges_y):
    # exact cell mass by integrating the mixture CDF over rectangles
    # for diagonal Gaussians we can use Normal CDF separability.
    mix = ds.dist
    cat = mix.mixture_distribution
    comp = mix.component_distribution
    pis = cat.probs.detach().cpu().numpy()                  # [K]
    loc = comp.loc.detach().cpu().numpy()                   # [K,2]
    scale = comp.scale_tril.detach().cpu().numpy()          # [K,2,2] (diag)

    K = len(pis)
    Ex, Ey = edges_x, edges_y
    H = np.zeros((len(Ex)-1, len(Ey)-1), dtype=np.float64)

    # vectorized over grid for speed
    for k in range(K):
        mx, my = loc[k,0], loc[k,1]
        sx, sy = scale[k,0,0], scale[k,1,1]
        Fx = norm.cdf((Ex - mx)/sx)
        Fy = norm.cdf((Ey - my)/sy)
        Px = (Fx[1:, None] - Fx[:-1, None])                 # [Bx,1]
        Py = (Fy[None,1:] - Fy[None,:-1])                   # [1,By]
        H += pis[k] * (Px * Py)                              # outer -> [Bx,By]
    return H

def _gridify(samples, edges_x, edges_y, alpha=1.0):
    H, *_ = np.histogram2d(samples[:,0], samples[:,1], bins=[edges_x, edges_y])
    Bx, By = len(edges_x)-1, len(edges_y)-1
    P = (H + alpha) / (H.sum() + alpha * Bx * By)
    return P, H.sum()

def _tv(Pt, Pm, area):
    return 0.5 * np.sum(np.abs(Pt - Pm)) * area

def _js(P, Q, eps=1e-12):
    M = 0.5*(P+Q)
    def _kl(a,b): return np.sum(a*np.log((a+eps)/(b+eps)))
    return 0.5*_kl(P, M) + 0.5*_kl(Q, M)

def _sliced_wasserstein_2d(X, Y, n_proj=64, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    X = np.asarray(X); Y = np.asarray(Y)
    sw = 0.0
    for _ in range(n_proj):
        v = rng.normal(size=2); v /= np.linalg.norm(v) + 1e-12
        x = X @ v; y = Y @ v
        x = np.sort(x); y = np.sort(y)
        n = min(len(x), len(y))
        sw += np.mean((x[:n] - y[:n])**2)**0.5
    return sw / n_proj

def _median_heuristic_gamma(X, Y):
    Z = np.vstack([X, Y])
    # subsample for speed
    m = min(len(Z), 2000)
    idx = np.random.default_rng(0).choice(len(Z), m, replace=False)
    Z = Z[idx]
    D = np.linalg.norm(Z[:,None,:]-Z[None,:,:], axis=-1)
    med = np.median(D[D>0])
    gamma = 1.0/(2*(med**2 + 1e-12))
    return gamma

def _mmd_rbf_stat(X, Y):
    # hyppo has .statistic(X, Y), avoid permutations
    return float(MMD(compute_kernel="rbf").statistic(X, Y))

def _energy_stat(X, Y):
    return float(Energy().statistic(X, Y))

def divergence_metrics_plus(
    ds,
    gen_xy,
    n_true=60_000,
    grid=200,
    alpha=1.0,
    two_sample_max_n=10_000,
    rng_seed=2025,
):
    rng = np.random.default_rng(rng_seed)
    true_xy = get_true_samples(ds, n_true)

    # 1D W2 (marginals)
    w2_x = wasserstein_distance(gen_xy[:,0], true_xy[:,0])
    w2_y = wasserstein_distance(gen_xy[:,1], true_xy[:,1])

    # Grid (use exact Pt from analytic mixture)
    R = ds.R
    edges = np.linspace(-R, R, grid + 1)
    Pt = _true_cell_mass_grid(ds, edges, edges)              # exact reference
    Pm, _ = _gridify(gen_xy, edges, edges, alpha=alpha)
    area = (2*R)**2 / (grid*grid)
    Bx, By = Pt.shape
    Pt = (Pt + alpha) / (Pt.sum() + alpha * Bx * By)
    
        # --- Max squared error (mass space) + where it occurs ---
    D = Pt - Pm
    max_sq_err = float(np.max(D**2))
    imax = np.unravel_index(np.argmax(D**2), D.shape)
    # coords of that cell (useful for annotating heatmaps)
    ix, iy = imax
    edges = np.linspace(-R, R, grid + 1)
    worst_cell = dict(
        ix=int(ix), iy=int(iy),
        x_interval=(float(edges[ix]), float(edges[ix+1])),
        y_interval=(float(edges[iy]), float(edges[iy+1])),
        value=max_sq_err,
    )

    # Optional: per-cell *density* contribution variant (matches your density-MSE scaling)
    # Each cell contributes (Pt-Pm)^2 / area to the L2(p - q) integral.
    max_sq_err_density_contrib = float(np.max((D**2) / area))


    out = dict(
        W2_x=w2_x,
        W2_y=w2_y,
        TV=_tv(Pt, Pm, area),
        JS=_js(Pt, Pm),
        KL_forward=np.sum(Pm * np.log((Pm+1e-12)/(Pt+1e-12))),  # Jeffreys optional
        KL_reverse=np.sum(Pt * np.log((Pt+1e-12)/(Pm+1e-12))),
        MSE=np.sum((Pt - Pm)**2) * area,
        mean_abs_delta=np.mean(np.abs(Pt - Pm)),
        MaxSquaredError=max_sq_err,                           # mass-space L∞^2 on the grid
        MaxSquaredError_DensityContrib=max_sq_err_density_contrib, # useful for heatmaps
    )

    # Two-sample tests on matched subsamples
    n = min(len(true_xy), len(gen_xy), two_sample_max_n)
    idxT = rng.choice(len(true_xy), n, replace=False)
    idxG = rng.choice(len(gen_xy),  n, replace=False)
    X = true_xy[idxT]; Y = gen_xy[idxG]

    out["EnergyDistance_stat"] = _energy_stat(X, Y)
    # out["MMD_RBF_stat"] = _mmd_rbf_stat(X, Y)
    out["SlicedWasserstein"] = _sliced_wasserstein_2d(X, Y, n_proj=64, rng=rng)

    return out