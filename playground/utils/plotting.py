import matplotlib.pyplot as plt
import numpy as np
import torch


def make_position_stats_heatmap(traj):
    first = traj["first_unmask_step"].float()
    first[first < 0] = float(traj["x_history"].shape[0] - 1)

    stats = torch.stack(
        [
            first.mean(dim=0),
            traj["finalization_step"].float().mean(dim=0),
            traj["num_remasks"].float().mean(dim=0),
            traj["num_unmasks"].float().mean(dim=0),
        ],
        dim=0,
    ).cpu().numpy()

    labels = ["first_unmask", "finalization", "num_remasks", "num_unmasks"]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(stats, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("token position")
    ax.set_title("ReMDM trajectory statistics by position")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def make_transition_event_heatmap(traj, sample_idx=0, mask_token=2):
    xh = traj["x_history"][:, sample_idx]
    prev = xh[:-1]
    cur = xh[1:]

    prev_mask = prev == mask_token
    cur_mask = cur == mask_token

    events = torch.zeros_like(prev)
    events[prev_mask & (~cur_mask)] = 1
    events[(~prev_mask) & cur_mask] = 2
    events[(~prev_mask) & (~cur_mask) & (prev != cur)] = 3

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(events.cpu().numpy(), aspect="auto", interpolation="nearest")
    ax.set_xlabel("token position")
    ax.set_ylabel("sampling step")
    ax.set_title(f"Transition events, sample {sample_idx}")
    fig.colorbar(im, ax=ax, label="0=no change, 1=reveal, 2=remask, 3=token change")
    fig.tight_layout()
    return fig


def plot_samples(ds, gen_xy, grid=400, levels=50, plot_raw=True):
    R = float(ds.R)

    if hasattr(ds, "device"):
        dist_dev = ds.device
    elif hasattr(ds, "dist"):
        try:
            dist_dev = ds.dist.mixture_distribution.probs.device
        except Exception:
            comp = ds.dist.component_distribution
            if hasattr(comp, "loc"):
                dist_dev = comp.loc.device
            else:
                base = getattr(comp, "base_dist", None)
                if base is not None and hasattr(base, "low"):
                    dist_dev = base.low.device
                else:
                    dist_dev = torch.device("cpu")
    else:
        dist_dev = torch.device("cpu")

    gen_xy = np.asarray(gen_xy, dtype=np.float64)

    if plot_raw:
        gen_uv = (gen_xy + R) / (2 * R)
        us = torch.linspace(0.0, 1.0, grid, device=dist_dev)
        uu, vv = torch.meshgrid(us, us, indexing="ij")
        coords_orig = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1) * (2 * R) - R

        lp = None
        if hasattr(ds, "dist"):
            with torch.no_grad():
                lp = ds.dist.log_prob(coords_orig).view(grid, grid).detach().cpu().numpy()

        uu_np = uu.detach().cpu().numpy()
        vv_np = vv.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        if lp is not None:
            ax.contour(uu_np, vv_np, lp, levels=levels)
        ax.scatter(gen_uv[:, 0], gen_uv[:, 1], s=6, c="red", alpha=0.35)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title("Generated samples (0-1 space)")
    else:
        xs = torch.linspace(-R, R, grid, device=dist_dev)
        xx, yy = torch.meshgrid(xs, xs, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

        lp = None
        if hasattr(ds, "dist"):
            with torch.no_grad():
                lp = ds.dist.log_prob(coords).view(grid, grid).detach().cpu().numpy()

        xx_np = xx.detach().cpu().numpy()
        yy_np = yy.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        if lp is not None:
            ax.contour(xx_np, yy_np, lp, levels=levels)
        ax.scatter(gen_xy[:, 0], gen_xy[:, 1], s=6, c="red", alpha=0.35)
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_aspect("equal")
        ax.set_title("Generated samples")

    plt.tight_layout()
    return fig
