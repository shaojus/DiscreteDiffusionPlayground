import os, sys
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch, numpy as np
from torch.utils.data import DataLoader
from tqdm import trange
import wandb
import matplotlib.pyplot as plt
import sys, inspect

from playground.data.gmm_binary_stream import GMMBinaryStream
from playground.data.checkerboard import CheckerboardBinaryStream
from playground.utils.metrics import divergence_metrics_plus

def _wandb_setup(cfg):
    if cfg.get('wandb', {}).get('enabled', True):
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=OmegaConf.to_container(cfg, resolve=True))
    else:
        os.environ['WANDB_DISABLED'] = 'true'

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def plot(ds, gen_xy, grid=400, levels=50, plot_raw=True):
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
        # Plot in 0-1 space (raw coordinates)
        gen_uv = (gen_xy + R) / (2 * R)

        # Grid in 0-1 space
        us = torch.linspace(0.0, 1.0, grid, device=dist_dev)
        uu, vv = torch.meshgrid(us, us, indexing="ij")

        # Convert grid to original space for log prob
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
        # Plot in mapped/original space
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


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg: DictConfig):
    torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device)
    L = 2 * int(cfg.data.n_bits)
    if cfg.data.dataset == "checkerboard":
        ds = CheckerboardBinaryStream(
            n_bits=cfg.data.n_bits,
            device=str(device)
    )
    elif cfg.data.dataset == "gmm":
        ds = GMMBinaryStream(
            n_mixes=cfg.data.n_mixes,
            n_bits=cfg.data.n_bits,
            device=str(device)
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset}")

    loader = DataLoader(ds, batch_size=cfg.train.batch_size)
    cfg.model.d_hid = 4 * cfg.model.d_model
    _wandb_setup(cfg)
    model = instantiate(cfg.model, max_len=L, device=device)
    print("[DEBUG] class:", model.__class__)
    print("[DEBUG] module:", model.__class__.__module__)
    mod = sys.modules[model.__class__.__module__]
    print("[DEBUG] file:", getattr(mod, "__file__", None), flush=True)

    n_params = int(count_params(model))
    print(f"[info] trainable params: {n_params:,}")

    # if using wandb
    wandb.config.update({"model/params": n_params}, allow_val_change=True)
    wandb.log({"model/params": n_params}, step=0)
    model = model.to(device)
    # opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, betas=(0.9, 0.95), weight_decay=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    it = iter(loader)
    model.train()
    for step in trange(int(cfg.train.steps)):
        x = next(it).to(device).long()
        loss = model.training_loss(x)
        opt.zero_grad(); loss.backward(); opt.step()

        if (step % int(cfg.log.every)) == 0:
            wandb.log({'loss': float(loss)}, step=step)

        if ((step + 1) % int(cfg.eval.interval)) == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                toks = model.sample(int(cfg.eval.n_samples), L, device)
                toks_np = toks.detach().cpu().numpy()
                gen_xy = np.stack([ds.decode(seq) for seq in toks_np], axis=0)
                mets = divergence_metrics_plus(ds, gen_xy)
                wandb.log({f'eval/{k}': v for k, v in mets.items()}, step=step)
                fig = plot(ds, gen_xy)
                wandb.log({'samples_plot': wandb.Image(fig)}, step=step)
                plt.close(fig)
            model.train()

    os.makedirs(os.path.dirname(cfg.train.save_path), exist_ok=True)
    torch.save({'model': model.state_dict(), 'cfg': OmegaConf.to_container(cfg, resolve=True)}, cfg.train.save_path)
    print(f"Saved checkpoint to {cfg.train.save_path}")

if __name__ == '__main__':
    main()
